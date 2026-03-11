import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction & Alert System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #065A82;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1C7293;
        font-weight: bold;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #065A82;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Cache functions
@st.cache_resource
def load_model_and_scaler():
    """Load trained model and scaler"""
    try:
        model = load_model("stock_lstm_model.h5",compile=False) ## added compile=false 
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_data
def load_stock_data(stock_name):
    try:
        df = pd.read_csv(f'data/{stock_name}.csv', index_col=0)

        # Convert index to datetime
        df.index = pd.to_datetime(df.index, errors='coerce')

        # Ensure index name is Date
        df.index.name = "Date"

        # Convert numeric columns
        cols = ['Open','High','Low','Close','Volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Remove invalid rows
        df = df.dropna(subset=['Close'])

        return df

    except Exception as e:
        st.error(f"Error loading {stock_name} data: {e}")
        return None

def predict_future(model, scaler, data, days=7):
    """Predict future stock prices"""
    # Scale data
    scaled_data = scaler.transform(data)
    
    # Use last 60 days
    last_60_days = scaled_data[-60:]
    
    # Predict future
    predictions = []
    current_batch = last_60_days.copy()
    
    for i in range(days):
        current_batch_reshaped = current_batch.reshape(1, 60, 1)
        pred_scaled = model.predict(current_batch_reshaped, verbose=0)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        predictions.append(pred_price)
        
        # Update batch
        current_batch = np.append(current_batch[1:], pred_scaled)
    
    return predictions

def generate_alert(current_price, predicted_price, threshold=3.0):
    """Generate price alert based on prediction"""
    price_change = ((predicted_price - current_price) / current_price) * 100
    
    if price_change >= threshold:
        return "BUY", price_change, "success"
    elif price_change <= -threshold:
        return "SELL", price_change, "danger"
    else:
        return "HOLD", price_change, "warning"

# Main App
def main():
    # Header
    st.markdown('<p class="main-header">📈 Stock Price Prediction & Automated Alert Engine</p>', unsafe_allow_html=True)
    st.markdown("### ML-Assisted Real-Time Stock Monitoring System")
    st.markdown("---")
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        st.error("⚠️ Model or scaler not found! Please run 'stock_lstm.ipynb' first to train the model.")
        return
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    st.sidebar.markdown("---")
    
    # Stock selection
    nifty50_stocks = [
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
        "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
        "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND",
        "WIPRO", "ONGC", "NTPC", "POWERGRID", "ADANIENT"
    ]
    
    selected_stock = st.sidebar.selectbox(
        "📊 Select Stock",
        nifty50_stocks,
        index=0
    )
    
    # Prediction days
    prediction_days = st.sidebar.slider(
        "🔮 Prediction Period (Days)",
        min_value=1,
        max_value=30,
        value=7,
        step=1
    )
    
    # Alert threshold
    alert_threshold = st.sidebar.slider(
        "⚠️ Alert Threshold (%)",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip:** Adjust the alert threshold to customize sensitivity of buy/sell signals.")
    
    # Load stock data
    df = load_stock_data(selected_stock)
    
    if df is None:
        st.error(f"⚠️ Data for {selected_stock} not found!")
        return
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    with col1:
        st.metric(
            label="💵 Current Price",
            value=f"₹{current_price:.2f}",
            delta=f"{price_change_pct:.2f}%"
        )
    
    with col2:
        st.metric(
            label="📊 Volume",
            value=f"{df['Volume'].iloc[-1]/1e6:.2f}M"
        )
    
    with col3:
        st.metric(
            label="📈 High (Today)",
            value=f"₹{df['High'].iloc[-1]:.2f}"
        )
    
    with col4:
        st.metric(
            label="📉 Low (Today)",
            value=f"₹{df['Low'].iloc[-1]:.2f}"
        )
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Predictions", "📊 Analysis", "⚠️ Alerts", "📋 Data"])
    
    # Tab 1: Predictions
    with tab1:
        st.markdown('<p class="sub-header">Future Price Predictions</p>', unsafe_allow_html=True)
        
        if st.button("🔮 Generate Predictions", type="primary"):
            with st.spinner("Analyzing historical patterns and generating predictions..."):
                # Predict
                data = df[['Close']].values
                future_prices = predict_future(model, scaler, data, prediction_days)
                
                # Create future dates
                last_date = df.index[-1]
                future_dates = pd.date_range(
                    start=last_date + timedelta(days=1),
                    periods=prediction_days,
                    freq='D'
                )
                
                # Create prediction dataframe
                pred_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': future_prices
                })
                
                # Display predictions
                col_pred1, col_pred2 = st.columns([2, 1])
                
                with col_pred1:
                    # Plot
                    fig = go.Figure()
                    
                    # Historical data (last 90 days)
                    hist_data = df.tail(90)
                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['Close'],
                        mode='lines',
                        name='Historical',
                        line=dict(color='#065A82', width=2)
                    ))
                    
                    # Predictions
                    fig.add_trace(go.Scatter(
                        x=pred_df['Date'],
                        y=pred_df['Predicted Price'],
                        mode='lines+markers',
                        name='Predicted',
                        line=dict(color='#FF6B6B', width=2, dash='dash'),
                        marker=dict(size=8)
                    ))
                    
                    fig.update_layout(
                        title=f'{selected_stock} - Price Prediction',
                        xaxis_title='Date',
                        yaxis_title='Price (₹)',
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_pred2:
                    st.markdown("#### 📅 Prediction Table")
                    for i, row in pred_df.iterrows():
                        day_num = i + 1
                        date_str = row['Date'].strftime('%Y-%m-%d')
                        price = row['Predicted Price']
                        
                        st.markdown(f"**Day {day_num}** ({date_str})")
                        st.markdown(f"₹{price:.2f}")
                        st.markdown("---")
    
    # Tab 2: Analysis
    with tab2:
        st.markdown('<p class="sub-header">Historical Analysis</p>', unsafe_allow_html=True)
        
        # Time period selection
        period = st.selectbox(
            "Select Time Period",
            ["1 Month", "3 Months", "6 Months", "1 Year", "All Time"],
            index=3
        )
        
        # Filter data based on period
        if period == "1 Month":
            plot_df = df.tail(30)
        elif period == "3 Months":
            plot_df = df.tail(90)
        elif period == "6 Months":
            plot_df = df.tail(180)
        elif period == "1 Year":
            plot_df = df.tail(365)
        else:
            plot_df = df
        
        # Candlestick chart
        fig_candle = go.Figure(data=[go.Candlestick(
            x=plot_df.index,
            open=plot_df['Open'],
            high=plot_df['High'],
            low=plot_df['Low'],
            close=plot_df['Close'],
            name='OHLC'
        )])
        
        fig_candle.update_layout(
            title=f'{selected_stock} - Candlestick Chart',
            yaxis_title='Price (₹)',
            xaxis_title='Date',
            height=500
        )
        
        st.plotly_chart(fig_candle, use_container_width=True)
        
        # Volume chart
        fig_volume = px.bar(
            plot_df.reset_index(),
            x='Date',
            y='Volume',
            title=f'{selected_stock} - Trading Volume',
            labels={'Volume': 'Volume'},
            color_discrete_sequence=['#1C7293']
        )
        
        fig_volume.update_layout(height=300)
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # Tab 3: Alerts
    with tab3:
        st.markdown('<p class="sub-header">Automated Price Alerts</p>', unsafe_allow_html=True)
        
        if st.button("🔔 Generate Alert", type="primary"):
            with st.spinner("Analyzing price movements..."):
                # Predict next day
                data = df[['Close']].values
                next_day_price = predict_future(model, scaler, data, days=1)[0]
                
                # Generate alert
                action, change_pct, alert_type = generate_alert(
                    current_price,
                    next_day_price,
                    alert_threshold
                )
                
                # Display alert
                if alert_type == "success":
                    st.markdown(
                        f'<div class="alert-box alert-success">'
                        f'<h3>🟢 {action} SIGNAL</h3>'
                        f'<p><strong>Current Price:</strong> ₹{current_price:.2f}</p>'
                        f'<p><strong>Predicted Price (Next Day):</strong> ₹{next_day_price:.2f}</p>'
                        f'<p><strong>Expected Change:</strong> +{change_pct:.2f}%</p>'
                        f'<p>💡 <em>The model predicts an upward price movement. Consider buying.</em></p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                elif alert_type == "danger":
                    st.markdown(
                        f'<div class="alert-box alert-danger">'
                        f'<h3>🔴 {action} SIGNAL</h3>'
                        f'<p><strong>Current Price:</strong> ₹{current_price:.2f}</p>'
                        f'<p><strong>Predicted Price (Next Day):</strong> ₹{next_day_price:.2f}</p>'
                        f'<p><strong>Expected Change:</strong> {change_pct:.2f}%</p>'
                        f'<p>⚠️ <em>The model predicts a downward price movement. Consider selling.</em></p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="alert-box alert-warning">'
                        f'<h3>🟡 {action} SIGNAL</h3>'
                        f'<p><strong>Current Price:</strong> ₹{current_price:.2f}</p>'
                        f'<p><strong>Predicted Price (Next Day):</strong> ₹{next_day_price:.2f}</p>'
                        f'<p><strong>Expected Change:</strong> {change_pct:+.2f}%</p>'
                        f'<p>📊 <em>The model predicts minimal price movement. Consider holding.</em></p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                # Additional metrics
                st.markdown("### 📊 Supporting Indicators")
                
                col_ind1, col_ind2, col_ind3 = st.columns(3)
                
                # Calculate simple moving averages
                sma_20 = df['Close'].tail(20).mean()
                sma_50 = df['Close'].tail(50).mean()
                
                with col_ind1:
                    st.metric("20-Day SMA", f"₹{sma_20:.2f}")
                
                with col_ind2:
                    st.metric("50-Day SMA", f"₹{sma_50:.2f}")
                
                with col_ind3:
                    trend = "Bullish" if current_price > sma_50 else "Bearish"
                    st.metric("Trend", trend)
    
    # Tab 4: Data
    with tab4:
        st.markdown('<p class="sub-header">Historical Data</p>', unsafe_allow_html=True)
        
        # Display dataframe
        st.dataframe(
            df.tail(100).sort_index(ascending=False),
            use_container_width=True,
            height=500
        )
        
        # Download button
        csv = df.to_csv()
        st.download_button(
            label="📥 Download Full Dataset",
            data=csv,
            file_name=f'{selected_stock}_historical_data.csv',
            mime='text/csv'
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p><strong>ML-Assisted Stock Price Monitoring & Automated Alert Engine</strong></p>
            <p>Powered by LSTM Neural Networks | Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
