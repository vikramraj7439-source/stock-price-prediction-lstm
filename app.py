"""
Alternative Simplified Streamlit App
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

st.title("📈 Stock Price Prediction System")
st.write("LSTM-based Stock Market Prediction")

# Load model
@st.cache_resource
def load_resources():
    model = load_model("stock_lstm_model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_resources()
    st.success("✓ Model and Scaler loaded successfully!")
except:
    st.error("❌ Please run 'stock_lstm.ipynb' first to train the model!")
    st.stop()

# Stock selection
stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
selected_stock = st.selectbox("Select Stock:", stocks)

# Load data
try:
    df = pd.read_csv(f'data/{selected_stock}.csv', index_col=0, parse_dates=True)
    
    # Show current price
    current_price = df['Close'].iloc[-1]
    st.metric("Current Price", f"₹{current_price:.2f}")
    
    # Plot historical data
    st.subheader("Historical Prices")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index[-100:], df['Close'].iloc[-100:], linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    
    # Predict
    if st.button("🔮 Predict Next 7 Days"):
        data = df[['Close']].values
        scaled_data = scaler.transform(data)
        last_60 = scaled_data[-60:]
        
        predictions = []
        current_batch = last_60.copy()
        
        for i in range(7):
            pred_scaled = model.predict(current_batch.reshape(1, 60, 1), verbose=0)
            pred_price = scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(pred_price)
            current_batch = np.append(current_batch[1:], pred_scaled)
        
        st.subheader("📅 Next 7 Days Predictions")
        for i, price in enumerate(predictions, 1):
            st.write(f"Day {i}: ₹{price:.2f}")
            
        # Alert
        change = ((predictions[0] - current_price) / current_price) * 100
        if change > 3:
            st.success(f"🟢 BUY Signal (+{change:.2f}%)")
        elif change < -3:
            st.error(f"🔴 SELL Signal ({change:.2f}%)")
        else:
            st.warning(f"🟡 HOLD Signal ({change:+.2f}%)")
    
except Exception as e:
    st.error(f"Error: {e}")

st.info("💡 This is a simplified version. Run 'streamlit run main.py' for full features!")
