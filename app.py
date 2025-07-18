import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="📈 LSTM Stock Price Predictor", layout="centered")
st.title("📉 LSTM Stock Price Predictor")
st.markdown("Predict the next day's stock price using an LSTM model.")

# File upload
uploaded_file = st.file_uploader("Upload a Stock CSV File (e.g., AAPL.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if 'Close' not in df.columns:
        st.error("The uploaded CSV must contain a 'Close' column.")
    else:
        # Display data
        st.subheader("Raw Data")
        st.write(df.tail())

        # Prepare data
        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        last_60_days = scaled_data[-60:]
        X_input = np.array([last_60_days])  # shape = (1, 60, 1)

        # Load model
        model = load_model("lstm_stock_model.h5")

        # Prediction
        predicted_scaled_price = model.predict(X_input)
        predicted_price = scaler.inverse_transform(predicted_scaled_price)

        st.subheader("📈 Predicted Next Day Price")
        st.success(f"${predicted_price[0][0]:.2f}")

        # Optional Plot
        st.subheader("📊 Closing Prices")
        fig, ax = plt.subplots()
        ax.plot(df['Close'], label='Close Price')
        ax.set_xlabel("Days")
        ax.set_ylabel("Price ($)")
        ax.legend()
        st.pyplot(fig)

else:
    st.info("Please upload a stock CSV file to get started.")
