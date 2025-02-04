import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import os

# Function to train and save the model
def train_model(prices):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(np.array(prices).reshape(-1,1))

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=1, epochs=1)

    model.save("bitcoin_price_model.h5")
    return scaler

# Function to get real-time Bitcoin price
def get_bitcoin_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    response = requests.get(url).json()
    return response['bitcoin']['usd']

# Function to generate Bitcoin price predictions
def predict_price(data, scaler):
    scaled_data = scaler.transform(np.array(data).reshape(-1,1))
    scaled_data = np.reshape(scaled_data, (1, scaled_data.shape[0], 1))
    model = load_model("bitcoin_price_model.h5")
    prediction = model.predict(scaled_data)
    return scaler.inverse_transform(prediction)[0][0]

# Streamlit UI
st.title("Bitcoin Price Predictor")

# Get real-time price
current_price = get_bitcoin_price()
st.metric(label="Current Bitcoin Price (USD)", value=f"${current_price:.2f}")

# Load historical data
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90"
data = requests.get(url).json()['prices']
prices = [item[1] for item in data]

# Check if model exists, otherwise train it
if not os.path.exists("bitcoin_price_model.h5"):
    st.info("Training model... This might take a few moments.")
    scaler = train_model(prices)
    st.success("Model trained and saved successfully!")
else:
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit_transform(np.array(prices).reshape(-1,1))

# User Input for Prediction Days
days = st.slider("Select prediction timeframe (days):", 1, 30, 7)

# Predict future price
predicted_price = predict_price(prices[-60:], scaler)
st.success(f"Predicted Bitcoin Price in {days} days: ${predicted_price:.2f}")

# Plot historical and predicted prices
fig = go.Figure()
fig.add_trace(go.Scatter(y=prices, x=list(range(len(prices))), mode='lines', name='Historical Price'))
fig.add_trace(go.Scatter(y=[None]*(len(prices)-1) + [predicted_price], x=list(range(len(prices))),
                         mode='markers', marker=dict(color='red', size=10), name='Predicted Price'))
st.plotly_chart(fig)

# Run the app using: streamlit run app.py
