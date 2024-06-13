import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
from datetime import datetime, timedelta

# Function to create a new model
def create_model(dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Try to load the model, or create a new one if it doesn't exist
try:
    model = load_model('lstm_stock_predictor.keras')
    scaler = joblib.load('scaler.pkl')
    model_info = joblib.load('model_info.pkl')
    st.sidebar.success("Loaded pre-trained model!")
except (OSError, IOError) as e:
    st.sidebar.warning("Pre-trained model not found. Using a new, untrained model.")
    st.sidebar.warning("Please train the model before making predictions.")
    model = create_model()
    scaler = MinMaxScaler(feature_range=(0, 1))
    model_info = {
        'ticker': 'AAPL',  # Default ticker
        'start_date': '2010-01-01',
        'end_date': datetime.now().strftime('%Y-%m-%d'),
        'sequence_length': 60,
        'best_params': {'dropout_rate': 0.2}
    }

# Extract info
sequence_length = model_info['sequence_length']

# Dictionary mapping company names to their tickers
company_dict = {
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'Amazon': 'AMZN',
    'Google': 'GOOGL',
    'Facebook': 'META',
    'Tesla': 'TSLA',
    'Netflix': 'NFLX'
}

# Function to get the ticker symbol for a given company name
def get_ticker(company_name):
    return company_dict.get(company_name, None)

# Function to get the latest stock data
def get_latest_data(ticker, start_date):
    try:
        data = yf.download(ticker, start=start_date)
        return data['Close'].fillna(method='ffill')  # Forward fill any NaN values
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.Series()

# Function to preprocess data for prediction
def preprocess_data(data):
    if len(data) == 0 or data.isnull().all():
        st.warning("Warning: Empty or invalid data received.")
        return None
    
    data = data.fillna(method='ffill').fillna(method='bfill')  # Forward and backward fill
    data = data.values.reshape(-1, 1)
    data = scaler.transform(data)
    X = []
    if len(data) >= sequence_length:
        X.append(data[-sequence_length:, 0])
    else:
        # If data is shorter than sequence_length, pad with data's mean
        padding = np.full((sequence_length - len(data), 1), data.mean())
        padded_data = np.vstack([padding, data])
        X.append(padded_data[:, 0])
    X = np.array(X).reshape(-1, sequence_length, 1)
    return X

# Function to predict next day's price
def predict_next_day(model, data):
    X = preprocess_data(data)
    if X is None:
        return None
    try:
        predicted = model.predict(X, verbose=0)
        return scaler.inverse_transform(predicted)[0, 0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Streamlit app layout
st.title('Stock Price Predictor')

# Sidebar for user input
st.sidebar.header('User Input')
company_name = st.sidebar.text_input("Enter a company name:")
days_to_predict = st.sidebar.slider('Days to Predict:', min_value=1, max_value=30, value=7)



# Get the ticker symbol for the given company name
ticker = get_ticker(company_name)

if ticker is None:
    st.warning(f"Unable to find the ticker symbol for '{company_name}'")
else:
    st.write(f"Predicting stock prices for {company_name} ({ticker})")

    # Get the latest data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=sequence_length + 365)).strftime('%Y-%m-%d')  # Get more data
    data = get_latest_data(ticker, start_date)

    if not data.empty:
        # Make predictions
        predictions = []
        last_actual_date = data.index[-1]

        for i in range(days_to_predict):
            # Get the last sequence_length days of actual data
            last_sequence = data.tail(sequence_length)
            
            # Make prediction
            next_day_pred = predict_next_day(model, last_sequence)
            if next_day_pred is not None:
                predictions.append(next_day_pred)
                
                # Create next business day's date
                next_day = last_actual_date + pd.offsets.BDay(i + 1)
                
                # Add the prediction to our dataset
                data.loc[next_day] = next_day_pred
            else:
                break

        # Create a date range for predictions (use business days)
        date_range = pd.bdate_range(start=last_actual_date + pd.offsets.BDay(1), periods=len(predictions))
        predictions_df = pd.Series(predictions, index=date_range, name='Predicted Price')

        # Prepare data for plotting
        plot_start_date = last_actual_date - pd.offsets.BDay(30)
        plot_data = data[data.index <= last_actual_date]
        plot_data = plot_data[plot_data.index >= plot_start_date]

        # Display the predictions
        st.subheader('Predicted Stock Prices for (USD):')
        st.write(predictions_df)

        # Plot the graph
        plt.figure(figsize=(12, 6))
        plt.plot(plot_data.index, plot_data, label='Historical Prices')
        plt.plot(predictions_df.index, predictions_df, label='Predicted Prices', linestyle='--', marker='o')
        plt.title(f'{company_name} ({ticker}) Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        # Calculate and display prediction confidence
        plot_data_no_nan = plot_data.dropna()
        if len(plot_data_no_nan) >= 2:  # Need at least 2 points for std
            last_30_days_std = np.std(plot_data_no_nan)
            confidence_level = 0.95
            z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}[confidence_level]
            margin_error = z_score * last_30_days_std / np.sqrt(len(plot_data_no_nan))

            st.subheader(f'Prediction Confidence ({confidence_level * 100}%)')
            st.write(f"Based on the stock's recent volatility:")
            for date, price in predictions_df.items():
                lower_bound = max(0, price - margin_error)  # Ensure non-negative price
                upper_bound = price + margin_error
                st.write(f"- {date.date()}: ${price:.2f} (Range: ${lower_bound:.2f} to ${upper_bound:.2f})")
        else:
            st.warning("Not enough data to calculate prediction confidence.")

# Display model info
st.sidebar.subheader('Model Information')
if 'start_date' in model_info and 'end_date' in model_info:
    st.sidebar.write(f"Trained on: {model_info['start_date']} to {model_info['end_date']}")
if 'best_params' in model_info:
    st.sidebar.write(f"Best Parameters:")
    for param, value in model_info['best_params'].items():
        st.sidebar.write(f"- {param}: {value}")

# Additional stock information
@st.cache_data
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info
    except Exception as e:
        st.error(f"Error fetching stock info: {e}")
        return {}

info = get_stock_info(ticker)
if info:
    st.sidebar.subheader('Stock Information')
    st.sidebar.write(f"Company: {info.get('longName', 'N/A')}")
    st.sidebar.write(f"Industry: {info.get('industry', 'N/A')}")
    st.sidebar.write(f"Market Cap: ${info.get('marketCap', 'N/A'):,.0f}")
    st.sidebar.write(f"52-Week Range: ${info.get('fiftyTwoWeekLow', 'N/A'):.2f} - ${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}")
else:
    st.error("Failed to fetch stock data. Please try again later.")

# Disclaimer

# Disclaimer
    st.warning('Disclaimer: This is a machine learning model based on historical data. '
          'Stock market predictions are subject to many external factors and uncertainties. '
          'The confidence intervals provided are based on recent volatility and do not '
          'account for unforeseen events. Use this information at your own risk and always '
          'consult with financial professionals before making investment decisions.')