#!/usr/bin/env python
# coding: utf-8

# In[4]:


import aiohttp
import asyncio
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import StandardScaler
import pytz  # Import pytz to handle time zones

# Load saved models and scaler
deep_model = load_model('deep_model.h5')
rf_model = joblib.load('rf_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')
lr_model = joblib.load('lr_model.pkl')
scaler = joblib.load('scaler.pkl')

# Helper functions for feature extraction
def get_column_from_spin(spin):
    if (spin - 1) % 3 == 0:
        return 1  # 1st column
    elif (spin - 2) % 3 == 0:
        return 2  # 2nd column
    else:
        return 3  # 3rd column

def extract_categorical_features(number):
    features = {}
    features['even_odd'] = 1 if number % 2 == 0 else 0  # 1 = Even, 0 = Odd
    features['high_low'] = 1 if 19 <= number <= 36 else 0  # 1 = High, 0 = Low (1-18)
    features['dozen'] = (number - 1) // 12 + 1  # 1 = 1-12, 2 = 13-24, 3 = 25-36
    return features

def extract_previous_spins_features(data, sequence_length=10):
    features = {}
    if len(data) >= sequence_length:
        last_n_spins = data[-sequence_length:]
        features['rolling_mean'] = np.mean(last_n_spins)
        features['rolling_std'] = np.std(last_n_spins)
        features['rolling_min'] = np.min(last_n_spins)
        features['rolling_max'] = np.max(last_n_spins)
    return features

def create_features(df, sequence_length=10):
    categorical_features = df['result'].apply(extract_categorical_features).apply(pd.Series)
    
    # Create rolling features manually
    rolling_features = []
    for i in range(sequence_length, len(df)):
        spins_sequence = df['result'].iloc[i-sequence_length:i]
        features = extract_previous_spins_features(spins_sequence, sequence_length)
        rolling_features.append(features)
    
    rolling_features_df = pd.DataFrame(rolling_features)
    
    # Merge categorical and rolling features
    df_features = pd.concat([df.iloc[sequence_length:], categorical_features, rolling_features_df], axis=1)
    
    # Drop NaN values (shouldn't be any after the rolling features extraction)
    df_features = df_features.dropna()
    
    return df_features

# Function to fetch data from the API
async def fetch_data():
    url = "https://api.tracksino.com/lightningroulette_history"
    headers = {
        "Authorization": "Bearer 35423482-f852-453c-97a4-4f5763f4796f",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://tracksino.com"
    }
    params = {
        "sort_by": "",
        "sort_desc": "false",
        "per_page": 100,  # 100 results per page
        "period": "1month",
        "table_id": 4,
        "page": 1  # Fetch only page 1
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return [{'when': entry['when'], 'result': entry['result']} for entry in data.get('data', [])]
            else:
                print(f"Error fetching data: {response.status}")
                return []

# Function to fetch and process the data
async def get_processed_data():
    data = await fetch_data()
    if data:
        # Convert the fetched data into a DataFrame
        df = pd.DataFrame(data)
        
        # We do not need to reverse the data, as the most recent spin is already the first row.
        df['Datetime'] = pd.to_datetime(df['when'], unit='s')
        
        # Convert to IST (Indian Standard Time)
        df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        
        df['Hour'] = df['Datetime'].dt.hour
        df['Minute'] = df['Datetime'].dt.minute
        df['Second'] = df['Datetime'].dt.second
        df['Weekday'] = df['Datetime'].dt.weekday
        df['Month'] = df['Datetime'].dt.month
        df['Column'] = df['result'].apply(get_column_from_spin)
        df['Column 11th Spin'] = df['Column'].shift(-10)
        df = df.dropna(subset=['Column 11th Spin'])

        # Create a column with formatted timestamp (IST)
        df['Formatted Time'] = df['Datetime'].dt.strftime('%H:%M:%S')

        df_features = create_features(df)
        X_input = df_features[['result', 'Hour', 'Minute', 'Second', 'Weekday', 'Month', 
                               'even_odd', 'high_low', 'dozen', 
                               'rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max']]
        X_input_scaled = scaler.transform(X_input)
        
        return df, X_input_scaled  # Return the data for further processing
    else:
        return None, None

# Streamlit interface
st.title("Lightning Roulette Prediction")

# Add a button to trigger prediction
if st.button('Get Prediction'):
    # Fetch the data and make predictions
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    df, X_input_scaled = loop.run_until_complete(get_processed_data())

    if df is not None and X_input_scaled is not None:
        # Display the last 10 results (most recent spin first)
        last_10_results = df[['Formatted Time', 'result']].head(10)  # The API already returns the most recent first
        st.write("Latest 10 Results (Most Recent First in IST):")
        st.dataframe(last_10_results)

        # Predict using the deep learning model
        dl_probs = deep_model.predict(X_input_scaled)
        rf_probs = rf_model.predict_proba(X_input_scaled)
        xgb_probs = xgb_model.predict_proba(X_input_scaled)
        lr_probs = lr_model.predict_proba(X_input_scaled)

        # Combine the probabilities using soft voting (average)
        avg_probs = (dl_probs + rf_probs + xgb_probs + lr_probs) / 4

        # Get the top 2 predictions
        top_2_predictions = np.argsort(avg_probs, axis=1)[:, -2:]  # Get indices of top 2 predictions

        # Convert to 1-indexed labels
        top_2_predictions = top_2_predictions + 1

        # Column Mapping
        column_mapping = {1: '1st Column', 2: '2nd Column', 3: '3rd Column'}
        top_2_mapped = [column_mapping[pred] for pred in top_2_predictions[0]]

        st.write(f"Predicted Top 2 Columns for the 11th Spin: {', '.join(top_2_mapped)}")
    else:
        st.write("Error fetching or processing data.")


# In[ ]:




