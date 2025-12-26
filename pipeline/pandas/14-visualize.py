#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

# Use a shorter path or split the long string
file_path = (
    "C:/Users/User/Downloads/"
    "coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"
)
df = from_file(file_path, ',')

# YOUR CODE HERE

# Remove Weighted_Price column
df = df.drop('Weighted_Price', axis=1)

# Rename Timestamp to Date
df = df.rename(columns={'Timestamp': 'Date'})

# Convert timestamp values to date values
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index the dataframe on Date
df = df.set_index('Date')

# Handle missing values
# Missing values in Close should be set to the previous row value
df['Close'] = df['Close'].fillna(method='ffill')

# Missing values in High, Low, Open should be set to same row's Close value
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])

# Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Filter data from 2017 and beyond
df_2017 = df[df.index >= '2017-01-01']

# Resample to daily intervals and aggregate
daily_df = df_2017.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Return the transformed DataFrame before plotting
print(daily_df)
