# Time Series Forecasting — BTC Price Prediction

## Description

This project implements an end-to-end deep learning pipeline for
Bitcoin (BTC) price forecasting using Recurrent Neural Networks.
Using minute-level historical data from Coinbase and Bitstamp exchanges,
the model uses a 24-hour sliding window to predict the BTC closing
price one hour ahead.

## Dataset

- **Coinbase:** `coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv`
- **Bitstamp:** `bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv`
- **Frequency:** 1 minute per row
- **Columns:** Timestamp, Open, High, Low, Close, Volume (BTC),
  Volume (USD), Weighted Price

## Requirements

- Python 3.9
- TensorFlow 2.15
- NumPy 1.25.2
- Pandas 2.2.2
- scikit-learn

## Installation
