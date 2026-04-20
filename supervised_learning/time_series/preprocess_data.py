#!/usr/bin/env python3
"""
Module for preprocessing BTC time series data.

This module loads, cleans, merges, normalizes and creates
sliding windows from Coinbase and Bitstamp datasets.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_and_clean(filepath):
    """
    Load a CSV file and clean the Close price column.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame with Timestamp and Close.
    """
    df = pd.read_csv(filepath)
    df = df[['Timestamp', 'Close']].copy()
    df['Close'] = df['Close'].ffill()
    df.dropna(inplace=True)
    return df


def merge_datasets(coinbase_path, bitstamp_path):
    """
    Merge Coinbase and Bitstamp by averaging Close prices.

    Args:
        coinbase_path (str): Path to Coinbase CSV file.
        bitstamp_path (str): Path to Bitstamp CSV file.

    Returns:
        pd.DataFrame: Merged DataFrame with Timestamp and Close.
    """
    cb = load_and_clean(coinbase_path)
    bs = load_and_clean(bitstamp_path)

    merged = pd.merge(cb, bs, on='Timestamp',
                      suffixes=('_cb', '_bs'))
    merged['Close'] = merged[['Close_cb', 'Close_bs']].mean(axis=1)
    merged = merged[['Timestamp', 'Close']]
    merged = merged.sort_values('Timestamp').reset_index(drop=True)

    return merged


def create_windows(data, window_size=24):
    """
    Create sliding windows from time series data.

    Args:
        data (np.ndarray): Normalized time series data.
        window_size (int): Number of timesteps per window.

    Returns:
        tuple: X (samples, window_size, 1) and y (samples,) arrays.
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y


def preprocess(coinbase_path, bitstamp_path,
               output_path='data/preprocessed.npz'):
    """
    Full preprocessing pipeline for BTC forecasting.

    Loads, merges, resamples, normalizes and saves the data
    as sliding windows split into train, validation and test sets.

    Args:
        coinbase_path (str): Path to Coinbase CSV file.
        bitstamp_path (str): Path to Bitstamp CSV file.
        output_path (str): Path to save the preprocessed .npz file.
    """
    print("Loading and merging datasets...")
    df = merge_datasets(coinbase_path, bitstamp_path)
    print("Merged shape: {}".format(df.shape))

    df_hourly = df.iloc[::60].reset_index(drop=True)
    print("Hourly rows: {}".format(len(df_hourly)))

    scaler = MinMaxScaler()
    df_hourly['Close_scaled'] = scaler.fit_transform(
        df_hourly[['Close']]
    )

    data = df_hourly['Close_scaled'].values
    X, y = create_windows(data, window_size=24)

    print("X shape: {}".format(X.shape))
    print("y shape: {}".format(y.shape))

    train_size = int(len(X) * 0.70)
    val_size = int(len(X) * 0.20)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    print("Train:      {:>10,} samples".format(len(X_train)))
    print("Validation: {:>10,} samples".format(len(X_val)))
    print("Test:       {:>10,} samples".format(len(X_test)))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path,
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)

    min_path = 'data/scaler_min.npy'
    scale_path = 'data/scaler_scale.npy'
    np.save(min_path, scaler.data_min_)
    np.save(scale_path, scaler.scale_)

    print("\nSaved -> {}".format(output_path))


if __name__ == '__main__':
    cb = 'data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    bs = 'data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
    preprocess(coinbase_path=cb, bitstamp_path=bs)
