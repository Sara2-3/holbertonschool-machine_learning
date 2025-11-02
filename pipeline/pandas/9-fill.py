#!/usr/bin/env python3
"""
Module for Filling
"""


def fill(df):
    """
    Removes Weighted_Price column and fills missing values
    """
    df = df.drop('Weighted_Price', axis=1)
    df['Close'] = df['Close'].fillna(method='ffill')
    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'])
    for col in ['Volume_(BTC)', 'Volume_(Currency)']:
        df[col] = df[col].fillna(0)
    return df
