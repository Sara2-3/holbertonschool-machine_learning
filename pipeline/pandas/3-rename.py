#!/usr/bin/env python3
"""
Module for renaming DataFrame columns
"""

import pandas as pd


def rename(df):
    """
    Renames Timestamp column to Datetime and converts values

    Args:
        df: pandas DataFrame

    Returns:
        Modified DataFrame with Datetime and Close columns
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    return df[['Datetime', 'Close']]
