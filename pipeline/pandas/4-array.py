#!/usr/bin/env python3
"""
Module for converting DataFrame to numpy array
"""


def array(df):
    """
    Selects last 10 rows of High and Close columns and converts to numpy array
    """
    return df[['High', 'Close']].tail(10).values
