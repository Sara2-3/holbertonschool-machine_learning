#!/usr/bin/env python3
"""
Module for converting DataFrame to numpy array
"""

import pandas as pd
import numpy as np


def array(df):
    """
    Selects last 10 rows of High and Close columns and converts to numpy array

    Args:
        df: pandas DataFrame with High and Close columns

    Returns:
        numpy.ndarray: Array with last 10 rows of High and Close values
    """
    selected_data = df[['High', 'Close']].tail(10)
    numpy_array = selected_data.to_numpy()
    return numpy_array
