#!/usr/bin/env python3
"""
Module for creating pandas DataFrames from numpy arrays
"""

import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray

    Args:
        array (np.ndarray): The numpy array from which to create the DataFrame

    Returns:
        pd.DataFrame: Newly created DataFrame with columns A, B, etc.
    """
    num_cols = array.shape[1]
    
    # Create column labels without using string module
    columns = [chr(65 + i) for i in range(num_cols)]  # 65 is 'A' in ASCII
    
    df = pd.DataFrame(array, columns=columns)
    
    return df
