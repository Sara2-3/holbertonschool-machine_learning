#!/usr/bin/env python3
"""
Module for creating pandas DataFrames from files
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Function that loads data from a file

    Args:
        filename (str): The file to load from
        delimiter (str): The column separator

    Returns:
        pd.DataFrame: The loaded DataFrame
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
