#!/usr/bin/env python3
"""
Module for concat
"""
import pandas as pd


def concat(df1, df2):
    """
    Concatenates two dataframes with specific conditions

    Args:
        df1: First dataframe (coinbase)
        df2: Second dataframe (bitstamp)

    Returns:
        Concatenated dataframe with keys
    """
    # Import the index function
    index = __import__('10-index').index

    # Index both dataframes on their Timestamp columns
    df1_indexed = index(df1, 'Timestamp')
    df2_indexed = index(df2, 'Timestamp')

    # Select rows from df2 up to and including timestamp 1417411920
    df2_selected = df2_indexed[df2_indexed.index <= 1417411920]

    # Concatenate with keys
    result = pd.concat([df2_selected, df1_indexed],
                       keys=['bitstamp', 'coinbase'])

    return result
