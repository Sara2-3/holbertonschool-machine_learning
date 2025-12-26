#!/usr/bin/env python3
"""
Module for hierarchy
"""
import pandas as pd


def hierarchy(df1, df2):
    """
    Rearranges MultiIndex and concatenates specific timestamp range

    Args:
        df1: First dataframe (coinbase)
        df2: Second dataframe (bitstamp)

    Returns:
        Concatenated pd.DataFrame with rearranged MultiIndex
    """
    # Import the index function
    index = __import__('10-index').index

    # Index both dataframes on their Timestamp columns
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    # Select rows from both dataframes in timestamp range
    # 1417411980 to 1417417980 inclusive
    df1_selected = df1_indexed[(df1_indexed.index >= 1417411980) &
                               (df1_indexed.index <= 1417417980)]
    df2_selected = df2_indexed[(df2_indexed.index >= 1417411980) &
                               (df2_indexed.index <= 1417417980)]

    # Concatenate with keys
    result = pd.concat([df2_selected, df1_selected],
                       keys=['bitstamp', 'coinbase'])

    # Rearrange MultiIndex so Timestamp is the first level
    result = result.swaplevel(0, 1)

    # Sort by Timestamp to ensure chronological order
    result = result.sort_index(level=0)

    return result
