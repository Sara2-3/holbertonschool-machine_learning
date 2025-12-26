#!/usr/bin/env python3
"""
Module for Flip it and Switch it
"""


def flip_switch(df):
    """Sorts data in reverse order and transposes it
    """
    df_sorted = df.sort_index(ascending=False)

    # Transpose the sorted dataframe
    df_transposed = df_sorted.T

    return df_transposed
