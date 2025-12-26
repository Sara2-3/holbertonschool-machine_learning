#!/usr/bin/env python3
"""
Module for Sorting
"""


def high(df):
    """Sorts it by the High price in descending order.
    """
    return df.sort_values('High', ascending=False)
