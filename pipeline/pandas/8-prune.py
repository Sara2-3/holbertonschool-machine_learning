#!/usr/bin/env python3
"""
Module for Prune
"""


def prune(df):
    """Removes any entries where Close has NaN values.
    """
    return df.dropna(subset=['Close'])
