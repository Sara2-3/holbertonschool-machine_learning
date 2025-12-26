#!/usr/bin/env python3
"""
Module for Analyzing
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns
    except the Timestamp column.
    """
    return df.drop('Timestamp', axis=1).describe()
