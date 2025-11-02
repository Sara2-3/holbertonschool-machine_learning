#!/usr/bin/env python3
"""
Module for Flip it and Switch it
"""


def flip_switch(df):
    """Sorts data in reverse order and transposes it
    """
    return df.sort_values('Timestamp', ascending=False).T
