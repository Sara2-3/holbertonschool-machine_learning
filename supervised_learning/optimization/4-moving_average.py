#!/usr/bin/env python3
"""
Module 4-moving_average
Provides a function to calculate the weighted moving average of a dataset
with bias correction.
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a dataset.

    Parameters:
    data : list
        list of data points
    beta : float
        weight used for the moving average (between 0 and 1)

    Returns:
    list of floats
        the moving averages of data
    """
    averages = []
    v = 0  # running weighted average
    for t, x in enumerate(data, start=1):
        v = beta * v + (1 - beta) * x
        # bias correction
        v_corrected = v / (1 - beta ** t)
        averages.append(v_corrected)
    return averages
