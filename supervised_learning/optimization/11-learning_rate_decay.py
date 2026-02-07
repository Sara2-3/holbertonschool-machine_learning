#!/usr/bin/env python3
"""
Module 11-learning_rate_decay
Provides a function to update the learning rate using
inverse time decay in NumPy.
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay.

    Parameters:
    alpha : float
        original learning rate
    decay_rate : float
        rate at which alpha decays
    global_step : int
        number of passes of gradient descent that have elapsed
    decay_step : int
        number of passes of gradient descent before alpha decays further

    Returns:
    float
        updated learning rate
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
