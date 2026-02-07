#!/usr/bin/env python3
"""
Module 13-batch_norm
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network
    code using batch normalization.

    Parameters:
    Z : numpy.ndarray of shape (m, n)
        m = number of data points
        n = number of features in Z
    gamma : numpy.ndarray of shape (1, n)
        scales used for batch normalization
    beta : numpy.ndarray of shape (1, n)
        offsets used for batch normalization
    epsilon : float
        small number to avoid division by zero

    Returns:
    numpy.ndarray of shape (m, n)
        normalized Z matrix
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(var + epsilon)
    return gamma * Z_norm + beta
