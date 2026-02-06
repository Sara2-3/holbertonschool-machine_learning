#!/usr/bin/env python3
"""
Module 2-shuffle_data
Provides a function to shuffle two datasets in the same way,
preserving the correspondence between their rows.
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Parameters:
    X : numpy.ndarray of shape (m, nx)
        m = number of data points
        nx = number of features in X
    Y : numpy.ndarray of shape (m, ny)
        m = same number of data points as in X
        ny = number of features in Y

    Returns:
    X_shuffled, Y_shuffled : numpy.ndarrays
        Shuffled versions of X and Y
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    return X[permutation], Y[permutation]
