#!/usr/bin/env python3
"""
Module 1-normalize
Provides a function to normalize (standardize) a dataset using
pre-computed mean and standard deviation for each feature.
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    Parameters:
    X : numpy.ndarray of shape (d, nx)
        d = number of data points
        nx = number of features
    m : numpy.ndarray of shape (nx,)
        mean of all features of X
    s : numpy.ndarray of shape (nx,)
        standard deviation of all features of X

    Returns:
    numpy.ndarray of shape (d, nx)
        The normalized X matrix
    """
    return (X - m) / s
