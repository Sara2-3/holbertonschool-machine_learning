#!/usr/bin/env python3
"""
Module 3-mini_batch
Provides a function to create mini-batches for training a neural network
using mini-batch gradient descent.
"""

import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches to be used for training.

    Parameters:
    X : numpy.ndarray of shape (m, nx)
        m = number of data points
        nx = number of features
    Y : numpy.ndarray of shape (m, ny)
        m = same number of data points as in X
        ny = number of classes for classification tasks
    batch_size : int
        number of data points in a batch

    Returns:
    list of tuples (X_batch, Y_batch)
        mini-batches of the dataset
    """
    # Shuffle the data first
    X, Y = shuffle_data(X, Y)
    m = X.shape[0]
    mini_batches = []

    # Create full batches
    for i in range(0, m, batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
