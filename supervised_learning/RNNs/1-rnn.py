#!/usr/bin/env python3
"""Module that defines the rnn function for simple RNN forward propagation."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Perform forward propagation for a simple RNN.

    Args:
        rnn_cell: instance of RNNCell used for forward propagation
        X: numpy.ndarray of shape (t, m, i) — input data
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0: numpy.ndarray of shape (m, h) — initial hidden state

    Returns:
        H: numpy.ndarray containing all hidden states, shape (t+1, m, h)
        Y: numpy.ndarray containing all outputs, shape (t, m, o)
    """
    t, m, _ = X.shape
    _, h = h_0.shape

    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    Y_list = []

    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        Y_list.append(y)

    Y = np.array(Y_list)

    return H, Y
