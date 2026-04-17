#!/usr/bin/env python3
"""Module that defines the deep_rnn function for deep RNN forward prop."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Perform forward propagation for a deep RNN.

    Args:
        rnn_cells: list of RNNCell instances of length l
        X: numpy.ndarray of shape (t, m, i) — input data
        h_0: numpy.ndarray of shape (l, m, h) — initial hidden states

    Returns:
        H: numpy.ndarray of shape (t+1, l, m, h) — all hidden states
        Y: numpy.ndarray of shape (t, m, o) — all outputs
    """
    t, m, _ = X.shape
    num_layers = len(rnn_cells)
    _, _, h = h_0.shape

    H = np.zeros((t + 1, num_layers, m, h))
    H[0] = h_0

    Y_list = []

    for step in range(t):
        x_input = X[step]
        for layer, cell in enumerate(rnn_cells):
            h_next, y = cell.forward(H[step, layer], x_input)
            H[step + 1, layer] = h_next
            x_input = h_next

        Y_list.append(y)

    Y = np.array(Y_list)

    return H, Y
