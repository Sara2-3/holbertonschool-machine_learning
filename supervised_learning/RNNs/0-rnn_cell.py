#!/usr/bin/env python3
"""Module that defines the RNNCell class for a simple RNN cell."""
import numpy as np


class RNNCell:
    """Represents a cell of a simple Recurrent Neural Network (RNN)."""

    def __init__(self, i, h, o):
        """
        Initialize the RNNCell.

        Args:
            i: dimensionality of the input data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) — previous hidden state
            x_t: numpy.ndarray of shape (m, i) — input data for the cell

        Returns:
            h_next: next hidden state
            y: output of the cell (softmax activated)
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(concat @ self.Wh + self.bh)

        logits = h_next @ self.Wy + self.by
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y = exp / exp.sum(axis=1, keepdims=True)

        return h_next, y
