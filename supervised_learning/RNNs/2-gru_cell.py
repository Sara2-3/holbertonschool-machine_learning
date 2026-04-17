#!/usr/bin/env python3
"""Module that defines the GRUCell class for a Gated Recurrent Unit."""
import numpy as np


class GRUCell:
    """Represents a Gated Recurrent Unit (GRU) cell."""

    def __init__(self, i, h, o):
        """
        Initialize the GRUCell.

        Args:
            i: dimensionality of the input data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wz = np.random.randn(h + i, h)
        self.Wr = np.random.randn(h + i, h)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
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

        z = self._sigmoid(concat @ self.Wz + self.bz)
        r = self._sigmoid(concat @ self.Wr + self.br)

        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_candidate = np.tanh(concat_r @ self.Wh + self.bh)

        h_next = (1 - z) * h_prev + z * h_candidate

        logits = h_next @ self.Wy + self.by
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y = exp / exp.sum(axis=1, keepdims=True)

        return h_next, y

    def _sigmoid(self, x):
        """Apply the sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
