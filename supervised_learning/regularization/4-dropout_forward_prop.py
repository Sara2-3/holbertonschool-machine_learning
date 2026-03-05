#!/usr/bin/env python3
"""
Forward Propagation with Dropout
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Parameters:
    - X: numpy.ndarray of shape (nx, m) containing input data
    - weights: dictionary of the weights and biases of the neural network
    - L: number of layers in the network
    - keep_prob: probability that a node will be kept

    Returns:
    - Dictionary containing the outputs of each layer and the dropout masks
    """
    cache = {}
    cache["A0"] = X

    for i in range(1, L + 1):
        W = weights[f"W{i}"]
        b = weights[f"b{i}"]
        Z = np.matmul(W, cache[f"A{i-1}"]) + b

        if i != L:
            # tanh activation
            A = np.tanh(Z)

            # dropout mask
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            cache[f"D{i}"] = D.astype(int)

            # apply mask and scale
            A = (A * cache[f"D{i}"]) / keep_prob
            cache[f"A{i}"] = A
        else:
            # softmax activation for last layer
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            cache[f"A{i}"] = A

    return cache
