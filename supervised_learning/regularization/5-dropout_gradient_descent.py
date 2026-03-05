#!/usr/bin/env python3
"""
Gradient Descent with Dropout
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.

    Parameters:
    - Y: one-hot numpy.ndarray of shape (classes, m)
    - weights: dict of weights and biases of the neural network
    - cache: dict of outputs and dropout masks of each layer
    - alpha: learning rate
    - keep_prob: probability that a node will be kept
    - L: number of layers in the network

    Returns:
    - None (updates weights in place)
    """
    m = Y.shape[1]
    # derivative for softmax output layer
    dZ = cache[f"A{L}"] - Y

    for i in range(L, 0, -1):
        A_prev = cache[f"A{i-1}"]
        W = weights[f"W{i}"]

        # gradients
        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # update parameters
        weights[f"W{i}"] -= alpha * dW
        weights[f"b{i}"] -= alpha * db

        if i > 1:
            # propagate error backwards
            dA_prev = np.matmul(W.T, dZ)
            # apply dropout mask and scale
            dA_prev = (dA_prev * cache[f"D{i-1}"]) / keep_prob
            # derivative of tanh
            dZ = dA_prev * (1 - cache[f"A{i-1}"] ** 2)
