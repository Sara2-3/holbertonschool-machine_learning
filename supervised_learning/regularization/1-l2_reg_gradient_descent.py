#!/usr/bin/env python3
"""
Gradient Descent with L2 Regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization.

    Parameters:
    - Y: one-hot numpy.ndarray of shape (classes, m)
         Correct labels for the data
    - weights: dict of weights and biases of the neural network
    - cache: dict of outputs of each layer of the neural network
    - alpha: learning rate
    - lambtha: L2 regularization parameter
    - L: number of layers of the network

    The network uses tanh activations on hidden layers
    and softmax on the output layer.
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y  # derivative for softmax output layer

    for i in range(L, 0, -1):
        A_prev = cache[f"A{i-1}"]
        W = weights[f"W{i}"]

        # Gradient of weights with L2 regularization
        dW = (np.matmul(dZ, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # Update parameters
        weights[f"W{i}"] -= alpha * dW
        weights[f"b{i}"] -= alpha * db

        if i > 1:
            # Backprop through tanh activation
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - cache[f"A{i-1}"] ** 2)
