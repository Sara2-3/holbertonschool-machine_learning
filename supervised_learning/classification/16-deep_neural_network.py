#!/usr/bin/env python3
"""
Defines a deep neural network performing binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Deep neural network class for binary classification
    """

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network

        Parameters
        ----------
        nx : int
            Number of input features
        layers : list
            List of number of nodes in each layer

        Raises
        ------
        TypeError
            If nx is not an integer
            If layers is not a list of positive integers
        ValueError
            If nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        for nodes in layers:
            if not isinstance(nodes, int) or nodes < 1:
                raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(self.L):
            layer_key = str(l + 1)
            nodes = layers[l]
            prev_nodes = nx if l == 0 else layers[l - 1]
            self.weights['W' + layer_key] = (
                np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            )
            self.weights['b' + layer_key] = np.zeros((nodes, 1))
