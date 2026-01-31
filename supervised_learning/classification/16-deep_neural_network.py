#!/usr/bin/env python3
"""
Module that defines a deep neural network performing binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Class that defines a deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network

        Parameters
        ----------
        nx : int
            Number of input features
        layers : list
            List representing the number of nodes in each layer

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
        if not all(isinstance(nodes, int) and nodes > 0 for nodes in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(1, self.L + 1):
            nodes = layers[l - 1]
            prev_nodes = nx if l == 1 else layers[l - 2]

            # He initialization for weights
            self.weights['W' + str(l)] = np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            # Biases initialized to zeros
            self.weights['b' + str(l)] = np.zeros((nodes, 1))
