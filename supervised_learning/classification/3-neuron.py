#!/usr/bin/env python3
"""
Module that defines a single Neuron for binary classification
"""

import numpy as np


class Neuron:
    """
    Class that defines a single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Initialize the neuron

        Parameters
        ----------
        nx : int
            Number of input features to the neuron

        Raises
        ------
        TypeError
            If nx is not an integer
        ValueError
            If nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        Parameters
        ----------
        X : numpy.ndarray
            Shape (nx, m) containing the input data

        Returns
        -------
        numpy.ndarray
            The activated output of the neuron
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Parameters
        ----------
        Y : numpy.ndarray
            Shape (1, m) containing the correct labels
        A : numpy.ndarray
            Shape (1, m) containing the activated output

        Returns
        -------
        float
            The logistic regression cost
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * np.log(1.0000001 - A))
        return cost
