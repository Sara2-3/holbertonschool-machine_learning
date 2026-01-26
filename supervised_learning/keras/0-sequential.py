#!/usr/bin/env python3
"""
Builds a Sequential neural network model with Keras
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network model using Keras Sequential API

    Parameters
    ----------
    nx : int
        Number of input features
    layers : list of int
        Number of nodes in each layer
    activations : list of str
        Activation functions for each layer
    lambtha : float
        L2 regularization parameter
    keep_prob : float
        Probability that a node will be kept for dropout

    Returns
    -------
    model : keras.Model
        The compiled Keras Sequential model
    """
    # Initialize Sequential model
    model = K.Sequential()

    # Add layers one by one
    for i in range(len(layers)):
        # L2 regularization
        regularizer = K.regularizers.l2(lambtha)

        # First layer needs input_dim
        if i == 0:
            model.add(K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer,
                input_dim=nx
            ))
        else:
            model.add(K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer
            ))

        # Add Dropout after each hidden layer (not after output)
        if i != len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
