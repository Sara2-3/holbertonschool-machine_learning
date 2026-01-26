#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras

def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network model using Keras functional API.

    Parameters:
    - nx: number of input features
    - layers: list with number of nodes in each layer
    - activations: list with activation functions for each layer
    - lambtha: L2 regularization parameter
    - keep_prob: probability that a node will be kept for dropout

    Returns:
    - keras Model
    """
    # Input layer
    inputs = keras.Input(shape=(nx,))
    x = inputs

    # Build hidden layers
    for i in range(len(layers)):
        # Dense layer with L2 regularization
        x = keras.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=keras.regularizers.l2(lambtha)
        )(x)

        # Apply dropout only if not the last layer
        if i < len(layers) - 1:
            x = keras.layers.Dropout(rate=1 - keep_prob)(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=x)

    return model
