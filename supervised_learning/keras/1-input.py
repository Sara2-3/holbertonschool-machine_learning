#!/usr/bin/env python3
import tensorflow.keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network model using Keras functional API.

    Parameters:
    nx: number of input features
    layers: list with number of nodes in each layer
    activations: list with activation functions used for each layer
    lambtha: L2 regularization parameter
    keep_prob: probability that a node will be kept for dropout

    Returns:
    Keras Model
    """
    # Input layer
    inputs = K.Input(shape=(nx,))
    x = inputs

    # Hidden layers
    for i in range(len(layers)):
        # Dense layer with L2 regularization
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

        # Dropout (not applied on last layer)
        if i < len(layers) - 1:
            x = K.layers.Dropout(rate=1 - keep_prob)(x)

    # Define the model
    model = K.Model(inputs=inputs, outputs=x)
    return model
