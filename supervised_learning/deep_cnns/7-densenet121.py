#!/usr/bin/env python3
"""Dense Block for DenseNet"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block as described in Densely Connected CNNs.

    Args:
        X: output from the previous layer
        nb_filters: integer representing the number of filters in X
        growth_rate: growth rate for the dense block
        layers: number of layers in the dense block

    Returns:
        The concatenated output of each layer within the Dense Block
        and the number of filters within the concatenated outputs
    """
    init = K.initializers.HeNormal(seed=0)

    for _ in range(layers):
        bn1 = K.layers.BatchNormalization()(X)
        relu1 = K.layers.Activation('relu')(bn1)
        conv1 = K.layers.Conv2D(
            4 * growth_rate,
            (1, 1),
            padding='same',
            kernel_initializer=init
        )(relu1)

        bn2 = K.layers.BatchNormalization()(conv1)
        relu2 = K.layers.Activation('relu')(bn2)
        conv2 = K.layers.Conv2D(
            growth_rate,
            (3, 3),
            padding='same',
            kernel_initializer=init
        )(relu2)

        X = K.layers.Concatenate()([X, conv2])
        nb_filters += growth_rate

    return X, nb_filters
