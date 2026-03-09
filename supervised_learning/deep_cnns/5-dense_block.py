#!/usr/bin/env python3
"""
Dense Block implementation for DenseNet-B
"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in Densely Connected
    Convolutional Networks.

    Args:
        X: output from the previous layer
        nb_filters: integer, number of filters in X
        growth_rate: growth rate for the dense block
        layers: number of layers in the dense block

    Returns:
        The concatenated output of each layer within the Dense Block
        and the number of filters within the concatenated outputs
    """
    he_normal = K.initializers.he_normal(seed=0)

    for i in range(layers):
        # Bottleneck layer: BN → ReLU → 1x1 Conv (4*growth_rate filters)
        bn1 = K.layers.BatchNormalization()(X)
        relu1 = K.layers.Activation('relu')(bn1)
        conv1 = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=he_normal
        )(relu1)

        # BN → ReLU → 3x3 Conv (growth_rate filters)
        bn2 = K.layers.BatchNormalization()(conv1)
        relu2 = K.layers.Activation('relu')(bn2)
        conv2 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=he_normal
        )(relu2)

        # Concatenate with input
        X = K.layers.Concatenate()([X, conv2])
        nb_filters += growth_rate

    return X, nb_filters
