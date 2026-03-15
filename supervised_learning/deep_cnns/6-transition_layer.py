#!/usr/bin/env python3
"""
Transition Layer implementation for DenseNet-C
"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Densely Connected
    Convolutional Networks.

    Args:
        X: output from the previous layer
        nb_filters: integer, number of filters in X
        compression: compression factor for the transition layer

    Returns:
        The output of the transition layer and the number of filters
        within the output, respectively
    """
    he_normal = K.initializers.he_normal(seed=0)

    bn = K.layers.BatchNormalization()(X)
    relu = K.layers.ReLU()(bn)
    compressed_filters = int(nb_filters * compression)
    conv = K.layers.Conv2D(
        filters=compressed_filters,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=he_normal
    )(relu)

    avg_pool = K.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv)

    return avg_pool, compressed_filters
