#!/usr/bin/env python3
"""
DenseNet-121 architecture
"""

from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Build the DenseNet-121 architecture.

    Args:
        growth_rate: the growth rate
        compression: the compression factor

    Returns:
        the keras model
    """
    init = K.initializers.HeNormal(seed=0)
    inputs = K.Input(shape=(224, 224, 3))
    nb_filters = 2 * growth_rate

    bn = K.layers.BatchNormalization()(inputs)
    relu = K.layers.Activation('relu')(bn)
    X = K.layers.Conv2D(
        nb_filters,
        (7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=init
    )(relu)
    X = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(X)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    X = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1)
    )(X)
    X = K.layers.Flatten()(X)
    output = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=init
    )(X)

    return K.Model(inputs=inputs, outputs=output)
