#!/usr/bin/env python3
"""
DenseNet-121 implementation
"""

from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture.

    Args:
        growth_rate: growth rate
        compression: compression factor

    Returns:
        The keras model
    """
    he_normal = K.initializers.he_normal(seed=0)

    # Input
    inputs = K.Input(shape=(224, 224, 3))

    # Initial convolution and pooling
    bn0 = K.layers.BatchNormalization()(inputs)
    relu0 = K.layers.ReLU()(bn0)
    conv0 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=he_normal
    )(relu0)
    pool0 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(conv0)

    nb_filters = 64

    # Dense Block 1 (6 layers)
    X, nb_filters = dense_block(pool0, nb_filters, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2 (12 layers)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3 (24 layers)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4 (16 layers)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Classification layer
    bn_final = K.layers.BatchNormalization()(X)
    relu_final = K.layers.ReLU()(bn_final)
    avg_pool = K.layers.GlobalAveragePooling2D()(relu_final)
    outputs = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=he_normal
    )(avg_pool)

    # Model
    model = K.models.Model(inputs=inputs, outputs=outputs)
    return model
