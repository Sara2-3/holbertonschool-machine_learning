#!/usr/bin/env python3
"""
Module 14-batch_norm
Provides a function to create a batch normalization layer
for a neural network in TensorFlow.
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network.

    Parameters:
    prev : tensor
        activated output of the previous layer
    n : int
        number of nodes in the layer to be created
    activation : function
        activation function to be used on the output

    Returns:
    tensor
        activated output for the layer
    """
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'
        ),
        use_bias=False
    )(prev)

    batch_norm = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        gamma_initializer='ones',
        beta_initializer='zeros'
    )(dense)

    return activation(batch_norm)
