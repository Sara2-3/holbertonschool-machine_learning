#!/usr/bin/env python3
"""
Create a Layer with Dropout
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.

    Parameters:
    - prev: tensor containing the output of the previous layer
    - n: number of nodes in the new layer
    - activation: activation function for the new layer
    - keep_prob: probability that a node will be kept
    - training: boolean indicating whether the model is in training mode

    Returns:
    - The output of the new layer
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode="fan_avg"
    )

    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )

    x = dense(prev)

    if training and keep_prob < 1.0:
        dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)
        x = dropout(x, training=True)

    return x
