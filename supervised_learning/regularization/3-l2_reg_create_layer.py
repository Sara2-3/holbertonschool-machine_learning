#!/usr/bin/env python3
"""
Create a Layer with L2 Regularization
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer with L2 regularization.

    Parameters:
    - prev: tensor containing the output of the previous layer
    - n: number of nodes in the new layer
    - activation: activation function to use
    - lambtha: L2 regularization parameter

    Returns:
    - The output of the new layer
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode="fan_avg"
    )

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.L2(lambtha)
    )

    return layer(prev)
