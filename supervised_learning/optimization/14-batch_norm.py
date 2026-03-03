#!/usr/bin/env python3
"""
Module that creates a batch normalization layer for a neural network
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow

    Parameters:
    - prev: activated output of the previous layer
    - n: number of nodes in the layer to be created
    - activation: activation function to be used

    Returns:
    - tensor of the activated output for the layer
    """
    dense_layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'
        )
    )
    dense = dense_layer(prev)

    mean, variance = tf.nn.moments(dense, axes=[0])

    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    epsilon = 1e-7
    normalized = (dense - mean) / tf.sqrt(variance + epsilon)
    bn = gamma * normalized + beta

    return activation(bn)
