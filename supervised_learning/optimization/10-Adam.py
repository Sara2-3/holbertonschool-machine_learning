#!/usr/bin/env python3
"""
Module 10-Adam
Provides a function to create a TensorFlow optimizer
using the Adam optimization algorithm.
"""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Sets up the Adam optimization algorithm in TensorFlow.

    Parameters:
    alpha : float
        learning rate
    beta1 : float
        weight used for the first moment
    beta2 : float
        weight used for the second moment
    epsilon : float
        small number to avoid division by zero

    Returns:
    optimizer : tf.keras.optimizers.Optimizer
        TensorFlow Adam optimizer
    """
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
