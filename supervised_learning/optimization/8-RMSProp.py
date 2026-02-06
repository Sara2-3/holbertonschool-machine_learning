#!/usr/bin/env python3
"""
Module 8-RMSProp
Provides a function to create a TensorFlow optimizer
using the RMSProp optimization algorithm.
"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow.

    Parameters:
    alpha : float
        learning rate
    beta2 : float
        RMSProp weight (discounting factor)
    epsilon : float
        small number to avoid division by zero

    Returns:
    optimizer : tf.keras.optimizers.Optimizer
        TensorFlow RMSProp optimizer
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
