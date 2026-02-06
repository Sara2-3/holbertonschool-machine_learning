#!/usr/bin/env python3
"""
Module 6-momentum
Provides a function to create a TensorFlow optimizer
using gradient descent with momentum.
"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization algorithm.

    Parameters:
    alpha : float
        learning rate
    beta1 : float
        momentum weight

    Returns:
    optimizer : tf.keras.optimizers.Optimizer
        TensorFlow SGD optimizer with momentum
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
