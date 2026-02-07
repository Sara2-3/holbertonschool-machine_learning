#!/usr/bin/env python3
"""
Module 12-learning_rate_decay
Provides a function to create a TensorFlow learning rate
decay operation using inverse time decay.
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in TensorFlow
    using inverse time decay.

    Parameters:
    alpha : float
        original learning rate
    decay_rate : float
        rate at which alpha decays
    decay_step : int
        number of passes of gradient descent before alpha decays further

    Returns:
    tf.keras.optimizers.schedules.LearningRateSchedule
        inverse time decay learning rate schedule
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
