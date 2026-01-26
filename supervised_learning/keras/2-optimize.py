#!/usr/bin/env python3
"""
Module that sets up Adam optimization for a Keras model.
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Configures a Keras model for training with Adam optimizer.

    Parameters
    ----------
    network : K.Model
        The model to optimize.
    alpha : float
        Learning rate.
    beta1 : float
        First Adam optimization parameter.
    beta2 : float
        Second Adam optimization parameter.

    Returns
    -------
    None
    """
    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )
    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
