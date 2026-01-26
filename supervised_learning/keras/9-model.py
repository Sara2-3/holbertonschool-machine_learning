#!/usr/bin/env python3
"""
Module that saves and loads entire Keras models.
"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire Keras model.

    Parameters
    ----------
    network : K.Model
        The model to save.
    filename : str
        Path of the file where the model should be saved.

    Returns
    -------
    None
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire Keras model.

    Parameters
    ----------
    filename : str
        Path of the file from which the model should be loaded.

    Returns
    -------
    K.Model
        The loaded Keras model.
    """
    return K.models.load_model(filename)
