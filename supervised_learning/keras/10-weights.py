#!/usr/bin/env python3
"""
Module that saves and loads the weights of a Keras model.
"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves the weights of a Keras model.

    Parameters
    ----------
    network : K.Model
        The model whose weights should be saved.
    filename : str
        Path of the file where the weights should be saved.
    save_format : str, optional
        Format in which the weights should be saved (default is 'keras').

    Returns
    -------
    None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Loads the weights into a Keras model.

    Parameters
    ----------
    network : K.Model
        The model to which the weights should be loaded.
    filename : str
        Path of the file from which the weights should be loaded.

    Returns
    -------
    None
    """
    network.load_weights(filename)
