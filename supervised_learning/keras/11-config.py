#!/usr/bin/env python3
"""
Module that saves and loads the configuration of a Keras model in JSON format.
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves the configuration of a Keras model in JSON format.

    Parameters
    ----------
    network : K.Model
        The model whose configuration should be saved.
    filename : str
        Path of the file where the configuration should be saved.

    Returns
    -------
    None
    """
    json_config = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_config)


def load_config(filename):
    """
    Loads a Keras model from a JSON configuration file.

    Parameters
    ----------
    filename : str
        Path of the file containing the model's configuration in JSON format.

    Returns
    -------
    K.Model
        The loaded Keras model.
    """
    with open(filename, 'r') as f:
        json_config = f.read()
    return K.models.model_from_json(json_config)
