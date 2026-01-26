#!/usr/bin/env python3
"""
Module that makes predictions using a Keras model.
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes predictions using a Keras model.

    Parameters
    ----------
    network : K.Model
        The model to make predictions with.
    data : numpy.ndarray
        Input data to make predictions on.
    verbose : bool, optional
        If True, output is printed during the prediction process.

    Returns
    -------
    numpy.ndarray
        The predictions for the input data.
    """
    return network.predict(data, verbose=verbose)
