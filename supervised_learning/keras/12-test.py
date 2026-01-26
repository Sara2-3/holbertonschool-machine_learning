#!/usr/bin/env python3
"""
Module that tests a Keras model using given data and labels.
"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a Keras model using the provided data and labels.

    Parameters
    ----------
    network : K.Model
        The model to test.
    data : numpy.ndarray
        Input data to test the model with.
    labels : numpy.ndarray
        Correct one-hot labels of the data.
    verbose : bool, optional
        If True, output is printed during the testing process.

    Returns
    -------
    tuple
        The loss and accuracy of the model with the testing data.
    """
    return network.evaluate(data, labels, verbose=verbose)
