#!/usr/bin/env python3
"""
Module that converts a label vector into a one-hot matrix.
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Parameters
    ----------
    labels : array-like
        Vector of labels to convert.
    classes : int, optional
        Number of classes. If None, inferred from labels.

    Returns
    -------
    numpy.ndarray
        One-hot encoded matrix with shape (len(labels), classes).
    """
    return K.utils.to_categorical(labels, num_classes=classes)
