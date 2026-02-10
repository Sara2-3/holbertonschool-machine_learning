#!/usr/bin/env python3
"""Module that calculates precision for each class
from a confusion matrix."""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Parameters
    ----------
    confusion : numpy.ndarray
        Confusion matrix of shape (classes, classes) where rows
        represent the true labels and columns represent the
        predicted labels.

    Returns
    -------
    numpy.ndarray
        Array of shape (classes,) containing the precision
        of each class.
    """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    precision = TP / (TP + FP)

    return precision
