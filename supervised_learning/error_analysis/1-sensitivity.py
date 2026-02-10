#!/usr/bin/env python3
"""
Module that calculates sensitivity (recall) for each class
from a confusion matrix.
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Parameters
    ----------
    confusion : numpy.ndarray
        Confusion matrix of shape (classes, classes) where rows represent
        the true labels and columns represent the predicted labels.

    Returns
    -------
    numpy.ndarray
        Array of shape (classes,) containing the sensitivity of each class.
    """
    # True Positives = diagonal values
    TP = np.diag(confusion)

    # False Negatives = sum of row - TP
    FN = np.sum(confusion, axis=1) - TP

    # Sensitivity = TP / (TP + FN)
    sensitivity = TP / (TP + FN)

    return sensitivity
