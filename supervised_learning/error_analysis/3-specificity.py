#!/usr/bin/env python3
"""Module that calculates specificity for each class
from a confusion matrix."""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Parameters
    ----------
    confusion : numpy.ndarray
        Confusion matrix of shape (classes, classes) where rows
        represent the true labels and columns represent the
        predicted labels.

    Returns
    -------
    numpy.ndarray
        Array of shape (classes,) containing the specificity
        of each class.
    """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    TN = np.sum(confusion) - TP - FP - FN
    specificity = TN / (TN + FP)

    return specificity
