#!/usr/bin/env python3
"""
Module that calculates sensitivity for each class in a confusion matrix
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix

    Parameters:
    - confusion: numpy.ndarray of shape (classes, classes)
      where row indices represent the correct labels and
      column indices represent the predicted labels

    Returns:
    - numpy.ndarray of shape (classes,) containing the sensitivity
      of each class
    """
    # True Positives janë diagonalja
    TP = np.diag(confusion)

    # Për secilën klasë, FN = total i rreshtit - TP
    FN = np.sum(confusion, axis=1) - TP

    # Sensitivity = TP / (TP + FN)
    sensitivity = TP / (TP + FN)

    return sensitivity
