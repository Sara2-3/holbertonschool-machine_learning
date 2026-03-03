#!/usr/bin/env python3
"""
Module that calculates specificity for each class in a confusion matrix
"""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix

    Parameters:
    - confusion: numpy.ndarray of shape (classes, classes)
      where row indices represent the correct labels and
      column indices represent the predicted labels

    Returns:
    - numpy.ndarray of shape (classes,) containing the specificity
      of each class
    """
    classes = confusion.shape[0]
    total = np.sum(confusion)

    # True Positives për secilën klasë
    TP = np.diag(confusion)

    # False Positives: total i kolonës - TP
    FP = np.sum(confusion, axis=0) - TP

    # False Negatives: total i rreshtit - TP
    FN = np.sum(confusion, axis=1) - TP

    # True Negatives: total - (TP + FP + FN)
    TN = total - (TP + FP + FN)

    # Specificity = TN / (TN + FP)
    specificity = TN / (TN + FP)

    return specificity
