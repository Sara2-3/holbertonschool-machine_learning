#!/usr/bin/env python3
"""Module for calculating precision from confusion matrix"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes) where row
                   indices represent the correct labels and column indices
                   represent the predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing the precision of
        each class
    """
    predicted_per_class = np.sum(confusion, axis=0)

    true_positives = np.diagonal(confusion)

    precision_values = true_positives / predicted_per_class

    return precision_values
