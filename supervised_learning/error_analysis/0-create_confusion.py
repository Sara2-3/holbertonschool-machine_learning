#!/usr/bin/env python3
"""
Module that creates a confusion matrix for classification tasks.
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Parameters
    ----------
    labels : numpy.ndarray
        One-hot array of shape (m, classes) containing the correct labels.
    logits : numpy.ndarray
        One-hot array of shape (m, classes) containing the predicted labels.

    Returns
    -------
    numpy.ndarray
        Confusion matrix of shape (classes, classes) where rows represent
        the true labels and columns represent the predicted labels.
    """
    # Convert one-hot arrays to class indices
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes), dtype=np.float64)

    # Fill the confusion matrix
    for t, p in zip(true_classes, pred_classes):
        confusion[t, p] += 1

    return confusion
