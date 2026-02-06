#!/usr/bin/env python3
"""
Module 5-momentum
Provides a function to update variables using gradient descent
with momentum optimization.
"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using gradient descent with momentum.

    Parameters:
    alpha : float
        learning rate
    beta1 : float
        momentum weight
    var : numpy.ndarray
        variable to be updated
    grad : numpy.ndarray
        gradient of var
    v : numpy.ndarray
        previous first moment of var

    Returns:
    var_updated : numpy.ndarray
        updated variable
    v_new : numpy.ndarray
        new momentum term
    """
    v_new = beta1 * v + (1 - beta1) * grad
    var_updated = var - alpha * v_new
    return var_updated, v_new
