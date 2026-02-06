#!/usr/bin/env python3
"""
Module 7-RMSProp
Provides a function to update variables using the RMSProp
optimization algorithm.
"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Parameters:
    alpha : float
        learning rate
    beta2 : float
        RMSProp weight (decay rate for squared gradients)
    epsilon : float
        small number to avoid division by zero
    var : numpy.ndarray
        variable to be updated
    grad : numpy.ndarray
        gradient of var
    s : numpy.ndarray
        previous second moment of var (running average of squared gradients)

    Returns:
    var_updated : numpy.ndarray
        updated variable
    s_new : numpy.ndarray
        new second moment
    """
    # Update the running average of squared gradients
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    # Update the variable using RMSProp rule
    var_updated = var - alpha * grad / (np.sqrt(s_new) + epsilon)

    return var_updated, s_new
