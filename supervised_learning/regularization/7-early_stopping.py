#!/usr/bin/env python3
"""
Early Stopping
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should be stopped early.

    Parameters:
    - cost: current validation cost of the neural network
    - opt_cost: lowest recorded validation cost of the network
    - threshold: threshold used for early stopping
    - patience: patience count used for early stopping
    - count: how long the threshold has not been met

    Returns:
    - (stop, count): tuple
        stop: boolean, True if training should stop early
        count: updated count
    """
    if cost < opt_cost:
        # found a new optimal cost → reset counter
        return False, 0

    # if cost has not decreased enough compared to opt_cost
    if cost - opt_cost <= threshold:
        count += 1
    else:
        count = 0

    if count >= patience:
        return True, count

    return False, count
