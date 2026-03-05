#!/usr/bin/env python3
"""
L2 Regularization Cost
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    - cost: tensor containing the cost of the network without L2 regularization
    - model: a Keras model that includes layers with L2 regularization

    Returns:
    - A tensor containing the total cost for each layer of the network,
      accounting for L2 regularization
    """
    # mbledhim penalitetet e regularizimit nga të gjitha shtresat
    reg_losses = model.losses

    # shtojmë koston bazë me penalitetet e secilës shtresë
    total_costs = [cost + reg_loss for reg_loss in reg_losses]

    return tf.convert_to_tensor(total_costs)
