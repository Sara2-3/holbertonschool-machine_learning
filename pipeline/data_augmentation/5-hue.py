#!/usr/bin/env python3
"""
Module for adjusting image hue using TensorFlow.
"""

import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.

    Args:
        image (tf.Tensor): 3D tensor containing the image to change.
        delta (float): Amount the hue should change.

    Returns:
        tf.Tensor: Hue-adjusted image tensor.
    """
    return tf.image.adjust_hue(image, delta)
