#!/usr/bin/env python3
"""
Module for randomly adjusting image brightness using TensorFlow.
"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    Args:
        image (tf.Tensor): 3D tensor containing the image to change.
        max_delta (float): Maximum amount the image should be
    brightened or darkened.


    Returns:
        tf.Tensor: Brightness-adjusted image tensor.
    """
    return tf.image.random_brightness(image, max_delta)
