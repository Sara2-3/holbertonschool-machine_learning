#!/usr/bin/env python3
"""
Module for randomly adjusting image contrast using TensorFlow.
"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Args:
        image (tf.Tensor): 3D tensor representing the input image.
        lower (float): Lower bound of the random contrast factor range.
        upper (float): Upper bound of the random contrast factor range.

    Returns:
        tf.Tensor: Contrast-adjusted image tensor.
    """
    return tf.image.random_contrast(image, lower, upper)
