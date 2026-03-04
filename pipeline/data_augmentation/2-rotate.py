#!/usr/bin/env python3
"""
Module for rotating images using TensorFlow.
"""

import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise.

    Args:
        image (tf.Tensor): 3D tensor containing the image to rotate.

    Returns:
        tf.Tensor: Rotated image tensor.
    """
    return tf.image.rot90(image)
