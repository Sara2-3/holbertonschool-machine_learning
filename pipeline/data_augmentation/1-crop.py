#!/usr/bin/env python3
"""
Module for performing random cropping on images using TensorFlow.
"""

import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image.

    Args:
        image (tf.Tensor): 3D tensor containing the image to crop.
        size (tuple): Desired size of the crop (height, width, channels).

    Returns:
        tf.Tensor: Cropped image tensor.
    """
    return tf.image.random_crop(image, size)
