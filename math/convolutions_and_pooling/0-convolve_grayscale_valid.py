#!/usr/bin/env python3
"""
Module that performs a valid convolution on grayscale images
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images

    Parameters
    ----------
    images : numpy.ndarray
        Shape (m, h, w) containing multiple grayscale images
        m: number of images
        h: height in pixels
        w: width in pixels
    kernel : numpy.ndarray
        Shape (kh, kw) containing the kernel for the convolution
        kh: kernel height
        kw: kernel width

    Returns
    -------
    numpy.ndarray
        Convolved images with shape (m, h - kh + 1, w - kw + 1)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Output dimensions
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Initialize output
    output = np.zeros((m, output_h, output_w))

    # Perform convolution using only two loops
    for i in range(output_h):
        for j in range(output_w):
            # Extract the slice of the image
            image_slice = images[:, i:i+kh, j:j+kw]
            # Element-wise multiplication and sum
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output
