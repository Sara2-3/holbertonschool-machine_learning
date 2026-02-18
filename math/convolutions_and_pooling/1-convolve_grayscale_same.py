#!/usr/bin/env python3
"""
Module that performs a same convolution on grayscale images
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images

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
        Convolved images with shape (m, h, w)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Compute padding sizes
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad images with zeros
    padded = np.pad(images,
                    ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant')

    # Initialize output with same size as input
    output = np.zeros((m, h, w))

    # Perform convolution using only two loops
    for i in range(h):
        for j in range(w):
            # Extract slice from padded image
            image_slice = padded[:, i:i+kh, j:j+kw]
            # Element-wise multiplication and sum
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output
