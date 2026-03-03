#!/usr/bin/env python3
"""
Module that performs convolution on grayscale images
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images

    Parameters:
    - images: numpy.ndarray of shape (m, h, w)
      containing multiple grayscale images
    - kernel: numpy.ndarray of shape (kh, kw)
      containing the kernel for the convolution
    - padding: either a tuple of (ph, pw), 'same', or 'valid'
    - stride: tuple of (sh, sw)

    Returns:
    - numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # --- Handle padding ---
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    else:  # 'valid'
        ph, pw = 0, 0

    # Pad images with zeros
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')

    # --- Output dimensions ---
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, out_h, out_w))

    # --- Perform convolution (only 2 loops: i, j) ---
    for i in range(out_h):
        for j in range(out_w):
            # slice the region
            region = padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
            # element-wise multiply and sum
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
