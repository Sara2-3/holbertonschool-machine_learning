#!/usr/bin/env python3
"""
Module that performs convolution on images using multiple kernels
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels

    Parameters:
    - images: numpy.ndarray of shape (m, h, w, c)
      containing multiple images
    - kernels: numpy.ndarray of shape (kh, kw, c, nc)
      containing the kernels for the convolution
    - padding: either a tuple of (ph, pw), 'same', or 'valid'
    - stride: tuple of (sh, sw)

    Returns:
    - numpy.ndarray containing the convolved images
      with shape (m, out_h, out_w, nc)
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
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
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')

    # --- Output dimensions ---
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, out_h, out_w, nc))

    # --- Perform convolution (3 loops: i, j, k) ---
    for i in range(out_h):
        for j in range(out_w):
            region = padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            for k in range(nc):
                output[:, i, j, k] = np.sum(region * kernels[:, :, :, k],
                                            axis=(1, 2, 3))

    return output
