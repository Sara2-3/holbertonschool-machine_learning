#!/usr/bin/env python3
"""
Pooling forward propagation for a CNN layer
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.

    Parameters
    ----------
    A_prev : numpy.ndarray
        Shape (m, h_prev, w_prev, c_prev), output of the previous layer.
    kernel_shape : tuple
        (kh, kw), size of the kernel for pooling.
    stride : tuple, optional
        (sh, sw), strides for pooling. Default is (1, 1).
    mode : str, optional
        'max' or 'avg', type of pooling. Default is 'max'.

    Returns
    -------
    numpy.ndarray
        The output of the pooling layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Compute output dimensions
    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1

    # Initialize output
    output = np.zeros((m, h_new, w_new, c_prev))

    for i in range(h_new):
        for j in range(w_new):
            # Define slice corners
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            # Extract slice
            slice_A = A_prev[:, vert_start:vert_end, horiz_start:horiz_end, :]

            if mode == 'max':
                output[:, i, j, :] = np.max(slice_A, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(slice_A, axis=(1, 2))

    return output
