#!/usr/bin/env python3
"""
Pooling back propagation for a CNN layer
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network.

    Parameters
    ----------
    dA : numpy.ndarray
        Shape (m, h_new, w_new, c_new), partial derivatives wrt output
        of the pooling layer.
    A_prev : numpy.ndarray
        Shape (m, h_prev, w_prev, c), output of the previous layer.
    kernel_shape : tuple
        (kh, kw), size of the kernel for pooling.
    stride : tuple, optional
        (sh, sw), strides for pooling. Default is (1, 1).
    mode : str, optional
        'max' or 'avg', type of pooling. Default is 'max'.

    Returns
    -------
    dA_prev : numpy.ndarray
        Partial derivatives wrt the previous layer.
    """
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    _, h_new, w_new, c_new = dA.shape

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c_idx in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == 'max':
                        a_slice = A_prev[i, vert_start:vert_end,
                                         horiz_start:horiz_end, c_idx]
                        mask = (a_slice == np.max(a_slice))
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, c_idx] += (
                            mask * dA[i, h, w, c_idx]
                        )
                    elif mode == 'avg':
                        da = dA[i, h, w, c_idx] / (kh * kw)
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, c_idx] += (
                            np.ones((kh, kw)) * da
                        )

    return dA_prev
