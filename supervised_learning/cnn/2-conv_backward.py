#!/usr/bin/env python3
"""
Convolutional back propagation for a CNN layer
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.

    Parameters
    ----------
    dZ : numpy.ndarray
        Shape (m, h_new, w_new, c_new), partial derivatives with respect to
        the unactivated output of the convolutional layer.
    A_prev : numpy.ndarray
        Shape (m, h_prev, w_prev, c_prev), output of the previous layer.
    W : numpy.ndarray
        Shape (kh, kw, c_prev, c_new), kernels for the convolution.
    b : numpy.ndarray
        Shape (1, 1, 1, c_new), biases applied to the convolution.
    padding : str, optional
        Either 'same' or 'valid', type of padding used. Default is 'same'.
    stride : tuple, optional
        (sh, sw), strides for the convolution. Default is (1, 1).

    Returns
    -------
    dA_prev : numpy.ndarray
        Partial derivatives with respect to the previous layer.
    dW : numpy.ndarray
        Partial derivatives with respect to the kernels.
    db : numpy.ndarray
        Partial derivatives with respect to the biases.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    _, h_new, w_new, _ = dZ.shape

    # Padding
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    A_prev_pad = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant'
    )
    dA_prev_pad = np.zeros_like(A_prev_pad)

    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = a_prev_pad[vert_start:vert_end,
                                         horiz_start:horiz_end, :]

                    # Vetëm nëse slice ka formën e plotë
                    if a_slice.shape == (kh, kw, c_prev):
                        da_prev_pad[vert_start:vert_end,
                                    horiz_start:horiz_end, :] += (
                            W[:, :, :, c] * dZ[i, h, w, c]
                        )
                        dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

        dA_prev_pad[i] = da_prev_pad

    if padding == "same":
        dA_prev = dA_prev_pad[:, ph:-ph or None, pw:-pw or None, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
