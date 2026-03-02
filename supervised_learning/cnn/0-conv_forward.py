#!/usr/bin/env python3
"""
Convolutional forward propagation module
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer.

    Parameters
    ----------
    A_prev : np.ndarray
        Shape (m, h_prev, w_prev, c_prev), output of previous layer.
    W : np.ndarray
        Shape (kh, kw, c_prev, c_new), kernels for convolution.
    b : np.ndarray
        Shape (1, 1, 1, c_new), biases applied to convolution.
    activation : function
        Activation function applied to convolution output.
    padding : str, optional
        Either 'same' or 'valid', type of padding used (default 'same').
    stride : tuple, optional
        (sh, sw), strides for convolution (default (1, 1)).

    Returns
    -------
    np.ndarray
        Output of the convolutional layer, shape (m, h_new, w_new, c_new).
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
    else:
        ph, pw = 0, 0

    A_padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant",
        constant_values=0
    )

    h_new = int((h_prev + 2 * ph - kh) / sh) + 1
    w_new = int((w_prev + 2 * pw - kw) / sw) + 1

    Z = np.zeros((m, h_new, w_new, c_new))

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    slice_A = A_padded[
                        i, vert_start:vert_end, horiz_start:horiz_end, :
                    ]
                    Z[i, h, w, c] = (
                        np.sum(slice_A * W[:, :, :, c]) + b[0, 0, 0, c]
                    )

    return activation(Z)
