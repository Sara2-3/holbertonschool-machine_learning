#!/usr/bin/env python3
"""
Module for matrix concatenation
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two matrices along specific axis
    Args:
        mat1: first numpy array
        mat2: second numpy array
        axis: axis along which to concatenate (default 0)
    Returns:
        numpy.ndarray: concatenated matrix
    """
    return np.concatenate((mat1, mat2), axis=axis)
