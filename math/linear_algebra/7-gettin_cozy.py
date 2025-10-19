#!/usr/bin/env python3
"""
Module for concatenating 2D matrices
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along specific axis
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        result = []
        for i in range(len(mat1)):
            result.append(mat1[i] + mat2[i])
        return result
    else:
        return None
