#!/usr/bin/env python3
"""
Module for element-wise operations
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise operations
    """
    sum = mat1 + mat2
    diff = mat1 - mat2
    prod = mat1 * mat2
    div = mat1 / mat2
    return (sum, diff, prod, div)
