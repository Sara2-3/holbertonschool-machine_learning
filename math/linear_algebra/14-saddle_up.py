#!/usr/bin/env python3
"""
Module for matrix multiplication
"""
import numpy as np


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication
    
    Args:
        mat1: first numpy array
        mat2: second numpy array
        
    Returns:
        numpy.ndarray: result of matrix multiplication
    """
    return np.matmul(mat1, mat2)