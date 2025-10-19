#!/usr/bin/env python3
"""
Module for adding arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise
    """
    # Kontrollo nëse kanë gjatësi të ndryshme
    if len(arr1) != len(arr2):
        return None
    
    # Krijo listën e re
    result = []
    
    # Mblidh element pas elementi
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    
    return result
