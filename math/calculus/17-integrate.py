#!/usr/bin/env python3
"""
Module for calculating the integral of a polynomial.
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial represented as a list.
    Returns None if input is invalid.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(x, (int, float)) for x in poly):
        return None
    if not isinstance(C, (int, float)):
        return None

    integral = [C]

    for i in range(len(poly)):
        val = poly[i] / (i + 1)
        if val == int(val):
            val = int(val)
        integral.append(val)

    return integral
