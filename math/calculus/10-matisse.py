#!/usr/bin/env python3
"""
Module for calculating the derivative of a polynomial.
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial represented as a list.

    Args:
        poly (list): List of polynomial coefficients. The index represents
                    the power of x.

    Returns:
        list: List representing the derivative of the polynomial.
              Returns None if input is invalid.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    derivative = []
    for power in range(1, len(poly)):
        derivative.append(power * poly[power])
    return derivative
