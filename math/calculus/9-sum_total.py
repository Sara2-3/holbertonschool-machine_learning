#!/usr/bin/env python3
"""
Module that calculates the summation of i squared up to n.
"""


def summation_i_squared(n):
    """
    Calculates the sum of squares from 1 to n.

    Args:
        n (int): the stopping condition.

    Returns:
        int: the sum of squares (1² + 2² + ... + n²),
            or None if n is not a valid positive integer.
    """
    if not isinstance(n, int) or n < 1:
        return None
    # formula for sum of squares: n(n + 1)(2n + 1) / 6
    return int(n * (n + 1) * (2 * n + 1) / 6)
