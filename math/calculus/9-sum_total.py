#!/usr/bin/env python3
"""
Module for calculating the sum of squares of integers from 1 to n.
"""


def summation_i_squared(n):
    """
    Calculate the sum of squares of integers from 1 to n.

    Uses the mathematical formula: n(n+1)(2n+1)/6

    Args:
        n (int): The stopping condition (upper limit)

    Returns:
        int: The integer value of the sum 1² + 2² + ... + n²
        None: If n is not a valid number
    """
    if not isinstance(n, int) or n < 1:
        return None

    return n * (n + 1) * (2 * n + 1) // 6


if __name__ == "__main__":
    n = 5
    print(summation_i_squared(n))
