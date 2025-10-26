#!/usr/bin/env python3
"""Exponential distribution module"""


class Exponential:
    """Represents a Exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Initialize Poisson distribution with data or lambtha"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            x= (sum(data) / len(data))
            self.lambtha = float(1/x)
