#!/usr/bin/env python3
"""Poisson Distribution"""

class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Initialize Poisson distribution"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))
    def pmf(self, k):
        """Calculate the PMF for given number of occurrences (k)"""
        k = int(k)
        if k < 0:
            return 0

        # Calculate factorial of k
        fact = 1
        for i in range(1, k + 1):
            fact *= i

        e = 2.7182818285
        # Poisson PMF formula: P(k) = (λ^k * e^-λ) / k!
        p = (self.lambtha ** k) * (e ** (-self.lambtha)) / fact
        return p