#!/usr/bin/env python3
"""Normal distribution module"""


class Normal:
    """Represents a Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            n = len(data)
            self.mean = float(sum(data) / n)
            variance = sum((x - self.mean) ** 2 for x in data) / n
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        pi = 3.141592653589793
        e = 2.718281828459045
        coefficient = 1 / (self.stddev * (2 * pi) ** 0.5)
        exponent = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)
        return coefficient * (e ** exponent)

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        z = (x - self.mean) / self.stddev
        return 0.5 * (1 + self._erf(z / (2 ** 0.5)))

    def _erf(self, x):
        """Error function approximation"""
        if x == 0:
            return 0
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        sign = 1 if x >= 0 else -1
        x = abs(x)
        t = 1.0 / (1.0 + p * x)
        term1 = ((((a5 * t + a4) * t) + a3) * t + a2) * t + a1
        y = 1.0 - term1 * t * (2.718281828459045 ** (-x * x))
        return sign * y
