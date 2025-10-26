#!/usr/bin/env python3
"""Normal distribution module"""

import math


class Normal:
    """Represents a Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1):
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

            self.mean = float(sum(data) / len(data))

            sum_sttdev = 0
            for x in data:
                diff = x - self.mean
                sum_sttdev += diff ** 2

            variance = sum_sttdev / (len(data) - 1)
            self.stddev = float(variance ** 0.5)
