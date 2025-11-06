#!/usr/bin/env python3
"""
Module for Line Graph
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """Plot y as a solid red line with x-axis from 0 to 10."""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(y, color='red')
    plt.xlim(0, 10)
    plt.show()
