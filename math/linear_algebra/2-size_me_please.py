#!/usr/bin/env python3
"""Module that calculates the shape of a matrix."""


def matrix_shape(matrix):
    """Calculates the shape of a matrix as a list of integers."""
    shape = []
    current_level = matrix
    while isinstance(current_level, list):
        shape.append(len(current_level))
        current_level = current_level[0]
    return shape


if __name__ == "__main__":
    mat1 = [[1, 2], [3, 4]]
    print(matrix_shape(mat1))  # [2, 2]

    mat2 = [
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
    ]
    print(matrix_shape(mat2))  # [2, 3, 5]
