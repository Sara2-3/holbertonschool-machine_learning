#!/usr/bin/env python3
"""Module that returns the transpose of a 2D matrix."""


def matrix_transpose(matrix):
    """Returns the transpose of a 2D matrix."""
    transposed = []
    # Loop për kolonat
    for c in range(len(matrix[0])):
        new_row = []
        # Loop për rreshtat
        for r in range(len(matrix)):
            new_row.append(matrix[r][c])
        transposed.append(new_row)
    return transposed


if __name__ == "__main__":
    mat1 = [[1, 2], [3, 4]]
    print("Original:", mat1)
    print("Transpose:", matrix_transpose(mat1))

    mat2 = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
        [26, 27, 28, 29, 30]
    ]
    print("Original:", mat2)
    print("Transpose:", matrix_transpose(mat2))
