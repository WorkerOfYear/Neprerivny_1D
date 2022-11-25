#matrix_alfa.py
import numpy as np


def matrix_alfa(x_unit_1, x_unit_2):
    h = x_unit_2 - x_unit_1
    row1 = np.array([36, 3 * h, -36, 3 * h])
    row2 = np.array([3 * h, 4 * h ** 2, -3 * h, -1 * h ** 2])
    row3 = np.array([-36, -3 * h, 36, -3 * h])
    row4 = np.array([3 * h, -1 * h ** 2, -3 * h, 4 * h ** 2])

    arr = 1/(30 * h) * np.array([row1, row2, row3, row4])
    return arr
