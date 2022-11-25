#P_fun.py
import numpy as np
from psi_function import psi


def P(x_in, x_unit, q):
    P_array = np.zeros((len(x_in)))
    for i in range(len(x_in)):
        for j in range(len(x_unit) - 1):
            if x_unit[j] <= x_in[i] < x_unit[j+1]:
                for k in range(4):
                    P_array[i] += q[2 * j + k] * psi(x_in[i],(x_unit[j], x_unit[j+1]), k)
    return P_array