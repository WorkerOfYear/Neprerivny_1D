# main.py
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from psi_function import psi
from scipy.interpolate import UnivariateSpline
from P_fun import P
from matrix_alfa import matrix_alfa


def slay_solver(x_in, f_in, x_unit, omega):
    "фунция формирует вектора A и b по входным x и f(x), а также узлам x_unit и весам omega,"
    "затем решает СЛАУ Aq=b и возвращает массив q"

    A_loc = np.zeros((len(x_unit) - 1, 4, 4))
    for n in range(len(x_unit) - 1):
        for i in range(4):
            for j in range(4):
                for k in range(len(x_in)):
                    A_loc[n][i][j] += omega[k] * psi(x_in[k], (x_unit[n], x_unit[n + 1]), i) * psi(x_in[k], (
                        x_unit[n], x_unit[n + 1]), j)

    alfa = 1
    A_loc_alfa = np.zeros((len(x_unit) - 1, 4, 4))
    for n in range(len(x_unit) - 1):
        A_loc_alfa[n] = alfa * matrix_alfa(x_unit[n], x_unit[n + 1])

    A = np.zeros((2 * len(x_unit), 2 * len(x_unit)))
    for n in range(len(x_unit) - 1):
        k = 2 * n
        A[k:k + 4, k:k + 4] += A_loc[n][:][:] + A_loc_alfa[n][:][:]

    b_loc = np.zeros((len(x_unit) - 1, 4))
    for n in range(len(x_unit) - 1):
        for i in range(4):
            for k in range(len(x_in)):
                b_loc[n][i] += omega[k] * psi(x_in[k], (x_unit[n], x_unit[n + 1]), i) * f_in[k]

    b = np.zeros((2 * len(x_unit)))
    for n in range(len(x_unit) - 1):
        k = 2 * n
        b[k:k + 4] += b_loc[n][:]

    q = np.linalg.solve(A, b)

    return q


# Задаём начальные данные[x_0 ; x_last]
x_0 = -math.pi  # начальное значение x
x_last = math.pi # конечное значение x
n = 30  # колличество точек x = n + 1
h = (x_last - x_0) / n  # шаг
x_in = np.arange(x_0, x_last + 0.1 * h, h)
f_in = np.zeros((len(x_in)))
for i in range(len(x_in)):
    f_in[i] = math.sin(x_in[i]) + random.random()  # random [0; 1)

# задаем узлы n_unit
# h_unit_1 = math.pi
# x_unit_1 = np.arange(x_0, x_last + h_unit_1, h_unit_1)
# print("Колличество узлов: %s" % len(x_unit_1))

n_unit = 5
x_unit = np.linspace(x_0, x_last + 0.1 * h, n_unit)
h_unit = (x_last + 0.1 * h - x_0) / (n_unit - 1)
print("Шаг между узлами: %s" % h_unit)
print("Колличество узлов: %s" % len(x_unit))
print("Узлы: %s" % x_unit)
print("Колличество локальных участков: %s" % (len(x_unit) - 1))



# # задаём массив весов omega
omega = np.ones((len(x_in)))
# q = slay_solver(x_in, f_in, x_unit, omega)

counter = 1 # задаём 1 чтобы начать цикл
# решаем слау Aq=b
while counter != 0:
    counter = 0
    q = slay_solver(x_in, f_in, x_unit, omega)
    # Находим среднее отклонение
    delta = np.zeros((len(f_in)))
    summ = 0
    f_new = P(x_in, x_unit, q)
    for i in range(len(f_in)):
        delta[i] = abs(f_new[i] - f_in[i])
        summ += delta[i]
    average_delta = summ / len(f_in)
    print("Среднее отклонение: %s" % average_delta)
    # Определяем точки в которых среднее отклонение превышает в n раз
    n = 2
    for i in range(len(delta)):
        if delta[i] >= average_delta * n and omega[i] == 1:
            omega[i] = omega[i] / n
            counter += 1 # считается колличество точек превышающих среднее отклонение
    print(counter)

# вызываем сглаживающую функцию
n_p = 1000
x_example = np.linspace(x_0, x_last, n_p)
P_n1 = np.zeros((len(x_example)))
P_n1 = P(x_example, x_unit, q)

# функция сглаживания в scipy
f2 = UnivariateSpline(x_in, f_in)
P_n_python = f2(x_example)

# выводим графики
plt.plot(x_example, P_n_python, label='scipy')
# plt.plot(x_example,P_n)
plt.plot(x_example, P_n1, label='me')
plt.scatter(x_in, f_in)
plt.legend()
plt.xlabel("x_in")
plt.ylabel("P_n")
plt.grid(True)
plt.tight_layout()
plt.show()