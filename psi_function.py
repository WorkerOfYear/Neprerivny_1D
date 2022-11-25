#psi_function.py
def psi(x, x_unit, l):
    global res
    if x_unit[0] <= x < x_unit[1]:
        h = x_unit[1] - x_unit[0]
        ksi = (x - x_unit[0]) / h
        if l == 0:
            res = 1 - 3 * ksi ** 2 + 2 * ksi ** 3
        elif l == 1:
            res = h * (ksi - 2 * ksi ** 2 + ksi ** 3)
        elif l == 2:
            res = 3 * ksi ** 2 - 2 * ksi ** 3
        elif l == 3:
            res = h * (-ksi ** 2 + ksi ** 3)
        else:
            print("Ошибка в заполнении пси функции")
    else:
        res = 0
    return res
