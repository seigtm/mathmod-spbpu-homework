"""
Название: Модуль численного интегрирования методом Рунге-Кутты 4-го порядка.
Автор: Баранов Константин Павлович.
Дата создания: 23.02.2024.

Описание:
Этот модуль содержит функции для выполнения численного интегрирования методом Рунге-Кутты 4-го
порядка. Включает реализации алгоритма для интегрирования дифференциальных уравнений первого
порядка и сохранения результатов в файл Excel, а также построения графика.

Используемые библиотеки:
- numpy;
- pandas;
- matplotlib.pyplot.

Пример использования:
>>> t_values, y_values = numerical_integration(step=0.01, decimal_places=5, num_iterations=100)
>>> save_to_excel(t_values, y_values, "runge_kutta_results.xlsx")
>>> plot_results(t_values, y_values, "runge_kutta_plot.png")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def numerical_integration(step, decimal_places, num_iterations):
    """
    Выполняет численное интегрирование с использованием заданного размера шага, числа десятичных
    знаков итераций.

    Параметры:
        - step (float): Размер шага для интегрирования.
        - decimal_places (int): Количество десятичных знаков для округления.
        - num_iterations (int): Количество итераций для интегрирования.

    Возвращает:
        tuple: Кортеж, содержащий списки значений t и соответствующих значений y.
    """
    t_values = np.linspace(start=0,
                           stop=step * num_iterations,
                           num=num_iterations,
                           endpoint=False)
    vector_of_z1 = np.zeros(num_iterations)
    vector_of_z2 = np.zeros(num_iterations)
    vector_of_z3 = np.zeros(num_iterations)
    dynamic_steps = np.arange(start=step,
                              stop=step * num_iterations,
                              step=step)

    for i in range(1, num_iterations):
        vector_of_z1[i] = vector_of_z1[
            i - 1] + dynamic_steps[i - 1] * vector_of_z2[i - 1]
        vector_of_z2[i] = vector_of_z2[
            i - 1] + dynamic_steps[i - 1] * vector_of_z3[i - 1]
        vector_of_z3[i] = vector_of_z3[i - 1] - dynamic_steps[i - 1] * (
            vector_of_z1[i - 1] + 9 * vector_of_z2[i - 1] +
            26 * vector_of_z3[i - 1] - 5) / 24

    vector_of_y = 4 * vector_of_z1 - 8 * vector_of_z2 - 12 * vector_of_z3
    # Round the arrays to the specified number of decimal places.
    vector_of_y = np.round(vector_of_y, decimal_places)
    return t_values, vector_of_y


def save_to_excel(t_values, y_values, filename="calculations.xlsx"):
    """
    Сохраняет значения t и y в файл Excel.

    Параметры:
        - t_values (list): Список значений t.
        - y_values (list): Список соответствующих значений y.
        - filename (str): Имя файла Excel.
    """
    data_frame = pd.DataFrame({"t": t_values, "y": y_values})
    data_frame.to_excel(filename, index=False)


def plot_results(t_values, y_values, filename="plot.png"):
    """
    Строит график значений t и y и сохраняет его в файл.

    Параметры:
        - t_values (list): Список значений t.
        - y_values (list): Список соответствующих значений y.
        - filename (str): Имя файла для сохранения графика.
    """
    plt.plot(t_values, y_values)
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    """Perform numerical integration and plot the results."""
    t_values, y_values = numerical_integration(step=0.01,
                                               decimal_places=5,
                                               num_iterations=100)
    save_to_excel(t_values, y_values)
    plot_results(t_values, y_values)
