"""
Название: Модуль координатного спуска с адаптивным шагом.
Автор: Баранов Константин Павлович.
Дата создания: 18.03.2024.

Описание:
Этот модуль содержит функции для оптимизации функций методом координатного спуска
с адаптивным шагом. Включает реализации алгоритма для функций эллипсоида и
Розенброка, а также функцию для сохранения результатов в файл Excel и
построения графика.

Используемые библиотеки:
- math;
- numpy;
- pandas;
- matplotlib.pyplot.

Пример использования:
>>> points, values = coordinate_adaptive_ellipsoid(1, 1)
>>> save_to_excel_and_plot(points, values, "ellipsoid_1_1")
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def coordinate_adaptive(function,
                        step,
                        init_point,
                        inc_coef,
                        dec_coef,
                        epsilon=1e-7,
                        max_iter=500000):
    """
    Функция для настройки адаптивного шага координатного спуска.

    Параметры:
        - function (callable): Функция, которую требуется минимизировать.
        - step (list): Шаги изменения координат.
        - init_point (list): Начальная точка для оптимизации.
        - inc_coef (float): Коэффициент для увеличения шага.
        - dec_coef (float): Коэффициент для уменьшения шага.
        - epsilon (float): Порог останова алгоритма.
        - max_iter (int): Максимальное количество итераций.

    Возвращает:
        - points (numpy.ndarray): Массив с координатами точек траектории поиска.
        - values (numpy.ndarray): Массив со значениями функции в каждой точке.
    """
    current_point = np.array(init_point, dtype=float)
    n = len(current_point)
    # Список для хранения всех точек.
    points = [current_point]
    # Список для хранения значений функции в каждой точке.
    values = [function(*current_point)]

    for _ in range(max_iter):
        for dim in range(n):
            new_point = points[-1].copy()  # Копируем текущую точку.
            new_point[dim] += step[dim]

            new_value = function(*new_point)
            current_value = function(*points[-1])

            if new_value < current_value:  # Если значение функции уменьшилось:
                points.append(new_point)  # Добавляем точку.
                values.append(new_value)
                step[dim] *= inc_coef  # Увеличиваем шаг.

            else:  # Если значение функции увеличилось:
                step[dim] *= dec_coef  # Уменьшаем шаг.

        if len(values) >= 2 and abs(
                values[-1] - values[-2]) < epsilon:  # Условие выхода из цикла.
            break

    return np.array(points), np.array(values)


def func_ellipsoid(x, y, A, B):
    """
    Функция для вычисления значений эллипсоида.

    Параметры:
        - x (float): Координата x.
        - y (float): Координата y.
        - A (float): Параметр A эллипсоида.
        - B (float): Параметр B эллипсоида.

    Возвращает:
        - float: Значение функции в заданных координатах.
    """
    return (x * x) / A + (y * y) / B


def coordinate_adaptive_ellipsoid(A, B, initial_point=[-1, -1]):
    """
    Функция для оптимизации эллипсоидальной функции методом координатного спуска с адаптивным шагом.

    Параметры:
        - A (float): Параметр A эллипсоида.
        - B (float): Параметр B эллипсоида.

    Возвращает:
        - points (numpy.ndarray): Массив с координатами точек траектории поиска.
        - values (numpy.ndarray): Массив со значениями функции в каждой точке.
    """
    func = lambda x, y: func_ellipsoid(x, y, A, B)
    steps = [0.0001, 0.0001]
    return coordinate_adaptive(func, steps, initial_point, 1.1, 0.5)


def rosenbrock(x, y):
    """
    Функция для вычисления значения функции Розенброка.

    Параметры:
        - x (float): Координата x.
        - y (float): Координата y.

    Возвращает:
        - float: Значение функции в заданных координатах.
    """
    return math.pow(1 - x, 2) + 100 * math.pow(y - math.pow(x, 2), 2)


def coordinate_adaptive_rosenbrock(initial_point):
    """
    Функция для оптимизации функции Розенброка методом координатного спуска с адаптивным шагом.

    Возвращает:
        - points (numpy.ndarray): Массив с координатами точек траектории поиска.
        - values (numpy.ndarray): Массив со значениями функции в каждой точке.
    """
    steps = [-0.1, -0.1]
    return coordinate_adaptive(rosenbrock, steps, initial_point, 3, 0.5)


def save_to_excel_and_plot(points, values, filename_prefix):
    """
    Функция для сохранения координат точек и значений функции в Excel-файле и построения графика.

    Параметры:
        - points (numpy.ndarray): Массив с координатами точек.
        - values (numpy.ndarray): Массив со значениями функции в каждой точке.
        - filename_prefix (str): Префикс имени файла.

    Возвращает:
        - None
    """
    df = pd.DataFrame({
        "x": [point[0] for point in points],
        "y": [point[1] for point in points],
        "value": values,
    })
    df.to_excel(f"{filename_prefix}.xlsx", index=False)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.plot(df["x"], df["y"], "-o")
    plt.savefig(f"{filename_prefix}.jpeg")
    plt.clf()
    plt.close()


points, values = coordinate_adaptive_ellipsoid(1, 1)
save_to_excel_and_plot(points, values, "ellipsoid_1_1")

points, values = coordinate_adaptive_ellipsoid(3, 5)
save_to_excel_and_plot(points, values, "ellipsoid_3_5")

points, values = coordinate_adaptive_rosenbrock([3, 4])
save_to_excel_and_plot(points, values, "rosenbrock")

# Edge case for Ellipsoid:
points, values = coordinate_adaptive_ellipsoid(1, 1, [0.5, 5])
save_to_excel_and_plot(points, values, "ellipsoid_05_5")

# Edge case for Rosenbrock:
points, values = coordinate_adaptive_rosenbrock([-2, 1.5])
save_to_excel_and_plot(points, values, "rozenbrok_-2_1_5")
