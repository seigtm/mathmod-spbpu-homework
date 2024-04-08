"""
Название: Параметрическая идентификация модели.
Автор: Баранов Константин Павлович.
Дата создания: 07.04.2024.

Описание:
1. numerical_integration - Переход от передаточной функции к 
    Дифференциальному уравнению и его решение метолдом Эйлера
    Выходные данные (y(t) - модельный)
2. Реализация собственного ГСЧ  методом ЦПТ
3. Формирование шума на графике (добавление шума к выходному сигналу) 
    - Экспериментальные данные
4. Формирование целевой функции (будет считать b1 и b3 неизвестными)
5. Метод оптимизации - покоординатный спуск с адаптивным шагом (4 тестовых точки)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from concurrent.futures import ThreadPoolExecutor


# Получение модельного Y - 1.
def numerical_integration(step,
                          decimal_places,
                          num_iterations,
                          b1=None,
                          b3=None):
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

    # Обьявление значений из первой работы.
    b2 = 3.0
    x_t = 5.0
    k = 4.0
    a1 = 3.0
    a2 = 1.0

    if b1 is None and b3 is None:
        b3 = 4.0
        b1 = -2.0

    for i in range(1, num_iterations):
        vector_of_z1[i] = vector_of_z1[
            i - 1] + dynamic_steps[i - 1] * vector_of_z2[i - 1]
        vector_of_z2[i] = vector_of_z2[
            i - 1] + dynamic_steps[i - 1] * vector_of_z3[i - 1]
        vector_of_z3[i] = vector_of_z3[i - 1] - dynamic_steps[i - 1] * (
            vector_of_z1[i - 1] + (b3 + b2 - b1) * vector_of_z2[i - 1] +
            ((b2 * b3) - (b1 * b3) -
             (b1 * b2)) * vector_of_z3[i - 1] - x_t) / abs(b1 * b2 * b3)

    vector_of_y = k * (vector_of_z1 + ((a2 - a1) * vector_of_z2) -
                       (a1 * a2 * vector_of_z3))
    vector_of_y = np.round(vector_of_y, decimal_places)
    return t_values, vector_of_y


# 2.
def cpt_generator():
    n = 12
    normalized = [random.uniform(0, 1) for _ in range(n)]
    v = sum(normalized)
    m_v = n / 2
    z = (v - m_v) / (n / 12)**0.5
    x = z * 0.05 * 19.99973
    return x


# 3.
def noise():
    math_func = numerical_integration(step=0.01,
                                      decimal_places=5,
                                      num_iterations=100)
    iterations = len(math_func[1])
    noise_num = [cpt_generator() for c in range(iterations)]
    y_noised = [
        round(x, 4) + round(y, 4) for x, y in zip(math_func[1], noise_num)
    ]
    noised_coord = {'t': math_func[0], 'y': y_noised}
    return noised_coord


#4.
def cf(y_actual: list, y_noised: list):
    return sum([(y_a - y_e)**2
                for y_a, y_e in zip(y_actual, y_noised)]) / (len(y_actual) + 1)


# 4-5. Функция для покоординатного спуска с шагом.
# Получаем значения numerical_integration с пробной точкой вычисляем значение целевой функции.
# Затем возвращаем целевую функцию.
#   b1 == test_y.
#   b3 == test_x.
def func(test_y, test_x):
    _, y_tmp = numerical_integration(step=0.01,
                                     decimal_places=5,
                                     num_iterations=100,
                                     b1=test_y,
                                     b3=test_x)
    return cf(y_actual=y_tmp, y_noised=noised_data['y'])


#5.
def coordinate_adaptive(function,
                        step,
                        init_point,
                        inc_coef,
                        dec_coef,
                        epsilon=1e-6,
                        max_iter=100000):

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
    save_to_excel_and_plot_coordinate_adaptive(np.array(points),
                                               np.array(values),
                                               coordinate_adaptive_file_name)

    # np.array(points), np.array(values) - возвращаем только последний (самый близкий).
    return np.array(points)[-1]


def save_to_excel_and_plot(t_values,
                           y_values,
                           filename,
                           caller_is_coordinate=False,
                           additional_data=None):
    # Сохраняем данные в Excel файл.
    excel_filename = filename + ".xlsx"
    data_frame = pd.DataFrame({"t": t_values, "y": y_values})
    data_frame.to_excel(excel_filename, index=False)

    # Строим, сохраняем и рисуем на экран график.
    plt.plot(t_values, y_values, color='blue')

    if additional_data:
        plt.plot(additional_data['t'],
                 additional_data['y'],
                 color='orange',
                 linestyle='--')

    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.savefig("plot_" + filename + ".png")
    plt.show()


def save_to_excel_and_plot_coordinate_adaptive(points, values,
                                               filename_prefix):

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


def main():
    # 1.
    t_values, y_values = numerical_integration(step=0.01,
                                               decimal_places=5,
                                               num_iterations=100)
    save_to_excel_and_plot(t_values, y_values, "Modal")
    # 2-3.
    global noised_data, coordinate_adaptive_file_name
    coordinate_adaptive_file_name = "Coordinate_adaptive_1"
    noised_data = noise()
    save_to_excel_and_plot(noised_data['t'],
                           noised_data['y'],
                           "Noised",
                           additional_data={
                               't': t_values,
                               'y': y_values
                           })
    # 4-5.
    print("1s")
    with ThreadPoolExecutor(max_workers=2) as _:
        # Находим минимум фукции с приближенными значениями b1 и b3.
        test_b_values = coordinate_adaptive(func, [-0.1, 0.1], [-0.5, -0.3], 2,
                                            0.5)
        # Вычисляем передаточную функцию с полученными b1 и b3 из coordinate_adaptive.
        t_values_test, y_values_test = numerical_integration(
            step=0.01,
            decimal_places=5,
            num_iterations=100,
            b1=test_b_values[0],
            b3=test_b_values[1])

    b1_avg = test_b_values[0]
    b3_avg = test_b_values[1]

    print("2s")
    coordinate_adaptive_file_name = "Coordinate_adaptive_2"
    with ThreadPoolExecutor(max_workers=2) as _:
        # Находим минимум фукции с приближенными значениями b1 и b3.
        test_b_values = coordinate_adaptive(func, [0.1, -0.1], [-4, 9], 2, 0.5)
        # Вычисляем передаточную функцию с полученными b1 и b3 из coordinate_adaptive.
        t_values_test2, y_values_test2 = numerical_integration(
            step=0.01,
            decimal_places=5,
            num_iterations=100,
            b1=test_b_values[0],
            b3=test_b_values[1])

    b1_avg += test_b_values[0]
    b3_avg += test_b_values[1]

    print("3s")

    coordinate_adaptive_file_name = "Coordinate_adaptive_3"
    with ThreadPoolExecutor(max_workers=2) as _:
        # Находим минимум фукции с приближенными значениями b1 и b3.
        test_b_values = coordinate_adaptive(func, [0.1, -0.1], [-4, 7], 2, 0.5)
        #Вычисляем передаточную функцию и полученными b1 и b3 из coordinate_adaptive.
        t_values_test3, y_values_test3 = numerical_integration(
            step=0.01,
            decimal_places=5,
            num_iterations=100,
            b1=test_b_values[0],
            b3=test_b_values[1])

    b1_avg += test_b_values[0]
    b3_avg += test_b_values[1]

    print("b1_Avg")
    print(b1_avg / 3)
    print("b3_Avg")
    print(b3_avg / 3)
    t_values_avg, y_values_avg = numerical_integration(step=0.01,
                                                       decimal_places=5,
                                                       num_iterations=100,
                                                       b1=(b1_avg / 3),
                                                       b3=(b3_avg / 3))

    # Строим на графике значение, полученное из numerical_integration с тестовыми b1 и b3 (4 тестовых случая).
    plt.plot(t_values_test, y_values_test, color='orange', linestyle='--')
    plt.plot(t_values_test2, y_values_test2, color='pink', linestyle='--')
    plt.plot(t_values_test3, y_values_test3, color='olive', linestyle='--')
    plt.plot(t_values_avg, y_values_avg, color='red', linestyle='--')

    save_to_excel_and_plot(t_values, y_values, "Tests")

    plt.plot(t_values_avg, y_values_avg, color='red', linestyle='--')
    save_to_excel_and_plot(t_values, y_values, "итоговый")


if __name__ == "__main__":
    main()
