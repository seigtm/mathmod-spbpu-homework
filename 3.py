"""
Название: Модуль проверки генераторов случайных чисел.
Автор: Баранов Константин Павлович.
Дата создания: 19.03.2024.

Описание:
Этот модуль содержит функции для проверки встроенного генератора случайных чисел и собственного
генератора, реализованного методом ЦПТ. Включает подсчет математического ожидания, дисперсии,
стандартного отклонения, а также построение частотной диаграммы.

Используемые библиотеки:
- random;
- math;
- matplotlib.pyplot.

Пример использования:
>>> verify_default_generator(100000)
>>> verify_cpt_generator(100000)
"""

import random
import math
import matplotlib.pyplot as plt
from typing import List


def verify_default_generator(num_numbers: int) -> None:
    """
    Проверяет встроенный генератор случайных чисел.

    Параметры:
        - num_numbers (int): Количество случайных чисел для генерации и анализа.

    Выводит на экран:
        - Математическое ожидание.
        - Дисперсию.
        - Стандартное отклонение.
        - Построенную частотную диаграмму.
    """

    # Генерация случайных чисел и подсчет математического ожидания.
    random_numbers = [random.uniform(0, 1) for _ in range(num_numbers)]
    mean = sum(random_numbers) / num_numbers
    print(f"Математическое ожидание: {mean}")

    # Подсчет дисперсии.
    variance = sum((x - mean)**2 for x in random_numbers) / num_numbers
    print(f"Дисперсия: {variance}")

    # Стандартное отклонение (корень из дисперсии).
    std_deviation = math.sqrt(variance)
    print(f"Стандартное отклонение: {std_deviation}")

    plot_histogram(random_numbers, 10, 0, 1,
                   "Частотная диаграмма для встроенного генератора")


def verify_cpt_generator(num_numbers: int) -> None:
    """
    Проверяет собственный генератор случайных чисел, реализованный методом ЦПТ.

    Параметры:
        - num_numbers (int): Количество случайных чисел для генерации и анализа.

    Выводит на экран:
        - Математическое ожидание.
        - Дисперсию.
        - Стандартное отклонение.
        - Построенную частотную диаграмму.
    """

    # Генерация случайных чисел и подсчет математического ожидания.
    random_numbers = [cpt_generator() for _ in range(num_numbers)]

    # Подсчет матожидания.
    mean = round(sum(random_numbers) / num_numbers, 4)
    print(f"Математическое ожидание: {mean}")

    # Подсчет дисперсии.
    variance = round(
        sum([(num - mean)**2 for num in random_numbers]) / num_numbers, 4)
    print(f"Дисперсия: {variance}")

    # Стандартное отклонение (корень из дисперсии).
    std_deviation = round(variance**0.5, 4)
    print(f"Стандартное отклонение: {std_deviation}")

    #Построение Гистограммы
    plot_histogram(random_numbers, 10, -2, 2,
                   "Частотная диаграмма для собственного генератора")


def cpt_generator() -> float:
    """
    Создает случайное число с использованием собственного генератора случайных чисел, реализованного методом ЦПТ.

    Возвращает:
        float: Случайное число.
    """

    n = 12
    normalized = [random.uniform(0, 1) for _ in range(n)]
    v = sum(normalized)
    m_v = n / 2
    z = (v - m_v) / (n / 12)**0.5
    x = z * 0.05 * 19.99973
    return x


def plot_histogram(numbers: List[float], num_bins: int, r1: float, r2: float,
                   title: str) -> None:
    """
    Строит частотную диаграмму для заданных чисел.

    Параметры:
        - numbers (list): Список чисел.
        - num_bins (int): Количество интервалов для построения.
        - r1 (float): Начальное значение диапазона.
        - r2 (float): Конечное значение диапазона.

    Выводит на экран:
        - Построенную частотную диаграмму.
    """

    hist, bins, _ = plt.hist(numbers,
                             bins=num_bins,
                             range=(r1, r2),
                             edgecolor='black')
    plt.xlabel('Интервалы')
    plt.ylabel('Частота')
    plt.title(title)

    # Добавляем подписи к столбцам.
    for i in range(len(hist)):
        plt.text(bins[i] + (bins[i + 1] - bins[i]) / 2,
                 hist[i],
                 str(int(hist[i])),
                 ha='center',
                 va='bottom')
    plt.show()


def main():
    """
    Основная функция программы. Выполняет проверку встроенного и собственного генераторов случайных
    чисел.
    """

    num_numbers = 100000  # Количество чисел.

    #Проверка встроенного ГСЧ.
    verify_default_generator(num_numbers)

    #Проверка собственного ГСЧ методом ЦПТ.
    verify_cpt_generator(num_numbers)


if __name__ == "__main__":
    main()
