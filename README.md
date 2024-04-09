# Mathematical Modeling labs

Here, you'll find practical implementations and exercises for the "**Mathematical Modeling**"
course at Saint Petersburg Polytechnic University, led by instructor Leontyeva T.V.

## Contents

1. [Numerical Integration](1.py).
2. [Coordinate Descent with Adaptive Step](2.py).
3. [Random Number Generator Verification](3.py).
4. [Parametric Model Identification](4.py).

## 1.py

### Description

This module implements functions for numerical integration using the 4th order Runge-Kutta method.
It includes implementations of the algorithm for integrating first-order differential equations,
saving the results to an Excel file, and plotting the results.

### Libraries Used

- numpy
- pandas
- matplotlib.pyplot

### Example Usage

```python
>>> t_values, y_values = numerical_integration(step=0.01, decimal_places=5, num_iterations=100)
>>> save_to_excel(t_values, y_values, "runge_kutta_results.xlsx")
>>> plot_results(t_values, y_values, "runge_kutta_plot.png")
```

### Functions

- `numerical_integration(step, decimal_places, num_iterations)`: Performs numerical integration
  using the specified step size, number of decimal places, and iterations. Returns lists of t and
  corresponding y values.
- `save_to_excel(t_values, y_values, filename="calculations.xlsx")`: Saves t and y values to an
  Excel file.
- `plot_results(t_values, y_values, filename="plot.png")`: Plots the t and y values and saves the
  plot to a file.

## 2.py

### Description

This module contains functions for optimizing functions using the coordinate descent method with
adaptive step. It includes implementations of the algorithm for ellipsoid and Rosenbrock function,
as well as a function for saving results to an Excel file and plotting.

### Libraries Used

- math
- numpy
- pandas
- matplotlib.pyplot

### Example Usage

```python
>>> points, values = coordinate_adaptive_ellipsoid(1, 1)
>>> save_to_excel_and_plot(points, values, "ellipsoid_1_1")
```

### Functions

- `coordinate_adaptive(function, step, init_point, inc_coef, dec_coef, epsilon, max_iter)`: Adjusts
  the adaptive step of coordinate descent.
- `func_ellipsoid(x, y, A, B)`: Computes the values of the ellipsoid function.
- `coordinate_adaptive_ellipsoid(A, B, initial_point=[-1, -1])`: Optimizes the ellipsoidal
  function using coordinate descent with adaptive step.
- `rosenbrock(x, y)`: Computes the value of the Rosenbrock function.
- `coordinate_adaptive_rosenbrock(initial_point)`: Optimizes the Rosenbrock function using
  coordinate descent with adaptive step.
- `save_to_excel_and_plot(points, values, filename_prefix)`: Saves the coordinates of points and
  function values in an Excel file and plots a graph.

## 3.py

### Description

This module contains functions for verifying the built-in random number generator and a custom
generator implemented using the Central Limit Theorem (CLT) method. It includes calculations of
mean, variance, standard deviation, and the construction of a frequency diagram.

### Libraries Used

- random
- math
- matplotlib.pyplot

### Example Usage

```python
>>> verify_default_generator(100000)
>>> verify_cpt_generator(100000)
```

### Functions

- `verify_default_generator(num_numbers)`: Verifies the built-in random number generator and
  displays statistics.
- `verify_cpt_generator(num_numbers)`: Verifies the custom random number generator implemented
  using the Central Limit Theorem (CLT) method and displays statistics.
- `cpt_generator()`: Generates a random number using the custom generator implemented using the
  Central Limit Theorem (CLT) method.
- `plot_histogram(numbers, num_bins, r1, r2, title)`: Plots a frequency histogram for the given
  numbers.

## 4.py

### Description

1. `numerical_integration` - Transition from transfer function to a differential equation and its
  solution using Euler's method. Output data (`y(t)` - model).
2. Implementation of a custom random number generator using the Central Limit Theorem (CLT) method.
3. Generating noise on the graph (adding noise to the output signal) - Experimental data.
4. Forming the objective function (considering `b1` and `b3` as unknowns).
5. Optimization method - coordinate descent with adaptive step (4 test points).

#### Libraries Used

- numpy
- pandas
- matplotlib.pyplot
- random
- concurrent.futures.ThreadPoolExecutor
