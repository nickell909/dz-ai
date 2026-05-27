import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from math import sin, exp

# Функция f(x)
def f(x):
    return sin(x / 5.0) * exp(x / 10.0) + 5 * exp(-x / 2.0)

# Построение графика функции f(x)
x_vals = np.linspace(1, 30, 1000)
y_vals = [f(x) for x in x_vals]

plt.figure(figsize=(12, 6))
plt.plot(x_vals, y_vals)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x)')
plt.grid(True)
plt.savefig('/workspace/02/f_plot.png')
plt.close()
print("График f(x) сохранен в f_plot.png")

# Задача 1: Минимизация гладкой функции с BFGS

# Шаг 6: начальное приближение x=2
result_bfgs_2 = minimize(f, x0=2, method='BFGS')
print("\n=== Задача 1 (BFGS, x0=2) ===")
print(f"x_min = {result_bfgs_2.x[0]}")
print(f"f(x_min) = {result_bfgs_2.fun}")
print(f"Количество итераций: {result_bfgs_2.nit}")
print(f"Количество вычислений функции: {result_bfgs_2.nfev}")
answer1_part1 = round(result_bfgs_2.fun, 2)
print(f"Ответ 1 (x0=2): {answer1_part1}")

# Шаг 7: начальное приближение x=30
result_bfgs_30 = minimize(f, x0=30, method='BFGS')
print("\n=== Задача 1 (BFGS, x0=30) ===")
print(f"x_min = {result_bfgs_30.x[0]}")
print(f"f(x_min) = {result_bfgs_30.fun}")
print(f"Количество итераций: {result_bfgs_30.nit}")
print(f"Количество вычислений функции: {result_bfgs_30.nfev}")
answer1_part2 = round(result_bfgs_30.fun, 2)
print(f"Ответ 1 (x0=30): {answer1_part2}")

print(f"\n=== ОТВЕТ ЗАДАЧА 1: {answer1_part1} {answer1_part2} ===")

# Задача 2: Глобальная оптимизация с дифференциальной эволюцией
bounds = [(1, 30)]
result_de = differential_evolution(f, bounds)
print("\n=== Задача 2 (Дифференциальная эволюция) ===")
print(f"x_min = {result_de.x[0]}")
print(f"f(x_min) = {result_de.fun}")
print(f"Количество итераций: {result_de.nit}")
print(f"Количество вычислений функции: {result_de.nfev}")
answer2 = round(result_de.fun, 2)
print(f"\n=== ОТВЕТ ЗАДАЧА 2: {answer2} ===")

# Задача 3: Минимизация негладкой функции h(x) = int(f(x))
def h(x):
    return int(f(x))

# Построение графика функции h(x)
y_vals_h = [h(x) for x in x_vals]

plt.figure(figsize=(12, 6))
plt.plot(x_vals, y_vals_h)
plt.xlabel('x')
plt.ylabel('h(x)')
plt.title('График функции h(x) = int(f(x))')
plt.grid(True)
plt.savefig('/workspace/02/h_plot.png')
plt.close()
print("\nГрафик h(x) сохранен в h_plot.png")

# Шаг 3: Минимизация h(x) с помощью BFGS, начальное приближение x=30
result_bfgs_h = minimize(h, x0=30, method='BFGS')
print("\n=== Задача 3 (BFGS, h(x), x0=30) ===")
print(f"x_min = {result_bfgs_h.x[0]}")
print(f"h(x_min) = {result_bfgs_h.fun}")
print(f"Количество итераций: {result_bfgs_h.nit}")
print(f"Количество вычислений функции: {result_bfgs_h.nfev}")
answer3_part1 = result_bfgs_h.fun
print(f"Ответ 3 (BFGS): {answer3_part1}")

# Шаг 4: Минимизация h(x) с помощью дифференциальной эволюции
result_de_h = differential_evolution(h, bounds)
print("\n=== Задача 3 (Дифференциальная эволюция, h(x)) ===")
print(f"x_min = {result_de_h.x[0]}")
print(f"h(x_min) = {result_de_h.fun}")
print(f"Количество итераций: {result_de_h.nit}")
print(f"Количество вычислений функции: {result_de_h.nfev}")
answer3_part2 = result_de_h.fun
print(f"Ответ 3 (DE): {answer3_part2}")

print(f"\n=== ОТВЕТ ЗАДАЧА 3: {answer3_part1} {answer3_part2} ===")

# Итоговые ответы
print("\n" + "="*50)
print("ИТОГОВЫЕ ОТВЕТЫ:")
print("="*50)
print(f"Задача 1: {answer1_part1} {answer1_part2}")
print(f"Задача 2: {answer2}")
print(f"Задача 3: {answer3_part1} {answer3_part2}")
