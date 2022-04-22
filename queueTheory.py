from importlib.abc import Traversable
from math import log, exp, floor, factorial
from pydoc import doc
from random import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from collections import Counter
from tabulate import tabulate




def F_reverse(x):
    return -log(1 - x)/lyambda

def F(x):
    return 1 - exp(-lyambda * x)

def F_empirical(x):
    count = 0
    for y in y_vec_sorted:
        if y >= x: break
        count += 1
    return count/N

def f(x):
    return lyambda * exp(-lyambda*x)

def math_expectation():
    return 1/lyambda

def math_expectation_empirical():
    return sum(y_vec)/N

def math_expectation_check():
    math_expectation_ = math_expectation()
    math_expectation_empirical_ = math_expectation_empirical()
    return abs(math_expectation_- math_expectation_empirical_) * 100 / math_expectation_

def dispersion():
    return 1/lyambda**2

def dispersion_empirical():
    m = math_expectation_empirical()
    return sum([(y - m)**2 for y in y_vec])/N

def dispersion_check():
    dispersion_ = dispersion()
    dispersion_empirical_ = dispersion_empirical()
    return abs(dispersion_ - dispersion_empirical_) * 100 / dispersion_

def standard_deviation():
    return sum([(f((z_vec[i+1]+z_vec[i])/2) - f_vec[i])**2 for i in range(len(f_vec))])

def plot_F_and_F_empirical():
    x_vec = np.linspace(y_vec_sorted[0], y_vec_sorted[-1], int((y_vec_sorted[-1] - y_vec_sorted[0])/0.001))
    F_vec = []
    F_empirical_vec = []
    for x in x_vec:
        F_vec.append(F(x))
        F_empirical_vec.append(F_empirical(x))
    plt.plot(x_vec, F_vec, '-')
    plt.plot(x_vec, F_empirical_vec, '-')
    plt.show()

def z_vec_init_1():
    return np.linspace(y_vec_sorted[0], y_vec_sorted[-1], round(1.44*log(N) + 1)+1)

def z_vec_init_2(n):
    z_vec = []
    leight = (N-1)//n
    remainder = (N-1)%n
    currentIndex = 0
    z_vec.append(y_vec_sorted[currentIndex])
    for i in range(n):
        currentIndex += leight + 1 if i >= n - remainder else leight
        z_vec.append(y_vec_sorted[currentIndex])
    return z_vec
    

def z_vec_init_3():
    z_vec = []
    while True:
        a = float(input())
        if a > y_vec_sorted[-1]:
            z_vec.append(a)
            break
        z_vec.append(a)
    return z_vec

def l_vec_init():
    l_vec, _ = np.histogram(y_vec_sorted, bins=z_vec)
    return l_vec

def f_vec_init():
    i_vec = [z_vec[i+1] - z_vec[i] for i in range(len(z_vec) - 1)]
    return [l/(N * i) for l,i in zip(l_vec,i_vec)]

def plot_f_and_f_empirical():
    x_vec = np.linspace(z_vec[0], z_vec[-1], int((z_vec[-1] - z_vec[0])/0.001))
    i = 1
    res = []; resf = []
    for x in x_vec:
        while x > z_vec[i]:
            i += 1
        res.append(f_vec[i-1])
        resf.append(f(x))
    plt.plot(x_vec, res, '-')
    plt.plot(x_vec, resf, '-')
    plt.show()

def F_empirical_2(x):
    count = 0
    for k,nk in table:
        if k >= x: break
        count += nk
    return count/m

def F_2(x):
    k = 0
    res = 0
    while True:
        if k >= x: break
        res += P_tau(t0, k)
        k += 1
    return res

def plot_F_2_and_F_2_empirical():
    x_vec = np.linspace(0, m*t0, int((m*t0)/0.001))
    F_vec = []
    F_empirical_vec = []
    for x in x_vec:
        F_vec.append(F_2(x))
        F_empirical_vec.append(F_empirical_2(x))
    plt.plot(x_vec, F_vec, '-')
    plt.plot(x_vec, F_empirical_vec, '-')
    plt.show()

    


def hiSquare():
    print("Гипотеза: полученная выборка имеет распределение Пуассона")
    p_vec = p_vec_init()
    R0 = sum([(l-N*p)**2/(N*p) for l,p in zip(l_vec, p_vec)])
    r = len(l_vec)-1
    alpha = float(input("Введите параметр alpha "))
    print(f"alpha = {alpha}, R0 = {R0}, r = {r}")
    lyambda = chi2.ppf(1-alpha, r)
    if R0 < lyambda:
        resultText=f"{R0} < {lyambda} => Гипотеза принята"
    else:
        resultText=f"{R0} >= {lyambda} => Гипотеза отвергнута"
    print(resultText)


def p_vec_init():
    return [F(z_vec[i+1]) - F(z_vec[i]) for i in range(len(z_vec)-1)]

def P_tau(t0, k):
    return exp(-lyambda*t0)*(lyambda*t0)**k/factorial(k)


    









N = int(input("Введите N "))
lyambda = float(input("Введите lyambda "))

tau_vec = []
y_vec = []
y_vec_sorted = []
z_vec=[]
currentTime = 0
for i in range(N):
    x = random()
    y = F_reverse(x)
    y_vec.append(y)
    tau_vec.append(currentTime + y)
    currentTime += y
y_vec_sorted = sorted(y_vec)



math_expectation_ = math_expectation()
math_expectation_empirical_ = math_expectation_empirical()
math_expectation_check_ = math_expectation_check()
dispersion_ = dispersion()
dispersion_empirical_ = dispersion_empirical()
dispersion_check_ = dispersion_check()

print("Математическое ожидание: ", math_expectation_, "\n")
print("Выборочное среднее: ", math_expectation_empirical_, "\n")
print("Различие математического ожидания и выборочного среднего: ", math_expectation_check_, "%", "\n")
print("Дисперсия: ", dispersion_, "\n")
print("Выборочная дисперсия: ", dispersion_empirical_, "\n")
print("Различие дисперсии и выборочной дисперсии: ", dispersion_check_, "%", "\n")


plot_F_and_F_empirical()


print("1) Разбиение на равные отрезки", "\n")
print("2) Примерно равное количество значений в каждом отрезке разбиения", "\n")
print("3) Ручной ввод", "\n")
variant = int(input("Выберете способ разбиения"))
if variant == 1:
    z_vec = z_vec_init_1()
elif variant == 2:
    n = int(input("Введите n"))
    z_vec = z_vec_init_2(n)
elif variant == 3:
    z_vec = z_vec_init_3()


l_vec = l_vec_init()
f_vec = f_vec_init()


plot_f_and_f_empirical()

standard_deviation_ = standard_deviation()
print("Расхождение между плотностью и гистограммой: ", standard_deviation_)


hiSquare()


t0 = float(input("Введите t0 "))


print("Гипотеза: nu(t0) имеет распределение Пуассона\n")
m = int((tau_vec[-1] - 0)/t0)
partition_tau_vec = np.linspace(0, m*t0, m+1)
partition_tau_vec_count, _ = np.histogram(tau_vec, bins=partition_tau_vec)
table = sorted(Counter(partition_tau_vec_count).items(), key=lambda kv: kv[0]) 
l = len(table)
i = 0
while i < l:
    if i != table[i][0]:
        table = table[:i] + [(i, 0),] + table[i:]
        l += 1
    i += 1
r = len(table)-1
print(tabulate(table, headers=["Количество заявок", "Число интервалов"], tablefmt="pretty"))
R0 = sum([(nk - m*P_tau(t0, k))**2/m*P_tau(t0, k) for k,nk in table])
alpha = float(input("Введите параметр alpha "))
print(f"\nalpha = {alpha}, R0 = {R0}, r = {r}\n")
lyambda_ = chi2.ppf(1-alpha, r)
if R0 < lyambda_:
    resultText=f"{R0} < {lyambda_} => Гипотеза принята"
else:
    resultText=f"{R0} >= {lyambda_} => Гипотеза отвергнута"
print(resultText)


math_expectation_ = lyambda*t0
math_expectation_empirical_ = sum([k*nk for k,nk in table])/m
math_expectation_check_ = abs(math_expectation_- math_expectation_empirical_) * 100 / math_expectation_
dispersion_ = lyambda*t0
dispersion_empirical_ = sum([(k - math_expectation_empirical_)**2*nk for k,nk in table])/(m-1)
dispersion_check_ = abs(dispersion_ - dispersion_empirical_) * 100 / dispersion_


print("Математическое ожидание: ", math_expectation_, "\n")
print("Выборочное среднее: ", math_expectation_empirical_, "\n")
print("Различие математического ожидания и выборочного среднего: ", math_expectation_check_, "%", "\n")
print("Дисперсия: ", dispersion_, "\n")
print("Выборочная дисперсия: ", dispersion_empirical_, "\n")
print("Различие дисперсии и выборочной дисперсии: ", dispersion_check_, "%", "\n")



plot_F_2_and_F_2_empirical()    









    



    















