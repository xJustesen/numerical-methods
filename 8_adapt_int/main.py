import numpy as np
from integrator import integrator
import sys
sys.setrecursionlimit(10000)

a = 0; b = 1

nevals = 0
def f1(x):
    global nevals
    nevals += 1
    return np.sqrt(x)

integral1 = integrator(f1)
Q = integral1.adaptive_trapz(a, b)
print('Function:\t\t f(x) = sqrt(x)')
print('Absolute accuracy:\t', 1e-4)
print('Relative accuracy:\t', 1e-4)
print('Integral bounds:\t', a, 4)
print('Integrand evaluations: \t', nevals)
print('Numeric integral:\t', Q)
print('Analytic integral: \t', 2/3)
print('')

nevals = 0
def f2(x):
    global nevals
    nevals += 1
    return 1/np.sqrt(x)

integral2 = integrator(f2)
Q = integral2.adaptive_trapz(a, b)
print('Function:\t\t f(x) = 1/sqrt(x)')
print('Absolute accuracy:\t', 1e-4)
print('Relative accuracy:\t', 1e-4)
print('Integral bounds:\t', a, b)
print('Integrand evaluations: \t', nevals)
print('Numeric integral:\t', Q)
print('Analytic integral: \t', 2)
print('')

nevals = 0
def f3(x):
    global nevals
    nevals += 1
    return np.log(x)/np.sqrt(x)

integral3 = integrator(f3)
Q = integral3.adaptive_trapz(a, b)
print('Function:\t\t f(x) = log(x)/sqrt(x)')
print('Absolute accuracy:\t', 1e-4)
print('Relative accuracy:\t', 1e-4)
print('Integral bounds:\t', a, b)
print('Integrand evaluations: \t', nevals)
print('Numeric integral:\t', Q)
print('Analytic integral: \t', -4)
print('')

nevals = 0
def f4(x):
    global nevals
    nevals += 1
    return 4*np.sqrt(1 - (1 - x)**2)

integral4 = integrator(f4, abs_acc = 1e-9, rel_acc = 1e-9)
Q = integral4.adaptive_trapz(a, b)
print('Function:\t\t f(x) = 4*sqrt(1 - (1 - x)**2)')
print('Absolute accuracy:\t', 1e-6)
print('Relative accuracy:\t', 1e-6)
print('Integral bounds:\t', a, b)
print('Integrand evaluations: \t', nevals)
print('Numeric integral:\t', Q)
print('Analytic integral: \t', np.pi)
print('')

nevals = 0
def f5(x):
    global nevals
    nevals += 1
    return np.exp(-x)

integral5 = integrator(f5)
Q = integral5.adaptive_trapz(1, np.PINF)
print('Function:\t\t f(x) = exp(-x)')
print('Absolute accuracy:\t', 1e-4)
print('Relative accuracy:\t', 1e-4)
print('Integral bounds:\t', 0, np.PINF)
print('Integrand evaluations: \t', nevals)
print('Numeric integral:\t', Q)
print('Analytic integral: \t', 1)
print('')

nevals = 0
def f6(x):
    global nevals
    nevals += 1
    return np.exp(x)

integral6 = integrator(f6)
Q = integral6.adaptive_trapz(np.NINF, 0)
print('Function:\t\t f(x) = exp(x)')
print('Absolute accuracy:\t', 1e-4)
print('Relative accuracy:\t', 1e-4)
print('Integral bounds:\t', np.NINF, 0)
print('Integrand evaluations: \t', nevals)
print('Numeric integral:\t', Q)
print('Analytic integral: \t', 1)
print('')

nevals = 0
def f7(x):
    global nevals
    nevals += 1
    return (1/np.sqrt(2*np.pi))*np.exp(-0.5*x**2)

integral7 = integrator(f7)
Q = integral7.adaptive_trapz(np.NINF, np.PINF)
print('Function:\t\t f(x) = exp(-0.5*x^2)/sqrt(2*pi)')
print('Absolute accuracy:\t', 1e-4)
print('Relative accuracy:\t', 1e-4)
print('Integral bounds:\t', np.NINF, np.PINF)
print('Integrand evaluations: \t', nevals)
print('Numeric integral:\t', Q)
print('Analytic integral: \t', 1)
