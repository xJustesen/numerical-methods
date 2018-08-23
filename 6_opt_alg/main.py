from optimize import optimize
from newton import newton
import matplotlib.pyplot as plt
import numpy as np

def rosenbrock(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

def dydx_rosenbrock(x):
    fx = 2*(200*x[0]**3 - 200*x[0]*x[1] + x[0] - 1)
    fy = 200*(x[1] - x[0]**2)
    return np.array([fx, fy])

def H_rosenbrock(x):
    H = np.empty((2, 2))
    H[0, 0] = 800*x[0]**2 - 400*(x[1] - x[0]**2) + 2
    H[1, 0] = -400*x[0]
    H[0, 1] = -400*x[0]
    H[1, 1] = 200
    return H

def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def dydx_himmelblau(x):
    fx = 2*(2*x[0]*(x[0]**2 + x[1]-11) + x[0] + x[1]**2 - 7)
    fy = 2*(x[0]**2 + 2*x[1]*(x[0] + x[1]**2 - 7) + x[1] - 11)
    return np.array([fx, fy])

def H_himmelblau(x):
    H = np.empty((2, 2))
    H[0, 0] = 8*x[0]**2 + 4*(x[0]**2 + x[1] - 11) + 2
    H[1, 0] = 4*(x[0] + x[1])
    H[0, 1] = 4*(x[0] + x[1])
    H[1, 1] = 8*x[1]**2 + 4*(x[1]**2 + x[0] - 7) + 2
    return H

def decay(p, t):
    return p[0] * np.exp(-t/p[1]) + p[2]

def master(t, y, s, p):
    n = t.size
    S = 0
    for i in range(n):
        S += ((decay(p, t[i]) - y[i])**2) / s[i]**2
    return S

startpoint = np.array([4, 5], dtype='float64')
startparams = np.array([1, 2, 3], dtype='float64')
t = np.array([0.23,1.29,2.35,3.41,4.47,5.53,6.59,7.65,8.71,9.77])
y = np.array([4.64,3.38,3.01,2.55,2.29,1.67,1.59,1.69,1.38,1.46])
s = np.array([0.42,0.37,0.34,0.31,0.29,0.27,0.26,0.25,0.24,0.24])
eps = 1e-6
delta = 1e-7

roots_rose = newton(startpoint, eps, delta)
xmin, nsteps = roots_rose.roots_numeric(dydx_rosenbrock)
print('###########\tROOT-FINDING\t###########')
print('Minimizing Rosenbrock\nStartpoint:\t', startpoint)
print('Found minimum:\t', xmin,'\nSteps used:\t', nsteps, '\nPrecision:\t', eps)
print('')

roots_himmel = newton(startpoint, eps, delta)
xmin, nsteps = roots_himmel.roots_numeric(dydx_himmelblau)
print('Minimizing Himmelblau\nStartpoint:\t', startpoint)
print('Found minimum:\t', xmin,'\nSteps used:\t', nsteps, '\nPrecision:\t', eps)
print('')

opt_rose = optimize(rosenbrock, startpoint, eps, delta)
xmin, nsteps = opt_rose.newton(dydx_rosenbrock, H_rosenbrock)
print('###########\tNEWTON\t\t###########')
print('Minimizing Rosenbrock\nStartpoint:\t', startpoint)
print('Found minimum:\t', xmin,'\nSteps used:\t', nsteps, '\nPrecision:\t', eps)
print('')

opt_himmel = optimize(himmelblau, startpoint, eps, delta)
xmin, nsteps = opt_himmel.newton(dydx_himmelblau, H_himmelblau)
print('Minimizing Himmelblau\nStartpoint:\t', startpoint)
print('Found minimum:\t', xmin,'\nSteps used:\t', nsteps, '\nPrecision:\t', eps)
print('')

xmin, nsteps = opt_rose.quasi_newton()
print('###########\tQUASI-NEWTON\t###########')
print('Minimizing Rosenbrock\nStartpoint:\t', startpoint)
print('Found minimum:\t', xmin,'\nSteps used:\t', nsteps, '\nPrecision:\t', eps)
print('')
xmin, nsteps = opt_himmel.quasi_newton()
print('Minimizing Himmelblau\nStartpoint:\t', startpoint)
print('Found minimum:\t', xmin,'\nSteps used:\t', nsteps, '\nPrecision:\t', eps)
print('')

S = lambda p : master(t, y, s, p)
opt_master = optimize(S, startparams, 1e-4, 1e-6)
params, nsteps = opt_master.quasi_newton()
print('Fitting decay-law\nStart-params:\t',startparams)
print('Fit params:\t', params,'\nSteps used:\t', nsteps, '\nPrecision:\t', eps)
