from newton import newton
import numpy as np


def drosenbrock(x):
    fx = 2*(200*x[0]**3 - 200*x[0]*x[1] + x[0] - 1)
    fy = 200*(x[1] - x[0]**2)
    return np.array([fx, fy])

def dhimmelblau(x):
    fx = 2*(2*x[0]*(x[0]**2 + x[1]-11) + x[0] + x[1]**2 - 7)
    fy = 2*(x[0]**2 + 2*x[1]*(x[0] + x[1]**2 - 7) + x[1] - 11)
    return np.array([fx, fy])

def dcamel(x):
    fx = x[0]**5 - 4.2*x[0]**3 + 4*x[0] + x[1]
    fy = x[0] + 2*x[1]
    return np.array([fx, fy])

def eq_system(x):
    A = 10000
    fx = A*x[0]*x[1] - 1
    fy = np.exp(-x[0]) + np.exp(-x[1]) - 1 - 1/A
    return np.array([fx, fy])

rosenbrock = newton(np.array([8.1, 3.1]))
sol, f_calls = rosenbrock.roots_numeric(drosenbrock)
print('Function test\t: Rosenbrock')
print('Global min.\t= [ 1, 1]')
print('Initial guess\t= [ 8, 3]')
print('Min. found\t=', sol)

himmelblau = newton(np.array([5.1, 2.1]))
sol, f_calls = himmelblau.roots_numeric(dhimmelblau)
print('\nFunction test\t: Himmelblau')
print('Local min.\t= [ 3, 2], \n\t\t  [ -2.805118, 3.131312], \n\t\t  [ -3.779310, -3.283186], \n\t\t  [ 3.584428, -1.848126]')
print('Initial guess\t= [ 5, 2]')
print('Min. found\t=', sol)

camel = newton(np.array([1.5, 0.5]))
sol, f_calls = camel.roots_numeric(dcamel)
print('\nFunction test\t: Three-hump camel function')
print('Local min.\t= [ 0, 0],\n\t\t  [ -1.74755, 0.873776],\n\t\t  [ 1.74755, -0.873776]')
print('Initial guess\t= [ 1.5, 0.5]')
print('Min. found\t=', sol)

eq_sys = newton(np.array([2.1, 1.1]))
sol, f_calls = eq_sys.roots_numeric(eq_system)
print('\nSolution of equation system')
print('Exact sol.\t= [ 9, 0]')
print('Initial guess\t= [ 2, 1]')
print('Sol. found\t=', sol)
