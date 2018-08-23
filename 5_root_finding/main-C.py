import numpy as np
from newton import newton

def frosenbrock(x):
    return (1-x[0])**2 + 100*(x[1] - x[0]**2)**2

def drosenbrock(x):
    fx = 2*(200*x[0]**3 - 200*x[0]*x[1] + x[0] - 1)
    fy = 200*(x[1] - x[0]**2)
    return np.array([fx, fy])

def rosenbrock_jacobian(x):
    J = np.empty((2, 2))
    J[0, 0] = 800*x[0]**2 - 400*(x[1] - x[0]**2) + 2
    J[1, 0] = -400*x[0]
    J[0, 1] = -400*x[0]
    J[1, 1] = 200
    return J

def dhimmelblau(x):
    fx = 2*(2*x[0]*(x[0]**2 + x[1]-11) + x[0] + x[1]**2 - 7)
    fy = 2*(x[0]**2 + 2*x[1]*(x[0] + x[1]**2 - 7) + x[1] - 11)
    return np.array([fx, fy])

def himmelblau_jacobian(x):
    J = np.empty((2, 2))
    J[0, 0] = 8*x[0]**2 + 4*(x[0]**2 + x[1] - 11) + 2
    J[1, 0] = 4*(x[0] + x[1])
    J[0, 1] = 4*(x[0] + x[1])
    J[1, 1] = 8*x[1]**2 + 4*(x[1]**2 + x[0] - 7) + 2
    return J

rosenbrock = newton(np.array([8.1, 3.1]))
sol_bt, f_calls_bt = rosenbrock.roots_analytic(drosenbrock, rosenbrock_jacobian, ['backtrack'])
sol_quad, f_calls_quad = rosenbrock.roots_analytic(drosenbrock, rosenbrock_jacobian, ['quadratic', frosenbrock])

print('Function test\t: Rosenbrock')
print('##### BACKTRACK LINESEARCH #####')
print('Global min.\t= [ 1, 1]')
print('Initial guess\t= [ 8, 3]')
print('Min. found\t=', sol_bt)
print('Func. calls\t=', f_calls_bt, '\n')

print('##### GOLDEN SECTION QUADRATIC INTERPOLATION #####')
print('Global min.\t= [ 1, 1]')
print('Initial guess\t= [ 8, 3]')
print('Min. found\t=', sol_quad)
print('Func. calls\t=', f_calls_quad)
