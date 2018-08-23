import numpy as np
from ode import ode_solver as ode

def f(x, y) :
	return np.array([y[1],-y[0]])

a = 0
b = 4*np.pi
eps = 1e-6
acc = 1e-6
h = 1e-6
x = [a]
y = [np.array([0, 1])]
ode = ode(f, x, y, eps = eps, acc = acc, step = h, printinfo = False)
ode.driver(b)

for i in range(len(ode.x)):
    print(ode.x[i]/np.pi, ' ', ode.y[i][0], ' ',ode.y[i][1])
