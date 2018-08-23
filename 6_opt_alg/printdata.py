from optimize import optimize
from newton import newton
import matplotlib.pyplot as plt
import numpy as np

# Function for fitting
def decay(p, t):
    return p[0] * np.exp(-t/p[1]) + p[2]

def master(t, y, s, p):
    n = t.size
    S = 0
    for i in range(n):
        S += ((decay(p, t[i]) - y[i])**2) / s[i]**2
    return S

# Input for optimization algorithms
startparams = np.array([1, 2, 3], dtype='float64')
t = np.array([0.23,1.29,2.35,3.41,4.47,5.53,6.59,7.65,8.71,9.77])
y = np.array([4.64,3.38,3.01,2.55,2.29,1.67,1.59,1.69,1.38,1.46])
s = np.array([0.42,0.37,0.34,0.31,0.29,0.27,0.26,0.25,0.24,0.24])
x = np.linspace(0, 10, 100)
eps = 1e-4
delta = 1e-6

# Do optimization
S = lambda p : master(t, y, s, p)
opt_master = optimize(S, startparams, 1e-4, 1e-6)
params, nsteps = opt_master.quasi_newton()

# Print data to terminal
for i in range(x.size):
    if i < t.size:
        print(str(x[i]),' ', decay(params, x[i]),' ', str(t[i]),' ',str(y[i]),' ',str(s[i]))
    else:
        print(str(x[i]),' ',decay(params, x[i]))
