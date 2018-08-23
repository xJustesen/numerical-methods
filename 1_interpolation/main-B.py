from interpolate import interpolate as interp
import numpy as np

N = 500
n = 20

x = np.linspace(0, 15, n)
y = np.cos(x)
z = np.linspace(0, 15, N)  # points for interpolation

interpcos = interp(x, y)  # initialze class interp

S_quad, S_quad_int, S_quad_der = interpcos.quad(z)  # quadratic interpolation

for i in range(0, N):
    if i < n:
        print(str(z[i]) + ' ' + str(S_quad[i]) + ' ' + str(S_quad_int[i]) + ' ' + str(S_quad_der[i]) + ' ' + str(x[i]) + ' ' + str(y[i]))
    else:
        print(str(z[i]) + ' ' + str(S_quad[i]) + ' ' + str(S_quad_int[i]) + ' ' + str(S_quad_der[i]))
