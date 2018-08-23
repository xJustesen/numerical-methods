from interpolate import interpolate as interp
import numpy as np

N = 500
n = 20

x = np.linspace(0, 15, n)
y = np.cos(x)
z = np.linspace(0, 15, N)  # points for interpolation

interpcos = interp(x, y)  # initialze class interp

S_lin, S_lin_int = interpcos.lin(z)  # linear interpolation

for i in range(0, N):
    if i < n:
        print(str(z[i]) + ' ' + str(S_lin[i]) + ' ' + str(S_lin_int[i]) + ' ' + str(x[i]) + ' ' + str(y[i]))
    else:
        print(str(z[i]) + ' ' + str(S_lin[i]) + ' ' + str(S_lin_int[i]))
