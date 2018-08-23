import numpy as np
import matplotlib.pyplot as plt
from lsfit import lsfit

def f0(x):
    return 1/x

def f1(x):
    return 1

def f2(x):
    return x

def f3(x):
    return x**2

fitfunctions = [f0, f1, f2]
fitfunctions1 = [f3, f1, f2]
x_dat = np.array([0.100, 0.145, 0.211, 0.307, 0.447, 0.649, 0.944, 1.372, 1.995, 2.900])
y_dat = np.array([12.644, 9.235, 7.377, 6.460, 5.555, 5.896, 5.673, 6.964, 8.896, 11.355])
dy_dat = np.array([0.858, 0.359, 0.505, 0.403, 0.683, 0.605, 0.856, 0.351, 1.083, 1.002])
m = len(fitfunctions)

lsfit1 = lsfit(fitfunctions, x_dat, y_dat, dy_dat)
lsfit2 = lsfit(fitfunctions1, x_dat, y_dat, dy_dat)
c, S = lsfit1.ordinary_lsfit()
c1, S1 = lsfit2.ordinary_lsfit()

dc = np.zeros(m)
for i in range(m):
    dc[i] = np.sqrt(S[i, i])

dc1 = np.zeros(m)
for i in range(m):
    dc1[i] = np.sqrt(S1[i, i])

def fit(x):
    y = 0
    for i in range(m):
        y += c[i] * fitfunctions[i](x)  # fitted function given by linear comb. of fitfunctions
    return y

def fit_up_err(x):
    y = 0
    for i in range(m):
        y += (c[i] + dc[i]) * fitfunctions[i](x)  # upper error bound
    return y

def fit_low_err(x):
    y = 0
    for i in range(m):
        y += (c[i] - dc[i]) * fitfunctions[i](x)  # lower error bound
    return y

def fit1(x):
    y = 0
    for i in range(m):
        y += c1[i] * fitfunctions1[i](x)  # fitted function given by linear comb. of fitfunctions
    return y

def fit_up_err1(x):
    y = 0
    for i in range(m):
        y += (c1[i] + dc1[i]) * fitfunctions1[i](x)  # upper error bound
    return y

def fit_low_err1(x):
    y = 0
    for i in range(m):
        y += (c1[i] - dc1[i]) * fitfunctions1[i](x)  # lower error bound
    return y


x = np.linspace(0.1, 3, 100)

for i in range(0, x.size):
    if i < x_dat.size:
        print(str(x[i]) + ' ' + str(fit(x[i])) + ' ' + str(fit_up_err(x[i])) + ' ' + str(fit_low_err(x[i])) + ' ' + str(fit1(x[i])) + ' ' + str(fit_up_err1(x[i])) + ' ' + str(fit_low_err1(x[i])) + ' ' + str(x_dat[i]) + ' ' + str(y_dat[i]) + ' ' + str(dy_dat[i]))
    else:
        print(str(x[i]) + ' ' + str(fit(x[i])) + ' ' + str(fit_up_err(x[i])) + ' ' + str(fit_low_err(x[i])) + ' ' + str(fit1(x[i])) + ' ' + str(fit_up_err1(x[i])) + ' ' + str(fit_low_err1(x[i])))

print('\nUncertainties of fit 1 =', dc ,'\n')
print('Uncertainties of fit 2 =', dc1)
