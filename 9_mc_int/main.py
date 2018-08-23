import numpy as np
import matplotlib.pyplot as plt
from mcint import integrator

def circ(x):
    if x[0]**2 + x[1]**2 < 1:
        return 1
    else:
        return 0

a = [0, 0]
b = [1, 1]
N = 10**5

circint = integrator(circ, a, b)
integral, error = circint.plain_mc(N)
print('Test funcion:\t\t x^2 + y^2 = 1')
print('Integral bounds:\t', a, '\n \t\t\t', b)
print('Analytic integral:\t', np.pi/4)
print('Numeric integral:\t', integral)
print('Numeric error:\t\t', error)
print('No. of samples:\t\t', N)

def singular(x):
    return (np.pi**(3)*(1 - np.cos(x[0])*np.cos(x[1])*np.cos(x[2])))**(-1)

a = [0, 0 ,0]
b = [np.pi, np.pi, np.pi]

singint = integrator(singular, a, b)
integral, error = singint.plain_mc(N)

print('')
print('Test funcion:\t\t 1/(pi^3 * (1 -cos(x)cos(y)cos(z)))')
print('Integral bounds:\t', a, '\n \t\t\t', b)
print('Analytic integral:\t', 1.3932039296856768591842462603255)
print('Numeric integral:\t', integral)
print('Numeric error:\t\t', error)
print('No. of samples:\t\t', N)

n = np.arange(1000, 11*10**3, 1000, dtype = np.int16); errors = []; samples = [];
for i in range(len(n)):
    integral, error = circint.plain_mc(n[i])
    errors.append(error)
    samples.append(n[i])

print('\nIf the error -> O(1/sqrt(N)) then error1/error2 = sqrt(N2)/sqrt(N1):\nTest function: x^2 + y^2 = 1 using bounds [0, 0] -> [1, 1]')
print('\nerror1/error2\t sqrt(N2)/sqrt(N1)\t N1 \t N2')
print('---------------------------------------------------------------')
for i in range(len(n)-1):
    print(errors[i]/errors[i+1],'\t',np.sqrt(samples[i+1])/np.sqrt(samples[i]),'\t\t',samples[i],'\t',samples[i+1])
