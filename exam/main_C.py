from newton_complex_1D import newton as newton_complex_1D
from newton_real import newton as newton_real
import numpy as np

startpoint = complex(1.0, 2.0)
startpoint_real = np.array([1.0, 2.0])
eps = complex(1e-6, 1e-6)
np.set_printoptions(precision = 5)

def complex_square(z):
    global nevals
    nevals += 1
    return pow(z, 2)

def complex_square_real(x):
    global nevals
    nevals += 1
    freal = np.real(pow(x[0] + x[1]*1j, 2))
    fimag = np.imag(pow(x[0] + x[1]*1j, 2))
    return np.array([freal, fimag])

def complex_exp(z):
    global nevals
    nevals += 1
    return np.exp(z) - complex(1.0, 1.0)

def complex_exp_real(x):
    global nevals
    nevals += 1
    freal = np.real(np.exp(x[0] + x[1]*1j) - complex(1.0, 1.0))
    fimag = np.imag(np.exp(x[0] + x[1]*1j) - complex(1.0, 1.0))
    return np.array([freal, fimag])

def complex_harmonic(z):
    global nevals
    nevals += 1
    return np.sin(z) + np.cos(z)

def complex_harmonic_real(x):
    global nevals
    nevals += 1
    freal = np.real(np.sin(x[0] + 1j*x[1]) + np.cos(x[0] + 1j*x[1]))
    fimag = np.imag(np.sin(x[0] + 1j*x[1]) + np.cos(x[0] + 1j*x[1]))
    return np.array([freal, fimag])

test_complex_Broyden = newton_complex_1D(startpoint, eps = eps, method = 'Broyden')
test_complex_Newton = newton_complex_1D(startpoint, eps = eps, method = 'Newton')
test_complex_SR1 = newton_complex_1D(startpoint, eps = eps, method = 'SR1')

test_real_Broyden = newton_real(startpoint_real, 'Broyden')
test_real_SR1 = newton_real(startpoint_real, 'SR1')
test_real_Newton = newton_real(startpoint_real, 'Newton')

print("Comparison between using quasi Newton's method with symmetric rank-1 (SR1) update, Broyden's method and using Newton's method.\n")
print('Initial guess\t:', "%.5f + %.5fj"%(startpoint.real,startpoint.imag))
print('Precision used\t:', "%.5f + %.5fj"%(eps.real, eps.imag))

print('\n######## Test function: f(z) = z*z ########## \n')
print('\n1D complex-valued function:\n')
nevals = 0
sol = test_complex_Broyden.quasi_newton(complex_square)
print('Root found with Broyden update is \t(', "%.5f + %.5fj"%(sol.real,sol.imag), ')\t with', nevals, 'function evalulations')
nevals = 0
sol = test_complex_SR1.quasi_newton(complex_square)
print('Root found with SR1 update is \t\t(', "%.5f + %.5fj"%(sol.real,sol.imag), ')\t with', nevals, 'function evalulations')
nevals = 0
sol = test_complex_Newton.quasi_newton(complex_square)
print('Root found with Newton method is \t(', "%.5f + %.5fj"%(sol.real,sol.imag), ')\t with', nevals, 'function evalulations')

print('\n2D real-valued function:\n')
nevals = 0
sol = test_real_Broyden.quasi_newton(complex_square_real)
print('Root found with Broyden update is\t', sol, '\twith', nevals, 'function evalulations')
nevals = 0
sol = test_real_SR1.quasi_newton(complex_square_real)
print('Root found with SR1 update is\t\t', sol, '\twith', nevals, 'function evalulations')
nevals = 0
sol = test_real_Newton.quasi_newton(complex_square_real)
print('Root found with Newton method is\t', sol, '\twith', nevals, 'function evalulations')
print('')

print('######## Test function: f(z) = exp(z) - (1+1i) ########## \n')
print('\n1D complex-valued function:\n')
nevals = 0
sol = test_complex_Broyden.quasi_newton(complex_exp)
print('Root found with Broyden update is \t(', "%.5f + %.5fj"%(sol.real,sol.imag), ')\t with', nevals, 'function evalulations')
nevals = 0
sol = test_complex_SR1.quasi_newton(complex_exp)
print('Root found with SR1 update is \t\t(', "%.5f + %.5fj"%(sol.real,sol.imag), ')\t with', nevals, 'function evalulations')
nevals = 0
sol = test_complex_Newton.quasi_newton(complex_exp)
print('Root found with Newton method is \t(', "%.5f + %.5fj"%(sol.real,sol.imag), ')\t with', nevals, 'function evalulations')

print('\n2D real-valued function:\n')
nevals = 0
sol = test_real_Broyden.quasi_newton(complex_exp_real)
print('Root found with Broyden update is\t', sol, '\twith', nevals, 'function evalulations')
nevals = 0
sol = test_real_SR1.quasi_newton(complex_exp_real)
print('Root found with SR1 update is\t\t', sol, '\twith', nevals, 'function evalulations')
nevals = 0
sol = test_real_Newton.quasi_newton(complex_exp_real)
print('Root found with Newton method is\t', sol, '\twith', nevals, 'function evalulations')
print('')

print('######## Test function: f(z) = sin(z) + cos(z) ########## \n')
print('\n1D complex-valued function:\n')
nevals = 0
sol = test_complex_Broyden.quasi_newton(complex_harmonic)
print('Root found with Broyden update is \t(', "%.5f + %.5fj"%(sol.real,sol.imag), ')\t with', nevals, 'function evalulations')
nevals = 0
sol = test_complex_SR1.quasi_newton(complex_harmonic)
print('Root found with SR1 update is \t\t(', "%.5f + %.5fj"%(sol.real,sol.imag), ')\t with', nevals, 'function evalulations')
nevals = 0
sol = test_complex_Newton.quasi_newton(complex_harmonic)
print('Root found with Newton method is \t(', "%.5f + %.5fj"%(sol.real,sol.imag), ')\t with', nevals, 'function evalulations')

print('\n2D real-valued function:\n')
nevals = 0
sol = test_real_Broyden.quasi_newton(complex_harmonic_real)
print('Root found with Broyden update is\t', sol, '\twith', nevals, 'function evalulations')
nevals = 0
sol = test_real_SR1.quasi_newton(complex_harmonic_real)
print('Root found with SR1 update is\t\t', sol, '\twith', nevals, 'function evalulations')
nevals = 0
sol = test_real_Newton.quasi_newton(complex_harmonic_real)
print('Root found with Newton method is\t', sol, '\twith', nevals, 'function evalulations')
print('')
