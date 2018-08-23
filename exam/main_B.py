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

test_complex = newton_complex_1D(startpoint, eps = eps, method = 'Broyden')
test_real = newton_real(startpoint_real, 'Broyden', linesearch = 'quadratic')

print("Rootfinding using quasi Newton's method with Broyden's update\n")

print('######## Test function: f(z) = z*z ########## \n')
nevals = 0
sol = test_complex.quasi_newton(complex_square)
print('Quasi Newton using complex-valued functions (1D)')
print('Initial guess\t:', "%.5f + %.5fj"%(startpoint.real,startpoint.imag))
print('Intial f(z)\t:', "%.5f + %.5fj"%(complex_square(startpoint).real,complex_square(startpoint).imag))
print('Min. found\t:', "%.5f + %.5fj"%(sol.real,sol.imag))
print('Final f(z)\t:', "%.5f + %.5fj"%(complex_square(sol).real,complex_square(sol).imag))
print('Precision used\t:', "%.5f + %.5fj"%(eps.real, eps.imag))
print('Function evals\t:', nevals)

print('\nQuasi Newton using real-valued functions (2D) (exercise 5)')
nevals = 0
sol = test_real.quasi_newton(complex_square_real)
print('Initial guess\t:', startpoint_real)
print('Intial f(z)\t:', complex_square_real(startpoint_real))
print('Min. found\t:', sol)
print('Final f(z)\t:', complex_square_real(sol))
print('Precision used\t:', 1e-4)
print('Function evals\t:', nevals)
print('')

print('######## Test function: f(z) = exp(z) - (1+1i) ########## \n')
nevals = 0
sol = test_complex.quasi_newton(complex_exp)
print('Quasi Newton using complex-valued functions (1D)')
print('Initial guess\t:', "%.5f + %.5fj"%(startpoint.real,startpoint.imag))
print('Intial f(z)\t:', "%.5f + %.5fj"%(complex_exp(startpoint).real,complex_exp(startpoint).imag))
print('Min. found\t:', "%.5f + %.5fj"%(sol.real,sol.imag))
print('Final f(z)\t:', "%.5f + %.5fj"%(complex_exp(sol).real, complex_exp(sol).imag))
print('Precision used\t:', "%.5f + %.5fj"%(eps.real, eps.imag))
print('Function evals\t:', nevals)

print('\nQuasi Newton using real-valued functions (2D) (exercise 5)')
nevals = 0
sol = test_real.quasi_newton(complex_exp_real)
print('Initial guess\t:', startpoint_real)
print('Intial f(z)\t:', complex_exp_real(startpoint_real))
print('Min. found\t:', sol)
print('Final f(z)\t:', complex_exp_real(sol))
print('Precision used\t:', 1e-4)
print('Function evals\t:', nevals)
print('')

print('######## Test function: f(z) = sin(z) + cos(z) ########## \n')
nevals = 0
sol = test_complex.quasi_newton(complex_harmonic)
print('Quasi Newton using complex-valued functions (1D)')
print('Initial guess\t:', "%.5f + %.5fj"%(startpoint.real,startpoint.imag))
print('Intial f(z)\t:', "%.5f + %.5fj"%(complex_harmonic(startpoint).real,complex_harmonic(startpoint).imag))
print('Min. found\t:', "%.5f + %.5fj"%(sol.real,sol.imag))
print('Final f(z)\t:', "%.5f + %.5fj"%(complex_harmonic(sol).real, complex_harmonic(sol).imag))
print('Precision used\t:', "%.5f + %.5fj"%(eps.real, eps.imag))
print('Function evals\t:', nevals)

print('\nQuasi Newton using real-valued functions (2D) (exercise 5)')
nevals = 0
sol = test_real.quasi_newton(complex_harmonic_real)
print('Initial guess\t:', startpoint_real)
print('Intial f(z)\t:', complex_harmonic_real(startpoint_real))
print('Min. found\t:', sol)
print('Final f(z)\t:', complex_harmonic_real(sol))
print('Precision used\t:', 1e-4)
print('Function evals\t:', nevals)
print('')
