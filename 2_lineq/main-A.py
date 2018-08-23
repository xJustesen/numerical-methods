import numpy as np
from linsolve import linsolve
from linsolve_func import gs_inv, gs_solver, qr_decompose

def printmatrix(M, nrows):
    for i in range(nrows):
        print('\t\t'.join(map(str, np.around(M[i, :], decimals = 3))))

def printvector(v):
    print('\t'.join(map(str, np.around(v, decimals = 3))))

# make random tall matrix
nrows = 8
ncols = 5
A = np.random.randn(nrows, ncols)
#R = np.zeros((ncols, ncols))
Ac = A.copy()  # make copy to check that QR = M, since A -> Q

# make Q,R decomposition of tall matrix
qr_tall = linsolve(A)

# make checks
print('\n1a:\n')
print('Check that Q is orthogonal by calculating Q^T * Q: \n')
printmatrix(np.dot(A.T, A), ncols)
print('\nCheck that R is upper triangular: \n')
printmatrix(qr_tall.R, ncols)
print('\nQR = \n')
printmatrix(np.dot(A, qr_tall.R), ncols)
print('\nA = \n')
printmatrix(Ac, ncols)

# make random square matrix and random vector b of same size
B = np.random.rand(ncols, ncols)
R = np.zeros((ncols, ncols))
Bc = B.copy()  # make copy to check that Bx = b, since B -> Q
b = np.random.rand(ncols)
bc = b.copy()

# make (Q,R) decomposition of square matrix, solve Bx = b
qr_square = linsolve(B, b)

# make checks
print('\n2a:\n')
print('Matrix A given by: \n')
printmatrix(Bc, ncols)
print('\nVector x given by:\n')
printvector(b)
print('\nMatrix product Ax given by \n')
printvector(np.dot(Bc, b))
print('\nVector b given by: \n')
printvector(bc)
