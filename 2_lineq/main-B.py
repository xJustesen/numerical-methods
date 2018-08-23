import numpy as np
from linsolve import linsolve

def printmatrix(M, nrows):
    for i in range(nrows):
        print('\t\t'.join(map(str, np.around(M[i, :], decimals = 3))))

def printvector(v):
    print('\t'.join(map(str, np.around(v, decimals = 3))))

# make square matrix
m = 5
A = np.random.rand(m, m)
Ainv = np.random.rand(m, m)
Ac = A.copy()

# make Q,R decomposition and find inverse
qr = linsolve(A, Minv = Ainv)

# make checks
print('b:')
print("A^(-1) =\n")
printmatrix(Ainv, m)
print('\nA^(-1)A = \n')
printmatrix(np.dot(Ainv, Ac), m)
