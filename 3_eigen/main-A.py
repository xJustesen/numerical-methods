import numpy as np
from eigen import eigen

def printmatrix(M, nrows):
    for i in range(nrows):
        print('\t'.join(map(str, M[i, :])))


N = 5
a = np.random.rand(N, N)
A_sym = (a + a.T)/2  # real symmetric NxN matrix
As = A_sym.copy()

print('Random symmetric matrix A =\n')
printmatrix(A_sym, N)
print('\nEigenvalues found using built-in NumPy function "eig":\n')
v, w = np.linalg.eig(A_sym)
print('\t'.join(map(str, v)))

EVD = eigen(A_sym, N)
s = EVD.jacobi_sweeps()
D = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            D[i, j] = EVD.e_val[i]

print('\nA after cyclic sweeps =\n')
printmatrix(EVD.A, N)
print('\nV.T * A * V =\n')
printmatrix(np.dot(np.dot(EVD.e_vec.T, As), EVD.e_vec), N)
print('\nD (diagonal matrix with eigenvalues on diagonal) =\n')
printmatrix(D, N)
