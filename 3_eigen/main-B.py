import numpy as np
from eigen import eigen

def printmatrix(M, nrows):
    for i in range(nrows):
        print('\t'.join(map(str, M[i, :])))

def make_D(N, e):
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                D[i, j] = e[i]
    return D

# Make symmetric matrix and find eigenvalues/eigenvectors using built-in NumPy  method
N = 5  # dimension of sym. matrix
a = np.random.rand(N, N)
A_sym = (a + a.T)/2  # real symmetric NxN matrix
Ab = A_sym.copy()
Ac = A_sym.copy()
Ad = A_sym.copy()
v, w = np.linalg.eig(A_sym)  # calculate eigenvalues using built-in NumPy method

# Print some stuff to output
print('Random symmetric matrix A =\n')
printmatrix(A_sym, N)

print('\nEigenvalues found using built-in NumPy function "eig":\n')
print('\t'.join(map(str, v)))

##################################  Ba  #######################################
na = 1
EVDa = eigen(A_sym, N)
sa, rota = EVDa.jacobi_values(na)
Da = make_D(N, EVDa.e_val)

print('\na) Find lowest eigenvalue of A:\n')
printmatrix(Da, N)
print('\nA after cyclic sweeps:\n')
printmatrix(EVDa.A, N)
print('\nThe smallest eigenvalue is found first due to the input given to atan2. The first input given is the element we are making zero, and the second input is the difference between the corresponding diagonal element in the current row and the diagonal element in the next row. When the first input given to atan2 is 0, atan2 will return either pi or 0 if the second input is < 0 or > 0. If the second input is > 0, then atan2 returns 0 and the algorithm will make no rotations. If, on the other hand, the second input is < 0 then +pi is returned and we make a series (sweep) of Jacobi rotations.\nWe can make the algorithm find the largest eigenvalue first by adding pi/2 to phi.')

##################################  Bb  #######################################
nb = 2
EVDb = eigen(Ab, N)
EVDb.jacobi_values(nb)
Db = make_D(N, EVDb.e_val)

print('\nb) Find 2 lowest eigenvalue of A:\n')
printmatrix(Db, N)
print('\nA after cyclic sweeps:\n')
printmatrix(EVDb.A, N)

##################################  Bc  #######################################
nc = N-1
EVDc = eigen(Ac, N)
EVDc2 = eigen(Ad, N)
EVDc.jacobi_values(nc)
sc, rotc = EVDc2.jacobi_sweeps()

Dc = make_D(N, EVDc.e_val)

print('\nc) Find remaining eigenvalues of A:\n')
printmatrix(Dc, N)
print('\nA after cyclic sweeps (zeros over diagnoal) =\n')
printmatrix(EVDc.A, N)
print('\nNumber of rotations to find lowest eigenvalue using value-by-value =', rota, '\n')
print('Number of rotations to fully diagonalize matrix using cyclic sweeps =', rotc)
