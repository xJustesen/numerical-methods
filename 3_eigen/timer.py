import numpy as np
import sys
from eigen import eigen

# Make symmetric matrix
N = 25  # dimension of sym. matrix
a = np.random.rand(N, N)
A_sym = (a + a.T)/2  # real symmetric NxN matrix


if sys.argv[1] == 'values':
    EVD = eigen(A_sym, N)
    EVD.jacobi_values(N-1)
elif sys.argv[1] == 'sweeps':
    EVD = eigen(A_sym, N)
    EVD.jacobi_sweeps()
else:
    print('Input argument given: ', sys.argv[1], '\ntimes.py must be given argument: "sweeps" or "values".')
