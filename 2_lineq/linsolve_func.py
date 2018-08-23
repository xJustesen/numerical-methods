import numpy as np

def qr_decompose(M, R):
    nrows = int(M.shape[0])  # no. of rows
    ncols = int(M.shape[1])  # no. of cols.

    # Do Q, R decomposition, note the original M is destroyed
    for i in range(ncols):
        R[i, i] = np.linalg.norm(M[:, i])
        M[:, i] /= R[i, i]
        for j in range(i+1, ncols):
            R[i, j] = np.dot(M[:, i], M[:, j])
            M[:, j] -= M[:, i]*R[i, j]

def backsub(Q, R, b):
    b[:] = np.dot(Q, b)
    ncols = int(Q.shape[1])  # no. of cols.
    for i in reversed(range(ncols)):
        b[i] /= R[i, i]
        for k in range(i + 1, ncols):
            b[i] -=  b[k]*R[i, k]/R[i, i]

def gs_solver(Q, R, b):
    b = np.dot(Q, b)
    backsub(Q, R, b)
    print(b)
    return b

def gs_inv(Q, R, A_inv):
    eye = np.identity(self.ncols)
    for i in range(self.ncols):
        Ainv[:, i] = self.gs_solver(Q, R, eye[:, i])
