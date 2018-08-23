import numpy as np

class linsolve:
    '''
    Solves linear eq. systems and finds matrix inverse by computing the (Q, R) decomposition
    using the Gram–Schmidt orthogonalization.
    '''

    def __init__(self, M, b = None, Minv = None):
        self.nrows = int(M.shape[0])  # no. of rows
        self.ncols = int(M.shape[1])  # no. of cols.
        self.R = np.zeros((self.ncols, self.ncols))

        # Do Q, R decomposition, note the original M is destroyed
        self.__qr_decompose(M)

        # If a vector b is given, thenn solv equation system (b -> x)
        if b is not None:
            self.gs_solver(M, b)

        # If square matrix and Minv is allocated then calculate inverse:
        if self.ncols == self.nrows and Minv is not None: self.gs_inv(M, Minv)

    def __qr_decompose(self, M):
        # Do Q, R decomposition, note that M -> Q
        for i in range(self.ncols):
            self.R[i, i] = np.linalg.norm(M[:, i])
            M[:, i] /= self.R[i, i]
            for j in range(i+1, self.ncols):
                self.R[i, j] = np.dot(M[:, i], M[:, j])
                M[:, j] -= M[:, i]*self.R[i, j]

    def gs_solver(self, Q, b):
        b[:] = np.dot(Q.T, b)
        for i in reversed(range(self.ncols)):
            b[i] /= self.R[i, i]
            for j in range(i + 1, self.ncols):
                b[i] -=  b[j]*self.R[i, j]/self.R[i, i]

    def gs_inv(self, Q, Minv):
        eye = np.identity(self.ncols)
        for i in range(self.ncols):
            Minv[:, i] =  eye[:, i]
            self.gs_solver(Q, Minv[:, i])
