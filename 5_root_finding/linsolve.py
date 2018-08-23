import numpy as np

class linsolve:
    ''' Solves linear eq. systems and finds matrix inverse by computing the (Q, R) decomposition
    using the Gram–Schmidt orthogonalization. '''

    def __init__(self, M):
        self.nrows = int(M.shape[0])  # no. of rows
        self.ncols = int(M.shape[1])  # no. of cols.
        self.Q = M  # relabel M to Q for convenience
        self.R = np.ones((self.ncols, self.ncols))

        # Do Q, R decomposition
        for i in range(self.ncols):
            self.R[i, i] = np.linalg.norm(self.Q[:, i])
            self.Q[:, i] /= self.R[i, i]
            for j in range(i+1, self.ncols):
                self.R[i, j] = np.dot(self.Q[:, i], self.Q[:, j])
                self.Q[:, j] -= self.Q[:, i]*self.R[i, j]

    def gs_bacskub_solver(self, b):
        c = np.dot(self.Q.T, b)
        x = np.zeros(c.size)
        x[-1] = c[-1]/self.R[-1, -1]
        for i in reversed(range(self.ncols-1)):
            for k in range(i + 1, self.ncols):
                x[i] += self.R[i, k] * x[k]
            x[i] = (c[i] - x[i]) / self.R[i, i]
        return x

    def gs_inv(self):
        Minv = np.zeros((self.nrows, self.ncols))
        eye = np.identity(self.ncols)
        for i in range(self.ncols):
            Minv[:, i] = self.gs_bacskub_solver(eye[:, i])
        return Minv
