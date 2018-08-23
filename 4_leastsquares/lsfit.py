import numpy as np
from linsolve import linsolve

class lsfit:
    
    def __init__(self, f, x, y, dy):
        self.nrows = x.shape[0]
        self.ncols = len(f)
        self.A = np.zeros((self.nrows, self.ncols))
        self.b = np.zeros(self.nrows)
        
        for i in range(self.nrows):
            self.b[i] = y[i]/dy[i]
            for k in range(self.ncols):
                self.A[i, k] = f[k](x[i])/dy[i]


    def ordinary_lsfit(self):
        # determine fit coefficients:
        decomp_A = linsolve(self.A)  # (Q, R) decomp of A
        c = decomp_A.gs_bacskub_solver(self.b)
        # determine covariance matrix:
        decomp_R = linsolve(decomp_A.R)  # initialize QR decomp of R to find inverse of R
        Rinv = decomp_R.gs_inv()
        S = Rinv * Rinv.T  # covariance matrix
        return c, S
    