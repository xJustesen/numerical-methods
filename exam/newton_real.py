import numpy as np
from linsolve import linsolve
from linesearch import linesearch

class newton:

    def __init__(self, xstart, method, eps = 1e-6, delta = 1e-6, linesearch = 'backtrack'):
        self.x0 = xstart
        self.eps = eps
        self.delta = delta
        self.method = method
        self.ls = linesearch

    def quasi_newton(self, f):
        x = self.x0.copy()
        n = x.size
        J = np.ones((n, n))
        fx = f(x)
        self.__build_Jacobian(n, f, x, fx, J)
        l_prev = 0
        while True:
            Dx = self.__backsub_solver(f, x, fx, J)
            x, fx, l = linesearch.backtrack(x, Dx, fx, f, lmin = 1/64)
            if np.linalg.norm(Dx) < self.delta or np.linalg.norm(fx) < self.eps:
                break
            df = f(x) - fx

            if self.method == 'Broyden':  # Broyden's method
                if l <= 1/64 and l_prev <= 1/64:
                    self.__build_Jacobian(n, f, x, fx, J)
                else:
                    self.__broyden_update(df, Dx, J)
            elif self.method == 'SR1':  # Symmetric rank-1 Update
                if l <= 1/64 and l_prev <= 1/64:
                    self.__build_Jacobian(n, f, x, fx, J)
                else:
                    self.__SR1_update(df, Dx, J)
            elif self.method == 'Newton':  # Newton's method
                self.__build_Jacobian(n, f, x, fx, J)

            l_prev = l
        return x

    def __backsub_solver(self, f, x, fx, J):  # solver with numeric Jacobian
        qr_decomp = linsolve(J)
        Dx = qr_decomp.gs_bacskub_solver(-fx)
        return Dx

    def __build_Jacobian(self, n, f, x, fx, J):
        for i in range(n):  # construct Jacobian matrix
            x[i] += self.delta
            df = f(x) - fx
            for j in range(n):
                J[j, i] = df[j] / self.delta  # approximate derivatives
            x[i] -= self.delta

    def __broyden_update(self, df, dx, J):
        J += np.dot(df - np.dot(J, dx), dx)/np.outer(dx, dx)

    def __SR1_update(self, df, dx, J):
        J += np.dot(df - np.dot(J, dx), df - np.dot(J, dx))/np.outer(df - np.dot(J, dx), dx)
