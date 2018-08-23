import numpy as np
from linsolve import linsolve
from linesearch import linesearch

class newton:

    def __init__(self, xstart):
        self.x0 = xstart
        self.eps = 1e-3
        self.delta = 1e-6

    def __numeric_J_solver(self, f, x):  # solver with numeric Jacobian
        n = x.size
        J = np.ones((n, n))  # Jacobian matrix
        fx = f(x)
        for i in range(n):  # construct Jacobian matrix
            x[i] += self.delta
            df = f(x) - fx
            for j in range(n):
                J[j, i] = df[j] / self.delta  # approximate derivatives
            x[i] -= self.delta
        qr_decomp = linsolve(J)
        Dx = qr_decomp.gs_bacskub_solver(-fx)
        return fx, x, Dx

    def __analytic_J_solver(self, f, x, J):  # solver with numeric Jacobian
        Jx = J(x)
        fx = f(x)
        qr_decomp = linsolve(Jx)
        Dx = qr_decomp.gs_bacskub_solver(-fx)
        return fx, x, Dx

    def roots_numeric(self, f):  # Newton algorithm with numeric Jacobian and backtrack line search
        x = self.x0.copy()
        f_calls = 0
        while True:
            fx, x, Dx = self.__numeric_J_solver(f, x)
            x, fx = linesearch.backtrack(x, Dx, fx, f)
            if np.linalg.norm(Dx) < self.delta or np.linalg.norm(fx) < self.eps:
                break
            f_calls += 3
        return x, f_calls

    def roots_analytic(self, f, J, ls):  # Newton algorithm with numeric Jacobian and backtrack line search
        x = self.x0.copy()
        f_calls = 0
        while True:
            fx, x, Dx = self.__analytic_J_solver(f, x, J); f_calls += 1
            if ls[0] == 'backtrack':
                x, fx = linesearch.backtrack(x, Dx, fx, f); f_calls += 1
            elif ls[0] == 'quadratic':
                f_calls += 1
                def fp(alpha):  # re-define f as a function of step-size alpha
                    return ls[1](x + alpha*Dx)
                a = linesearch.quadratic(fp, 1, 0.0001, 0.0001); f_calls += 1
                x = x + a*Dx  # update x
                fx = f(x); f_calls += 1
            if np.linalg.norm(Dx) < self.delta or np.linalg.norm(fx) < self.eps:
                break
        return x, f_calls
