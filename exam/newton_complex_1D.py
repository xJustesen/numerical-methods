import numpy as np

class newton:
    def __init__(self, xstart, eps = 1e-6, delta = complex(1e-6, 1e-6), method = 'Broyden'):
        self.x0 = xstart
        self.eps = eps
        self.delta = delta
        self.method = method

    def quasi_newton(self, f):
        x = self.x0
        J = (f(x + self.delta) - f(x)) / self.delta
        while True:  # Find roots
            fx = f(x)
            Dx = -fx/J
            x = self.__backtrack(x, Dx, f, fx)
            if np.absolute(Dx) < self.delta or np.absolute(fx) < self.eps:
                break
            df = f(x) - fx
            if self.method == 'Broyden' or self.method == 'SR1':
                J += (df - J*Dx)/Dx  # update Jacobian using Broyden's method
            elif self.method == 'Newton':
                J = (f(x + self.delta) - f(x)) / self.delta
        return x

    def __backtrack(self, x, Dx, f, fx):
        l = 1
        while True:
            y = x + Dx*l
            fy = f(y)
            if np.linalg.norm(fy) < (1 - l/2)*np.linalg.norm(fx) or l < 1/64:
                break
            l /= 2
        return y
