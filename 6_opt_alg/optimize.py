import numpy as np
from linsolve import linsolve
from linesearch import linesearch

class optimize:

    def __init__(self, objective_function, startpoint, eps, delta):
        self.f = objective_function
        self.x0 = startpoint
        self.n = startpoint.size
        self.eps = eps
        self.delta = delta

    def newton(self, gf, H):
        x = self.x0.copy()
        nsteps = 0
        self.dx = 0
        while True:
            fx = self.f(x); self.gfx = gf(x); Hx = H(x)
            qr = linsolve(Hx)
            self.dx = qr.gs_bacskub_solver(-self.gfx)
            l, a, b = linesearch.backtrack(x, self.dx, fx, self.f, self.__convergence_newton);
            x += l*self.dx
            nsteps += 1
            if np.linalg.norm(gf(x)) < self.eps: break
        return x, nsteps

    def quasi_newton(self):
        x = self.x0.copy()
        fx = self.f(x)
        H_inv = np.identity(self.n)
        self.gfx = self.__gradient(x)
        self.dx = 0
        nsteps = 0
        while True:
            self.dx = np.dot(H_inv, -self.gfx)
            l, z, fz = linesearch.backtrack(x, self.dx, fx, self.f, self.__convergence_quasi_newton)
            if np.linalg.norm(l*self.dx) < 1e-6:
                H_inv = np.identity(self.n)
            self.dx *= l
            gfz = self.__gradient(z);
            y = gfz - self.gfx
            H_inv = self.__broyden(y, H_inv)
            x = z; fx = fz; self.gfx = gfz;
            nsteps += 1
            if np.linalg.norm(self.dx) < 1e-7 or np.linalg.norm(self.gfx) < self.eps: break
        return x, nsteps

    def __gradient(self, x):
        fx = self.f(x)
        gfx = np.empty(x.size)
        for i in range(x.size):
            x[i] += self.delta
            gfx[i] = (self.f(x) - fx)/self.delta
            x[i]-= self.delta
        return gfx

    def __broyden(self, y, H_inv):
        if abs(np.dot(np.dot(y.T, H_inv), self.dx)) < 1e-9:
            H_inv = np.identity(self.n)
        else:
            H_inv += np.dot(np.dot(self.dx - np.dot(H_inv, y), self.dx.T), H_inv)/np.dot(np.dot(y.T, H_inv), self.dx)
        return H_inv

    def __convergence_newton(self, fy, fx, l):
        if np.linalg.norm(fy) < (1 - l/2)*np.linalg.norm(fx) or l < 1/64:
            flag = True
        else:
            flag = False
        return flag

    def __convergence_quasi_newton(self, fy, fx, l):
        if abs(fy) < abs(fx) + 0.001*np.dot(l*self.dx, self.gfx):
            flag = True
        elif np.linalg.norm(l*self.dx) < 1e-6:
            flag = True
        else:
            flag = False
        return flag
