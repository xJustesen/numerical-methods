import numpy as np

class integrator:

    def __init__(self, f, rel_acc = 1e-4, abs_acc = 1e-4):
        self.eps = rel_acc
        self.delta = abs_acc
        self.f = f
        self.x = [1/6, 2/6, 4/6, 5/6]  # reusable points on interval [0, 1]
        self.w = [2/6, 1/6, 1/6, 2/6]  # trapezoid weights on interval [0, 1]
        self.v = [1/4, 1/4, 1/4, 1/4]  # rectangle weights on interval [0, 1]

    def __check_limits(self, a, b, f):
        if np.any(np.isinf([a, b])): # do variable transformation if integral is improper
            if a is np.NINF and b is not np.PINF:
                self.f = lambda t : f(b + t/(1+t))*(1/(1+t)**2)
                a = -1; b = 0;
            elif b is np.PINF and a is not np.NINF:
                self.f = lambda t : f(a + t/(1-t))*(1/(1 - t)**2)
                a = 0; b = 1
            else:
                self.f = lambda t : f(t/(1-t**2))*((1+t**2)/(1 - t**2)**2)
                a = -1; b = 1
        return a, b

    def __weighted_sum(self, w, v, x):
        S1 = 0; S2 = 0;
        for i in range(len(w)): S1 += w[i]*self.f(x[i]);
        for i in range(len(v)): S2 += v[i]*self.f(x[i]);
        dS = abs(S1 - S2)
        return S1, S2, dS

    def __rescale(self, a, b, w, v):  # rescales points and weights from interval [0, 1] to [a, b]
        wrs = []; vrs = []; xrs = []
        for i in range(len(self.w)):
            wrs.append((b - a)*w[i])
            vrs.append((b - a)*v[i])
            xrs.append(a + (b - a)*self.x[i])
        return wrs, vrs, xrs

    def __recursive_integrator(self, a, b):
        if a == 0 and b == 1: w = self.w; v = self.v; x = self.x
        else: w, v, x = self.__rescale(a, b, self.w, self.v)
        Q, q, dQ = self.__weighted_sum(w, v, x)
        tol = self.delta + abs(Q)*self.eps
        if dQ < tol:  # accept integral
            return Q
        else:  # recursively update interval and absolute accuracy delta until integral is accepted
            self.delta /= np.sqrt(2)
            Q1 = self.__recursive_integrator(a, (a+b)/2)
            Q2 = self.__recursive_integrator((a+b)/2, b)
            return Q1 + Q2

    def adaptive_trapz(self, a, b):
        a, b = self.__check_limits(a, b, self.f)
        Q = self.__recursive_integrator(a, b)
        return Q
