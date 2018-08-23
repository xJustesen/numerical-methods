import numpy as np

class integrator:

    def __init__(self, func, a, b):
        assert len(a) == len(b)
        self.f = func
        self.a = a
        self.b = b

    def __randx(self):
        randx = []
        for i in range(len(self.a)):
            randx.append(self.a[i] + np.random.random()*(self.b[i] - self.a[i]))
        return randx

    def plain_mc(self, N):
        vol = 1; S1 = 0; S2 = 0;
        for i in range(len(self.a)): vol *= self.b[i] - self.a[i]  # calulate volume
        for i in range(N): fx = self.f(self.__randx()); S1 += fx; S2 += fx**2  # calculate sum
        mean = S1/N
        Sigma = np.sqrt(S2/N - mean**2)/np.sqrt(N)
        return mean*vol, Sigma*vol  # integral, error
