import numpy as np


class interpolate:
    """ Supports linear and quadratic interpolation """

    def __init__(self, x, y):  # initialize
        self.x = x
        self.y = y

        # build spline for quad.
        n = int(np.size(self.x))
        self.c = np.zeros(np.size(self.x))
        self.b = np.zeros(np.size(self.x))
        h = np.zeros(n-1)
        p = np.zeros(n-1)

        for i in range(0, n-1):  # find find interval h and slope p
            h[i] = self.x[i+1] - self.x[i]
            p[i] = (self.y[i+1] - self.y[i])/h[i]

        for i in range(0, n-2):  # forward recursion
            self.c[i+1] = (p[i+1]-p[i]-self.c[i]*h[i])/h[i+1]

        self.c[n-2] = self.c[n-2]/2  # this ensures that we take mean of backward + forward
        for i in range(n-3, -1, -1):  # backward recursion
            self.c[i] = (p[i+1]-p[i]-self.c[i+1]*h[i+1])/h[i]

        for i in range(0, n-1):
            self.b[i] = p[i] - self.c[i]*h[i]


    def __binarysearch(self, z, n):  # binary serach algorithm (private)
        i = 0
        j = n-1
        while j - i > 1:
            m = int((i+j)/2)  # middle index
            if z > self.x[m]:  # do binary search
                i = m
            else:
                j = m
        p = (self.y[i+1] - self.y[i])/(self.x[i+1] - self.x[i])  # calculate slope
        return p, i

    def __lin_int(self, z, n):  # linear integral (private)
        integral = 0
        p, i = self.__binarysearch(z, n)
        j = 0
        while j < i:  # evaluate integral
            h = self.x[j+1] - self.x[j]
            integral = integral+self.y[j]*h+0.5*(self.y[j+1]-self.y[j])*h**2
            j = j + 1
        integral = integral+self.y[i]*(z-self.x[i])+(1/2)*(self.y[i+1]-self.y[i])*(z-self.x[i])**2
        return integral

    def __quad_int(self, z, n):  # quadratic integral (private)
        integral = 0
        p, i = self.__binarysearch(z, n)
        j = 0
        while j < i:  # evaluate integral
            h = self.x[j+1] - self.x[j]
            integral = integral + self.y[j]*h + 0.5*self.b[j]*h**2 + (1/3)*self.c[j]*h**2
            j = j + 1
        hi = (z - self.x[i])
        bi = (self.y[j+1]-self.y[j]) - self.c[i]*hi
        integral = integral + self.y[i]*hi + 0.5*self.b[i]*hi**2 + (1/3)*self.c[i]*hi**2
        return integral

    def quad(self, z):  # quadratic interpolation (public)
        S = np.zeros(np.size(z))
        SI = np.zeros(np.size(z))
        SD = np.zeros(np.size(z))
        n = int(np.size(self.x))
        for j in range(0, np.size(z)):
            p, i = self.__binarysearch(z[j], n)
            S[j] = self.y[i] + self.b[i]*(z[j]-self.x[i]) + self.c[i]*(z[j]-self.x[i])**2  # func
            SD[j] = self.b[i] + 2*self.c[i]*(z[j]-self.x[i])  # derivative
            SI[j] = self.__quad_int(z[j], n)  # integral
        return S, SI, SD

    def lin(self, z):  # linear interpolation (public)
        S = np.zeros(np.size(z))
        SI = np.zeros(np.size(z))
        n = int(np.size(self.x))
        for j in range(0, np.size(z)):
            p, i = self.__binarysearch(z[j], n)
            S[j] = self.y[i] + p * (z[j] - self.x[i])  # func
            SI[j] = self.__lin_int(z[j], n)  # integral
        return S, SI
