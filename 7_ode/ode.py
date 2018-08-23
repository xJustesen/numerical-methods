import numpy as np

class ode_solver:
    ''' ODE solver which supports rk45 and rk12 steppers '''
    def __init__(self, f, ix, iy, step = 1e-6, stepper=45, eps = 1e-6, acc = 1e-6, printinfo = True):
        self.x = ix.copy()
        self.y = iy.copy()
        self.h = step
        self.f = f
        self.eps = eps
        self.acc = acc
        self.stepper = stepper
        self.xlist = []
        self.ylist = []
        self.printinfo = printinfo
        if self.stepper == 45:
            # Construct butcher's tableu for rkf45
            a1 = [1/4]
            a2 = [3/32, 9/32]
            a3 = [1932/2197, -7200/2197, 7296/2197]
            a4 = [439/216, -8, 3680/513, -845/4104]
            a5 = [-8/27, 2, -3544/2565, 1859/4104, -11/40]
            self.a = [a1, a2, a3, a4, a5]
            self.c = [1/4, 3/8, 12/13, 1, 1/2]
            self.b = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]
            self.bs = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]

    def __construct_k(self):  # private
        k = [self.f(self.x0, self.y0)]
        for i in range(len(self.a)):
            S = 0
            for j in range(i+1): S += self.a[i][j]*k[j];
            y_in = self.y0 + self.h*S
            x_in = self.x0 + self.c[len(self.a[i])-1]*self.h
            k.append(self.f(x_in, y_in))
        return k

    def __rkstep45(self):  # private
        S = 0; Se = 0
        # construct k's using values given in Butcher's tableu
        k = self.__construct_k()
        # Take one step
        for i in range(len(self.b)): S += self.b[i]*k[i];
        for i in range(len(self.b)): Se += k[i] * (self.b[i] - self.bs[i]);
        err = np.linalg.norm(self.h*Se)
        y_new = self.y0 + self.h*S
        return y_new, err

    def __rkstep12(self):  # private
        k0  = self.f(self.x0, self.y0)
        k1  = self.f(self.x0+self.h/2, self.y0+k0*(self.h/2))
        y   = self.y0+k1*self.h
        err = np.linalg.norm((k0-k1)*(self.h/2))
        return y, err

    def driver(self, endpoint):
        if self.printinfo:
            print('Using following parameters for driver:')
            print('Epsilon = \t', eps)
            print('Accuracy = \t', acc)
            print('Stepper = \t', stepper)

        startpoint = self.x[-1]
        self.nsteps = 0
        while True:
            self.x0 = self.x[-1]; self.y0 = self.y[-1]
            if self.x0 >= endpoint: break;
            if self.x0 + self.h > endpoint: self.h = endpoint - self.x0;
            if self.stepper == 45: y_new, err = self.__rkstep45();
            if self.stepper == 12: y_new, err = self.__rkstep12();
            tol = (self.acc + np.linalg.norm(y_new)*self.eps)*np.sqrt(self.h/(endpoint-startpoint))
            if err < tol:  # accept y_new and continue stepping
                self.nsteps += 1
                self.x.append(self.x0 + self.h)
                self.y.append(y_new)
                self.ylist.append(self.y)
                self.xlist.append(self.x)
            if err == 0:
                self.h *= 2
            else:
                self.h *= ((tol/err)**0.25)*0.95
