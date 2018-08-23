import numpy as np


class linesearch:
    ''' Algorihms for exact line search '''
    def backtrack(x, Dx, fx, f, lmin = 1/64):  # backtrack linesearch
        l = 1
        while True:
            y = x + Dx*l
            fy = f(y)
            if np.linalg.norm(fy) < (1 - l/2)*np.linalg.norm(fx) or l < lmin:
                break
            l /= 2
        return y, fy, l
