import numpy as np


class linesearch:
    ''' Algorihms for exact line search '''
    def backtrack(x, Dx, fx, f):  # backtrack linesearch
        l = 1
        while True:
            y = x + Dx*l
            fy = f(y)
            if np.linalg.norm(fy) < (1 - l/2)*np.linalg.norm(fx) or l < 1/64:
                break
            l /= 2
        return y, fy

    def quadratic(f, startpoint, delta, eps):  # see book: Introduction to Optimum Design

        # determine interval using golden section search
        r = 1.618
        fold = f(startpoint)
        i = 0
        while True:
            alpha = delta*r**i  # step increment
            fnew = f(startpoint + alpha)
            if fnew >= fold:
                break
            else:
                fold = fnew
                i = i+1
        # define upper (up) and lower (low) limit of golden section
        up = alpha + startpoint
        if i > 1:
            low = startpoint + alpha/(r**2) * delta
        else:
            low = startpoint

        # find minimum using quadratic interpolation in interval
        ai = low + ((up - low)/2)  # mid-point in interval
        candidate = ai  # initial candiadate minimum point
        prev_candidate = startpoint  # previous candidate minimum point
        dif = np.linalg.norm(candidate-prev_candidate)

        while dif > eps:
            # parameters for quad. poly. a0 + a1*x + a2*x²
            a2 = (1/(up-ai)) * ((f(up)-f(low))/(up-low) - (f(ai)-f(low))/(ai-low))
            a1 = ((f(ai)-f(low))/(ai-low)) - a2 * (low+ai)
            a0 = f(low) - a1*low - a2*low**2

            abar = -a1/(2*a2)  # candidate minimum from interpolation
            fbar = f(abar)  # function evaluated at interpoltation candidate
            fai = a0 + a1*ai + a2*ai**2  # poly at ai

            # determine new section based on candidate minimum
            if ai < abar:
                if fai < fbar:
                    up = abar
                else:
                    low = ai
                    ai = abar
            else:
                if fai < fbar:
                    low = abar
                else:
                    up = ai
                    ai = abar

            # update candidate, dif, previous candidate
            candidate = abar
            dif = np.linalg.norm(candidate-prev_candidate)
            prev_candidate = candidate
        return candidate
