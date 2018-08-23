import numpy as np

class eigen:

    def __init__(self, M, nrows):
        self.nrows = nrows  # no. of rows
        self.e_val = np.diag(M); self.e_val.setflags(write=True)  # vector with eigenvalues
        self.e_vec = np.identity(int(M.shape[0]))  # matrix with eigenvectors
        self.A = M.copy()

    def jacobi_sweeps(self):
        sweeps = 0; rotations = 0
        while True:
            sweeps += 1; convergence_flag = True
            for p in range(self.nrows):
                for q in range(p + 1, self.nrows):
                    aqq = self.e_val[q]
                    app = self.e_val[p]
                    apq = self.A[p, q]
#                    phi = 0.5*np.arctan2(2*apq, aqq - app) + np.pi/2  # find largest eigenvalue first
                    phi = 0.5*np.arctan2(2*apq, aqq - app)  # find smallest eigenvalue first
                    c = np.cos(phi); s = np.sin(phi)
                    app1 = c*c*app - 2*s*c*apq + s*s*aqq
                    aqq1 = s*s*app + 2*s*c*apq + c*c*aqq

                    if app1 != app or aqq1 != aqq:
                        convergence_flag = False; rotations += 1
                        self.e_val[p] = app1
                        self.e_val[q] = aqq1
                        self.A[p, q] = 0

                        for i in range(p):
                            aip = self.A[i, p]
                            aiq = self.A[i, q]
                            self.A[i, p] = c*aip - s*aiq
                            self.A[i, q] = c*aiq + s*aip

                        for i in range(p+1, q):
                            api = self.A[p, i]
                            aiq = self.A[i, q]
                            self.A[p, i] = c*api - s*aiq
                            self.A[i, q] = c*aiq + s*api

                        for i in range(q+1, self.nrows):
                            api = self.A[p, i]
                            aqi = self.A[q, i]
                            self.A[p, i] = c*api - s*aqi
                            self.A[q, i] = c*aqi + s*api

                        for i in range(self.nrows):
                            e_vec_ip = self.e_vec[i, p]
                            e_vec_iq = self.e_vec[i, q]
                            self.e_vec[i, p] = c*e_vec_ip - s*e_vec_iq
                            self.e_vec[i, q] = c*e_vec_iq + s*e_vec_ip

            if convergence_flag:
                break

        return sweeps, rotations

    def jacobi_values(self, n_vals):
        sweeps = 0; rotations = 0
        for p in range(n_vals):
            while True:
                sweeps += 1
                convergence_flag = True
                q = p + 1

                # Find largest entry in column:
                i_entry = abs(self.A[p, q])
                for i in range(q+1, self.nrows):
                    new_entry = abs(self.A[p, i])
                    if new_entry > i_entry:
                        i_entry = new_entry
                        q = i

                # Same as jacobi_sweeps
                aqq = self.e_val[q]
                app = self.e_val[p]
                apq = self.A[p, q]
#                phi = 0.5*np.arctan2(2*apq, aqq - app) + np.pi/2  # find largest eigenvalue first
                phi = 0.5*np.arctan2(2*apq, aqq - app)  # find smallest eigenvalue first
                c = np.cos(phi); s = np.sin(phi)
                app1 = c*c*app - 2*s*c*apq + s*s*aqq
                aqq1 = s*s*app + 2*s*c*apq + c*c*aqq

                if app1 != app or aqq1 != aqq:
                    convergence_flag = False; rotations += 1
                    self.e_val[p] = app1
                    self.e_val[q] = aqq1
                    self.A[p, q] = 0

                    for i in range(p+1, q):
                        api = self.A[p, i]
                        aiq = self.A[i, q]
                        self.A[p, i] = c*api - s*aiq
                        self.A[i, q] = c*aiq + s*api

                    for i in range(q+1, self.nrows):
                        api = self.A[p, i]
                        aqi = self.A[q, i]
                        self.A[p, i] = c*api - s*aqi
                        self.A[q, i] = c*aqi + s*api

                    for i in range(self.nrows):
                        e_vec_ip = self.e_vec[i, p]
                        e_vec_iq = self.e_vec[i, q]
                        self.e_vec[i, p] = c*e_vec_ip - s*e_vec_iq
                        self.e_vec[i, q] = c*e_vec_iq + s*e_vec_ip

                if convergence_flag:
                    break

        return sweeps, rotations
