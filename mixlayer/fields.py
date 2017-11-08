import numpy as np

class Fields(object):
    def __init__(self, p):
        dims = [p.N, p.N]
        self.rho = np.zeros(dims, dtype=np.float64)
        self.rho_u = np.zeros(dims, dtype=np.float64)
        self.rho_v = np.zeros(dims, dtype=np.float64)
        self.tmp = np.zeros(dims, dtype=np.float64)
        self.prs = np.zeros(dims, dtype=np.float64)
        self.egy = np.zeros(dims, dtype=np.float64)
        self.rho_rhs = np.zeros(dims, dtype=np.float64)
        self.rho_u_rhs = np.zeros(dims, dtype=np.float64)
        self.rho_v_rhs = np.zeros(dims, dtype=np.float64)
        self.egy_rhs = np.zeros(dims, dtype=np.float64)
        self.stream = np.zeros(dims, dtype=np.float64)
        self.vort = np.zeros(dims, dtype=np.float64)
