import numpy as np
from scipy.sparse import coo_matrix
import scipy.linalg as linalg
import scipy.sparse.linalg as splinalg

class PoissonSolver(object):
    """
    Solve the Poisson equation

        d^2/dx^2(u) + d^2/dy^(u) = rhs

    on a [N, N] grid, with periodic
    boundary conditions.
    """

    def __init__(self, N, dx, dy):
        self.N = N
        self.dx = dx
        self.dy = dy

    def solve(self, rhs, solution):
        N = self.N
        dx = self.dx
        dy = self.dy
        A = self.make_poisson_matrix()
        A = self.offset_elements(A, 1e-9/(dx*dx + dy*dy))
        A = A.tocsr()
        print("Solving....")
        solution[...] = splinalg.bicg(A, rhs.ravel())[0]

    def make_poisson_matrix(self):
        N = self.N
        dx = self.dx
        dy = self.dy
        coo_list = []
        for i in range(N):
            for j in range(N):
                
                dx_ = dx if np.isscalar(dx) else dx[i,j]
                dy_ = dy if np.isscalar(dy) else dy[i,j]

                left = right = bottom = top = diag = 0.0
                ix = i*N + j
                if j > 0:
                    left = 1./(dx_**2)
                    diag -= 1./(dx_**2)
                    coo_list.append([ix, ix-1, left])
                if j < (N-1):
                    right = 1./(dx_**2)
                    diag -= 1./(dx_**2)
                    coo_list.append([ix, ix+1, right])
                if i > 0:
                    bottom = 1./(dy_**2)
                    diag -= 1./(dy_**2)
                    coo_list.append([ix, ix-N, bottom])
                if i < (N-1):
                    top = 1./(dy_**2)
                    diag -= 1./(dy_**2)
                    coo_list.append([ix, ix+N, top])
                
                coo_list.append([ix, ix, diag])

        coo_list = np.array(coo_list)
        i        = coo_list[:, 0]
        j        = coo_list[:, 1]
        data     = coo_list[:, 2]

        coo = coo_matrix((data, (i, j)))
        return coo

    def offset_elements(self, A, alpha=1e-5):
        out = A.copy()
        alpha = alpha if np.isscalar(alpha) else alpha.ravel()
        out.setdiag(out.diagonal() + alpha)
        return out
