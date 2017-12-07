import numpy as np
from scipy.sparse import coo_matrix
import scipy.linalg as linalg
import scipy.sparse.linalg as splinalg

class PoissonSolver(object):
    """
    Solve the Poisson equation

        d^2/dx^2(u) + d^2/dy^(u) = rhs

    on a [N, N] grid, with periodic
    boundary conditions in x, and
    zero boundary conditions in y.
    """

    def __init__(self, N, dx, dy):
        self.N = N
        self.dx = dx
        self.dy = dy
        self.make_poisson_matrix()
        self.A = self.offset_elements(self.A, 1e-9/(dx*dx + dy*dy))
        self.A = self.A.tocsr()

    def solve(self, rhs, solution):
        solution[[0, -1], :] = 0
        N = self.N
        dx = self.dx
        dy = self.dy
        solution[...] = splinalg.bicg(self.A, rhs.ravel())[0].reshape([N, N])

    def make_poisson_matrix(self):
        N = self.N
        dx = self.dx
        dy = self.dy
        coo_list = []
        for i in range(N):
            for j in range(N):

                ix = i*N + j
                
                dx_ = dx if np.isscalar(dx) else dx[i,j]
                dy_ = dy if np.isscalar(dy) else dy[i,j]

                left = right = bottom = top = diag = 0.0

                if i == 0 or i == N-1: 
                    diag = -1./(dx_**2)
                    coo_list.append([ix, ix, diag])
                    continue

                if j > 0:
                    left = 1./(dx_**2)
                    diag -= 1./(dx_**2)
                    coo_list.append([ix, ix-1, left])
                else:
                    left = 1./(dx_**2)
                    diag -= 1./(dx_**2)
                    coo_list.append([ix, i*N+(N-1), left])
                if j < (N-1):
                    right = 1./(dx_**2)
                    diag -= 1./(dx_**2)
                    coo_list.append([ix, ix+1, right])
                else:
                    right = 1./(dx_**2)
                    diag -= 1./(dx_**2)
                    coo_list.append([ix, i*N, right])

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
        self.A = coo
        return coo

    def offset_elements(self, A, alpha=1e-5):
        out = A.copy()
        alpha = alpha if np.isscalar(alpha) else alpha.ravel()
        out.setdiag(out.diagonal() + alpha)
        return out
