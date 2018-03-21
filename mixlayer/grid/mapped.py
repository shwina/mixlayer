import numpy as np
from ..derivatives import dfdx, dfdy

class SinhGrid(object):

    def __init__(self, size, dims, boundaries, beta):
        """
        Creates a grid "stretched" in the y-direction with stretching function 'sinh(y)'.

        Parameters
        ----------
        size : int or tuple of int
            tuple (Nx, Ny) specifying the number of grid points in each co-ordinate direction.
        dims : float or tuple of float
            tuple (Lx, Ly) specifying the length of the grid in each co-ordinate direction.
        boundaries: tuple of mixlayer.derivatives.BoundaryType
            Tuple (bx, by) specifying the way boundaries are treated in each co-ordinate direction.
                BoundaryType.INNER : inner stencils used on the boundary)
                BoundaryType.PERIODIC : domain wraps around itself in this direction).
        beta : stretching parameter
            Extent of stretching (very small beta implies no stretching, more beta implies more stretching).
        """
        if np.isscalar(size):
            nx = ny = size
        else:
            nx, ny = size
        if np.isscalar(dims):
            Lx = Ly = dims
        else:
            Lx, Ly = dims
        bx, by = boundaries

        dx = Lx/(nx-1)
        dn = 1./(ny-1)
        x = np.arange(nx)*dx*np.ones([ny, nx])
        y = np.arange(0, 1+dn, dn)*np.ones([ny, nx])
        y = y.T
        grid_A = 1./(2*beta)*np.log((1 + (np.exp(beta) - 1)*((Ly/2)/Ly))/(
                1 + (np.exp(-beta) - 1)*((Ly/2)/Ly)))
        y = (Ly/2)*(1 + np.sinh(beta*(y - grid_A))/np.sinh(
            beta*grid_A))
        dndy = np.sinh(beta*grid_A)/(beta*(Ly/2)*(1+((y/(Ly/2))-1)**2*np.sinh(beta*grid_A)**2)**0.5)
        d2ndy2 = -Ly*(np.sinh(beta*grid_A))**3*((y/(Ly/2))-1)/(beta*
                    (Ly/2)**2*(
                    1 + ((y/(Ly/2))-1)**2*(np.sinh(beta*grid_A))**2)**1.5)/Ly
        dy = np.zeros_like(y)
        dy[:-1, :] = y[1:, :] - y[:-1, :]
        dy[-1, :] = y[-1, :] - y[-2, :]
        y = y-Ly/2

        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.dn = dn
        self.Lx = Lx
        self.Ly = Ly
        self.dndy = dndy
        self.d2ndy2 = d2ndy2
        self.bx = bx
        self.by = by
        self.shape = [ny, nx]

    def dfdx(self, f):
        return dfdx(f, self.dx, bc_type=self.bx)
        
    def dfdy(self, f):
        return dfdy(f, self.dn, bc_type=self.by)*self.dndy
