import numpy as np
from .derivatives import dfdx, dfdy

class AsinhGrid(object):

    def __init__(self, nx, ny, Lx, Ly, beta, bx, by):
        """

        """
        #FIXME: This needs to be Lx-1 for a general grid..
        dx = Lx/nx
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
        self.dndy = dndy
        self.d2ndy2 = d2ndy2
        self.bx = bx
        self.by = by

    def dfdx(self, f):
        return dfdx(f, self.dx, boundary_type=self.bx)
        
    def dfdy(self, f):
        return dfdy(f, self.dn, boundary_type=self.by)*self.dndy

