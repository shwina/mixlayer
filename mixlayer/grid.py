import numpy as np

def asinh_grid(nx, ny, Lx, Ly, beta):
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
    y = y-Ly/2
    return x, y, dndy, d2ndy2
