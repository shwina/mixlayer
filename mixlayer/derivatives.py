import numpy as np
from numba import jit, float64, prange

@jit(nopython=True, nogil=True, parallel=True)
def dfdx(f, dx):
    out = np.empty_like(f, dtype=np.float64)
    ny, nx = f.shape
    for i in prange(ny):
        for j in prange(4):
            out[i,j] = (1./dx)*((4./5)*(f[i,j+1] - f[i,j-1]) + 
                    (-1./5)*(f[i,j+2] - f[i,j-2]) +
                    (4./105)*(f[i,j+3] - f[i,j-3]) +
                    (-1./280)*(f[i,j+4] - f[i,j-4]))

        for j in range(4, nx-4):
            out[i,j] = (1./dx)*((4./5)*(f[i,j+1] - f[i,j-1]) + 
                    (-1./5)*(f[i,j+2] - f[i,j-2]) +
                    (4./105)*(f[i,j+3] - f[i,j-3]) +
                    (-1./280)*(f[i,j+4] - f[i,j-4]))
            
        for j in range(-1, -5, -1):
            out[i,j] = (1./dx)*((4./5)*(f[i,j+1] - f[i,j-1]) + 
                    (-1./5)*(f[i,j+2] - f[i,j-2]) +
                    (4./105)*(f[i,j+3] - f[i,j-3]) +
                    (-1./280)*(f[i,j+4] - f[i,j-4]))
    return out

@jit(nopython=True, nogil=True, parallel=True)
def dfdy(f, dy):
    ny, nx = f.shape
    out = np.empty_like(f, dtype=np.float64)
    for i in prange(4, ny-4):
        for j in prange(nx):
            out[i,j] =  (1./dy)*((4./5)*(f[i+1,j] - f[i-1,j]) +
                    (-1./5)*(f[i+2,j] - f[i-2,j]) +
                    (4./105)*(f[i+3,j] - f[i-3,j]) +
                    (-1./280)*(f[i+4,j] - f[i-4,j]))
            out[0,j] = (-11*f[0,j]+18*f[1,j]-9*f[2,j]+2*f[3,j])/(6.*dy)
            out[1,j] = (-2*f[0,j]-3*f[1,j]+6*f[2,j]-1*f[3,j])/(6.*dy)
            out[2,j] = 2*(f[3,j]-f[1,j])/(3*dy) - 1*(f[4,j]-f[0,j])/(12*dy)
            out[3,j] = 3*(f[4,j]-f[2,j])/(4*dy) - 3*(f[5,j]-f[1,j])/(20*dy) + 1*(f[6,j]-f[0,j])/(60*dy) 

            out[-1,j] = -((-11*f[-1,j]+18*f[-2,j]-9*f[-3,j]+2*f[-4,j])/(6.*dy))
            out[-2,j] = -((-2*f[-1,j]-3*f[-2,j]+6*f[-3,j]-1*f[-4,j])/(6.*dy))
            out[-3,j] = -(2*(f[-4,j]-f[-2,j])/(3*dy) - 1*(f[-5,j]-f[-1,j])/(12*dy))
            out[-4,j] = -(3*(f[-5,j]-f[-3,j])/(4*dy) - 3*(f[-6,j]-f[-2,j])/(20*dy) + 1*(f[-7,j]-f[-1,j])/(60*dy))
    return out
