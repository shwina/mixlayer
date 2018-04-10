import numpy as np
from numba import jit, float64, prange
from mixlayer.boundaries import BoundaryConditionType

@jit(nopython=True, nogil=True, parallel=True)
def dfdx(f, dx,
        left_bc_type=BoundaryConditionType.PERIODIC,
        left_bc_val=0,
        right_bc_type=BoundaryConditionType.PERIODIC,
        right_bc_val=0):
    """
    Compute the derivative of a 2-d array on a regularly spaced grid
    in the x-direction using an 8-th order finite difference approximation.
    
    Parameters
    ----------

    f : array like
        2-d array representing the function values at the grid points
    dx : float
        Spacing between 2 adjacent points in the x-direction
    bc: BoundaryCondition

    Returns
    -------
    out : array
        2-d array representing the approximation of dfdx at each grid point

    """
    out = np.empty_like(f, dtype=np.float64)
    ny, nx = f.shape
    for i in prange(ny):
        for j in prange(4, nx-4):
            out[i,j] = (1./dx)*(
                    ( 4 / 5  ) * (f[i,j+1] - f[i,j-1]) + 
                    (-1 / 5  ) * (f[i,j+2] - f[i,j-2]) +
                    ( 4 / 105) * (f[i,j+3] - f[i,j-3]) +
                    (-1 / 280) * (f[i,j+4] - f[i,j-4]))
        
        if left_bc_type == BoundaryConditionType.PERIODIC:
            for j in range(4):
                out[i,j] = (1./dx)*(
                        ( 4 / 5  ) * (f[i,j+1] - f[i,j-1]) + 
                        (-1 / 5  ) * (f[i,j+2] - f[i,j-2]) +
                        ( 4 / 105) * (f[i,j+3] - f[i,j-3]) +
                        (-1 / 280) * (f[i,j+4] - f[i,j-4]))
        else:
            out[i,0] = (-11*f[i,0] + 18*f[i,1] - 9*f[i,2] + 2*f[i,3]) / (6.*dx)
            out[i,1] = ( -2*f[i,0] -  3*f[i,1] + 6*f[i,2] - 1*f[i,3]) / (6.*dx)
            out[i,2] = 2*(f[i,3] - f[i,1]) / (3*dx) - (f[i,4] - f[i,0]) / (12*dx)
            out[i,3] = 3*(f[i,4] - f[i,2]) / (4*dx) - 3*(f[i,5] - f[i,1]) / (20*dx) + (f[i,6] - f[i,0]) / (60*dx) 

            if left_bc_type == BoundaryConditionType.NEUMANN:
                out[i, 0] = left_bc_val

        if right_bc_type == BoundaryConditionType.PERIODIC:
            for j in range(-1, -5, -1):
                out[i,j] = (1./dx)*(
                        ( 4 / 5  ) * (f[i,j+1] - f[i,j-1]) + 
                        (-1 / 5  ) * (f[i,j+2] - f[i,j-2]) +
                        ( 4 / 105) * (f[i,j+3] - f[i,j-3]) +
                        (-1 / 280) * (f[i,j+4] - f[i,j-4]))

        else:
            out[i,-1] = -(-11*f[i,-1] + 18*f[i,-2] - 9*f[i,-3] + 2*f[i,-4]) / (6.*dx)
            out[i,-2] = -( -2*f[i,-1] -  3*f[i,-2] + 6*f[i,-3] - 1*f[i,-4]) / (6.*dx)
            out[i,-3] = -(2*(f[i,-4] - f[i,-2]) / (3*dx) - (f[i,-5] - f[i,-1]) / (12*dx))
            out[i,-4] = -(3*(f[i,-5] - f[i,-3]) / (4*dx) - 3*(f[i,-6] - f[i,-2]) / (20*dx) + (f[i,-7] - f[i,-1]) / (60*dx))

            if right_bc_type == BoundaryConditionType.NEUMANN:
                out[i, -1] = right_bc_val

    return out

@jit(nopython=True, nogil=True, parallel=True)
def dfdy(f, dy,
        left_bc_type=BoundaryConditionType.PERIODIC,
        left_bc_val=0,
        right_bc_type=BoundaryConditionType.PERIODIC,
        right_bc_val=0):
    """
    Compute the derivative of a 2-d array on a regularly spaced grid
    in the y-direction using an 8-th order finite difference approximation.
    
    See: dfdx
    """
    ny, nx = f.shape
    out = np.empty_like(f, dtype=np.float64)
    for i in prange(4, ny-4):
        for j in prange(nx):
            out[i,j] = (1./dy)*(
                    ( 4 / 5  ) * (f[i+1,j] - f[i-1,j]) + 
                    (-1 / 5  ) * (f[i+2,j] - f[i-2,j]) +
                    ( 4 / 105) * (f[i+3,j] - f[i-3,j]) +
                    (-1 / 280) * (f[i+4,j] - f[i-4,j]))

    if left_bc_type == BoundaryConditionType.PERIODIC:
        for i in range(4):
            for j in prange(nx):
                out[i,j] = (1./dy)*(
                        ( 4 / 5  ) * (f[i+1,j] - f[i-1,j]) + 
                        (-1 / 5  ) * (f[i+2,j] - f[i-2,j]) +
                        ( 4 / 105) * (f[i+3,j] - f[i-3,j]) +
                        (-1 / 280) * (f[i+4,j] - f[i-4,j]))
    
    else:
        for j in prange(nx):
            out[0,j] = (-11*f[0,j] + 18*f[1,j] - 9*f[2,j] + 2*f[3,j]) / (6.*dy)
            out[1,j] = ( -2*f[0,j] -  3*f[1,j] + 6*f[2,j] - 1*f[3,j]) / (6.*dy)
            out[2,j] = 2*(f[3,j] - f[1,j]) / (3*dy) - (f[4,j] - f[0,j]) / (12*dy)
            out[3,j] = 3*(f[4,j] - f[2,j]) / (4*dy) - 3*(f[5,j] - f[1,j]) / (20*dy) + (f[6,j] - f[0,j]) / (60*dy) 


        if left_bc_type == BoundaryConditionType.NEUMANN:
            out[0, :] = left_bc_val

    if right_bc_type == BoundaryConditionType.PERIODIC:
        for i in range(-1, -5, -1):
            for j in prange(nx):
                out[i,j] = (1./dy)*(
                        ( 4 / 5  ) * (f[i+1,j] - f[i-1,j]) + 
                        (-1 / 5  ) * (f[i+2,j] - f[i-2,j]) +
                        ( 4 / 105) * (f[i+3,j] - f[i-3,j]) +
                        (-1 / 280) * (f[i+4,j] - f[i-4,j]))

    else:
        for j in prange(nx):
            out[-1,j] = -(-11*f[-1,j] + 18*f[-2,j] - 9*f[-3,j] + 2*f[-4,j]) / (6.*dy)
            out[-2,j] = -( -2*f[-1,j] -  3*f[-2,j] + 6*f[-3,j] - 1*f[-4,j]) / (6.*dy)
            out[-3,j] = -(2*(f[-4,j] - f[-2,j]) / (3*dy) - (f[-5,j] - f[-1,j]) / (12*dy))
            out[-4,j] = -(3*(f[-5,j] - f[-3,j]) / (4*dy) - 3*(f[-6,j] - f[-2,j]) / (20*dy) + (f[-7,j] - f[-1,j]) / (60*dy))

        if right_bc_type == BoundaryConditionType.NEUMANN:
            out[-1, :] = right_bc_val
    
    return out
