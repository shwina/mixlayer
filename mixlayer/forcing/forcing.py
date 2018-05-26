import numpy as np
from mixlayer.poisson import PoissonSolver

def forcing_2d(grid, ops, disturbance_wavelength, vorticity_thickness, Famp_2d,
        U_ref, nperiod):

    x, y, dx, dy = grid.x, grid.y, grid.dx, grid.dy
    N = grid.nx
    dfdx, dfdy = ops.dfdx, ops.dfdy
        
    fx = np.zeros_like(x, dtype=np.float64)
    fy = np.zeros_like(x, dtype=np.float64)
    fx_max = 0

    vort = np.zeros_like(grid.x, dtype=np.float64)
    stream = np.copy(vort)

    amplitudes = [1, 0.5, 0.35, 0.35]
    for i in range(4):
        fx += amplitudes[i]*np.abs(np.sin(np.pi*x/(2**i*disturbance_wavelength)))
        fx_max = np.max([np.max(fx), fx_max])
    
    fx = fx/fx_max
    fy = np.exp(-np.pi*y**2/vorticity_thickness**2)
    
    vort[...] = fx*fy
    circ = np.sum(dy*dx*vort)

    vort[...] = (vort*Famp_2d*disturbance_wavelength*U_ref) / (circ/nperiod)

    ps = PoissonSolver(N, dx, dy)
    ps.solve(-vort, stream)

    u_pert =  (np.roll(stream, -1, 0) - np.roll(stream, 1, 0))/(2*dy)
    v_pert = -(np.roll(stream, -1, 1) - np.roll(stream, 1, 1))/(2*dx)

    vort[...] = ((np.roll(stream, -1, 1) - 2*stream + np.roll(stream, 1, 1))/(dx**2) +
                 (np.roll(stream, -1, 0) - 2*stream + np.roll(stream, 1, 0))/(dy**2))

    circ = np.sum(dy*dx*vort)

    u_pert = u_pert*Famp_2d*disturbance_wavelength*U_ref / (circ/nperiod)
    v_pert = v_pert*Famp_2d*disturbance_wavelength*U_ref / (circ/nperiod)

    return u_pert, v_pert

