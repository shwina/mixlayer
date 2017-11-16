import numpy as np
import h5py
import sys
from numba import jit, float64, prange

from .derivatives import BoundaryType
from .params import Params
from .fields import Fields
from .grid import SinhGrid
from .timestepping import RK4
from .filtering import filter5
from .eos import IdealGasEOS

@jit(nopython=True, nogil=True)
def jacobi_step(f, dx, dn, rhs, dndy, d2ndy2):
    f_old = f.copy()
    denominator = 2./dx**2 + (2./dn**2)*dndy**2

    ny, nx = f.shape

    f[0, :] = 0
    f[-1, :] = 0
    f[:, 0] = f[:, -2]
    f[:, -1] = f[:, 1]

    for i in range(1, nx-1):
        for j in range(1, nx-1):
            fnew = (-rhs[i-1, j-1] +
                    (f[i,j+1] + f[i,j-1])/dx**2 +
                    (f[i+1,j] - f[i-1,j])/(2*dn)*d2ndy2[i-1,j-1] +
                    (f[i+1,j] + f[i-1,j])/(dn**2)*dndy[i-1,j-1]**2)/denominator[i-1,j-1]
            f[i,j] = f[i,j] + 1.6*(fnew - f[i,j])
    return np.linalg.norm(f-f_old)

def add_forcing(params, fields, grid):
    N = params.N
    dn = params.dn
    stream = fields.stream
    vort = fields.vort

    dn = grid.dn
    dndy = grid.dndy
    d2ndy2 = grid.d2ndy2
    x = grid.x
    y = grid.y

    U_ref = params.U_ref
    Famp_2d = params.Famp_2d
    vorticity_thickness = params.vorticity_thickness
    disturbance_wavelength = params.disturbance_wavelength
    nperiod = params.nperiod

    dx = 1./N
    dy = x.copy()
    dy[:-1, :] = y[1:, :] - y[:-1, :]
    dy[-1, :] = y[-1, :] - y[-2, :]

    fx = np.zeros_like(x, dtype=np.float64)
    fy = np.zeros_like(x, dtype=np.float64)
    fx_max = 0

    vort = np.zeros_like(x, dtype=np.float64)
    stream = np.zeros([N+2, N+2], dtype=np.float64)

    amplitudes = [1, 0.5, 0.35, 0.35]
    for i in range(4):
        fx += amplitudes[i]*np.abs(np.sin(np.pi*x/(2**i*disturbance_wavelength)))
        fx_max = np.max([np.max(fx), fx_max])
    
    fx = fx/fx_max
    fy = np.exp(-np.pi*y**2/vorticity_thickness**2)
    
    vort[...] = fx*fy
    circ = np.sum(dy*dx*vort)

    vort[...] = (vort*Famp_2d*disturbance_wavelength*U_ref) / (circ/nperiod)

    for i in range(50000):
        err = jacobi_step(stream, dx, dn, -vort, dndy, d2ndy2)
        if err <= 1e-5:
            break

    u_pert = (stream[2:,1:-1] - stream[0:-2,1:-1])*dndy/(2*dn)
    v_pert = -(stream[1:-1,2:] - stream[1:-1,0:-2])/(2*dx)
       
    vort[...] = ((stream[1:-1, 2:] - 2*stream[1:-1, 1:-1] + stream[1:-1, 0:-2])/(dx**2) +
            ((stream[2:, 1:-1] - stream[0:-2, 1:-1])*d2ndy2/(2*dn)) +
            ((stream[2:, 1:-1] - 2*stream[1:-1, 1:-1] + stream[0:-2, 1:-1])/(dn**2))*dndy**2)

    circ = np.sum(dy*dx*vort)

    u_pert = u_pert*Famp_2d*disturbance_wavelength*U_ref / (circ/nperiod)
    v_pert = v_pert*Famp_2d*disturbance_wavelength*U_ref / (circ/nperiod)

    return u_pert, v_pert

def calculate_timestep(params, fields, eos, x, y):

    fields.tmp[...] = (
            fields.egy -
            0.5*(fields.rho_u**2 + fields.rho_v**2)/fields.rho
        ) / (fields.rho*eos.Cv)

    eos.pressure(fields.tmp, fields.rho, fields.prs)

    rho = fields.rho
    rho_u = fields.rho_u
    rho_v = fields.rho_v
    egy = fields.egy
    tmp = fields.tmp
    prs = fields.prs

    gamma_ref = params.gamma_ref
    mu_ref = params.mu_ref
    kappa_ref = params.kappa_ref
    Cp = params.Cp
    Cv = params.Cv
    Rspecific = params.Rspecific
    cfl_vel = params.cfl_vel
    cfl_visc = params.cfl_visc

    # dx, dy
    N = np.shape(x)[0]
    dx = 1./N
    dy = x.copy()
    dy[:-1, :] = y[1:, :] - y[:-1, :]
    dy[-1, :] = y[-1, :] - y[-2, :]

    dxmin = np.minimum(dx, dy)

    # calculate diffusivities:
    alpha_1 = gamma_ref
    alpha_2 = mu_ref/rho
    alpha_3 = kappa_ref/(Cp*rho)
    alpha_max = np.maximum(np.maximum(alpha_1, alpha_2), alpha_3)

    # calculate C_sound
    C_sound = np.sqrt(Cp/Cv*Rspecific*tmp)
    test_1 = cfl_vel*dx/(C_sound + abs(rho_u/rho))
    test_2 = cfl_vel*dy/(C_sound + abs(rho_v/rho))
    test_3 = cfl_visc*(dxmin**2)/alpha_max

    dt = np.min(np.minimum(np.minimum(test_1, test_2), test_3))
    return dt

def rhs_euler_terms(params, fields, grid):
 
    rho = fields.rho
    rho_u = fields.rho_u
    rho_v = fields.rho_v
    egy = fields.egy
    tmp = fields.tmp
    prs = fields.prs
    rho_rhs = fields.rho_rhs
    rho_u_rhs = fields.rho_u_rhs
    rho_v_rhs = fields.rho_v_rhs
    egy_rhs = fields.egy_rhs

    dx = params.dx
    dn = params.dn

    rho_rhs[...] = -grid.dfdy(rho_v)
    rho_rhs[[0,-1], :] = 0
    rho_rhs[...] += -grid.dfdx(rho_u)

    rho_u_rhs[...] = -grid.dfdy(rho_u*rho_v/rho)
    rho_u_rhs[[0,-1], :] = 0
    rho_u_rhs += -grid.dfdx(rho_u*rho_u/rho + prs) 
    
    rho_v_rhs[...] = -grid.dfdy(rho_v*rho_v/rho + prs)
    rho_v_rhs[[0,-1], :] = 0 
    rho_v_rhs += -grid.dfdx(rho_v*rho_u/rho )
    
    egy_rhs_x = -grid.dfdx((egy + prs)*(rho_u/rho))
    egy_rhs_y = -grid.dfdy((egy + prs)*(rho_v/rho))
    
    egy_rhs_y[[0,-1], :] = 0
    egy_rhs[...] = egy_rhs_x + egy_rhs_y

def rhs_viscous_terms(params, fields, grid):

    rho = fields.rho
    rho_u = fields.rho_u
    rho_v = fields.rho_v
    egy = fields.egy
    tmp = fields.tmp
    prs = fields.prs
    rho_rhs = fields.rho_rhs
    rho_u_rhs = fields.rho_u_rhs
    rho_v_rhs = fields.rho_v_rhs
    egy_rhs = fields.egy_rhs
    
    dx = params.dx
    dn = params.dn
    mu = params.mu_ref
    kappa = params.kappa_ref

    div_vel = grid.dfdx(rho_u/rho) + grid.dfdy(rho_v/rho)
    tau_11 = -(2./3)*mu*div_vel + 2*mu*grid.dfdx(rho_u/rho) 
    tau_12 = mu*(grid.dfdx(rho_v/rho) + grid.dfdy(rho_u/rho))
    tau_22 = -(2./3)*mu*div_vel + 2*mu*grid.dfdy(rho_v/rho)
    
    tau_12[0, :] = (18.*tau_12[1, :] - 9*tau_12[2, :] + 2*tau_12[3, :])/11.
    tau_12[-1, :] = (18.*tau_12[-2, :] - 9*tau_12[-3, :] + 2*tau_12[-4, :])/11.

    rho_u_rhs += (grid.dfdx(tau_11) + grid.dfdy(tau_12))
    rho_v_rhs += (grid.dfdx(tau_12) + grid.dfdy(tau_22))
    egy_rhs += (grid.dfdx(rho_u/rho*tau_11) + grid.dfdx(rho_v/rho*tau_12) + grid.dfdy(rho_u/rho*tau_12) + grid.dfdy(rho_v/rho*tau_22)) + kappa*(grid.dfdx(grid.dfdx(tmp)) + grid.dfdy(grid.dfdy(tmp)))

def apply_filter(params, fields):
    for f in fields.rho, fields.rho_u, fields.rho_v, fields.egy:
        filter5(f)

def non_reflecting_boundary_conditions(params, fields, grid):
    
    rho = fields.rho
    rho_u = fields.rho_u
    rho_v = fields.rho_v
    egy = fields.egy
    tmp = fields.tmp
    prs = fields.prs
    rho_rhs = fields.rho_rhs
    rho_u_rhs = fields.rho_u_rhs
    rho_v_rhs = fields.rho_v_rhs
    egy_rhs = fields.egy_rhs

    dx = params.dx
    dn = params.dn
    filter_amplitude = params.filter_amplitude
    Ma = params.Ma
    Ly = params.Ly
    P_inf = params.P_inf

    C_sound = np.sqrt(params.Cp/params.Cv*params.Rspecific*tmp)

    dpdy = grid.dfdy(prs)
    drhody = grid.dfdy(rho)
    dudy = grid.dfdy(rho_u/rho)
    dvdy = grid.dfdy(rho_v/rho)
    
    L_1 = (rho_v/rho - C_sound) * (dpdy - rho*C_sound*dvdy)
    L_2 = rho_v/rho* (C_sound**2 * drhody - dpdy)
    L_3 = rho_v/rho* (dudy)
    L_4 = 0.4 * (1 - Ma**2)*C_sound/Ly*(prs - P_inf)

    d_1 = (1./C_sound**2)*(L_2 + 0.5*(L_4 + L_1))
    d_2 = 0.5*(L_1 + L_4)
    d_3 = L_3
    d_4 = 1./(2*rho*C_sound) * (L_4 - L_1)

    rho_rhs[0, :] = (rho_rhs - d_1)[0, :]
    rho_u_rhs[0, :] = (rho_u_rhs - rho_u/rho*d_1 - rho*d_3)[0, :]
    rho_v_rhs[0, :] = (rho_v_rhs - rho_v/rho*d_1 - rho*d_4)[0, :]
    egy_rhs[0, :] = (egy_rhs-
        0.5*np.sqrt((rho_u/rho)**2 + (rho_v/rho)**2)*d_1 -
        d_2*(prs + egy)/(rho*C_sound**2) -
        rho*(rho_v/rho*d_4+rho_u/rho*d_3))[0, :]

    L_1 = 0.4 * (1 - Ma**2)*C_sound/Ly*(prs - P_inf)
    L_2 = rho_v/rho* (C_sound**2 * drhody - dpdy)
    L_3 = rho_v/rho* (dudy)
    L_4 = (rho_v/rho + C_sound) * (dpdy + rho*C_sound*dvdy)

    d_1 = (1./C_sound**2)*(L_2 + 0.5*(L_4 + L_1))
    d_2 = 0.5*(L_1 + L_4)
    d_3 = L_3
    d_4 = 1./(2*rho*C_sound) * (L_4 - L_1)

    rho_rhs[-1, :] = (rho_rhs - d_1)[-1, :]
    rho_u_rhs[-1, :] = (rho_u_rhs - rho_u/rho*d_1 - rho*d_3)[-1, :]
    rho_v_rhs[-1, :] = (rho_v_rhs - rho_v/rho*d_1 - rho*d_4)[-1, :]
    egy_rhs[-1, :] = (egy_rhs-
        0.5*np.sqrt((rho_u/rho)**2 + (rho_v/rho)**2)*d_1 -
        d_2*(prs + egy)/(rho*C_sound**2) -
        rho*(rho_v/rho*d_4+rho_u/rho*d_3))[-1, :]

def rhs(eqvars, params, fields, eos, grid):

    fields.tmp[...] = (
            fields.egy -
            0.5*(fields.rho_u**2 + fields.rho_v**2)/fields.rho
        ) / (fields.rho*eos.Cv)

    eos.pressure(fields.tmp, fields.rho, fields.prs)

    rho = fields.rho
    rho_u = fields.rho_u
    rho_v = fields.rho_v
    egy = fields.egy
    tmp = fields.tmp
    prs = fields.prs
    rho_rhs = fields.rho_rhs
    rho_u_rhs = fields.rho_u_rhs
    rho_v_rhs = fields.rho_v_rhs
    egy_rhs = fields.egy_rhs

    rhs_euler_terms(params, fields, grid)

    rhs_viscous_terms(params, fields, grid)

    non_reflecting_boundary_conditions(params, fields, grid)

    return fields.rho_rhs, fields.rho_u_rhs, fields.rho_v_rhs, fields.egy_rhs

def main():
    paramfile = sys.argv[1]

    p = Params(paramfile)
    f = Fields(p)
    g = SinhGrid(p.N, p.N, p.Lx, p.Ly, p.grid_beta, BoundaryType.PERIODIC, BoundaryType.INNER)
    # make grid

    # initialize fields
    weight = np.tanh(np.sqrt(np.pi)*g.y/p.vorticity_thickness)

    f.tmp[:, :] = p.T_inf2 + (weight+1)/2.*(p.T_inf1-p.T_inf2)
    f.rho[:, :] = p.P_inf/(p.Rspecific*f.tmp[:, :])
    
    f.rho_u[:, :] = f.rho*(p.U_inf2+(weight+1)/2.*(p.U_inf1-p.U_inf2))
    f.rho_v[:, :] = 0.0

    u_pert, v_pert = add_forcing(p, f, g)

    f.rho_u += f.rho*u_pert
    f.rho_v += f.rho*v_pert

    f.egy[:, :] = 0.5*(f.rho_u**2 + f.rho_v**2)/f.rho + f.rho*p.Cv*f.tmp

    eos = IdealGasEOS()

    # make time stepper
    stepper = RK4([f.rho, f.rho_u, f.rho_v, f.egy],
            rhs, p, f, eos, g)
    
    # run simulation
    import timeit

    for i in range(p.timesteps):
        
        dt = calculate_timestep(p, f, eos, g.x, g.y)

        print("Iteration: {:10d}    Time: {:15.10e}    Total energy: {:15.10e}".format(i, dt*i, np.sum(f.egy)))

        stepper.step(dt)
        apply_filter(p, f)
    
        if p.writer:
            if i%200 == 0:
                outfile = h5py.File("{:05d}.hdf5".format(i))
                outfile.create_group("fields")
                outfile.create_dataset("fields/rho", data=f.rho)
                outfile.create_dataset("fields/rho_u", data=f.rho_u)
                outfile.create_dataset("fields/rho_v", data=f.rho_v)
                outfile.create_dataset("fields/tmp", data=f.tmp)

if __name__ == "__main__":
    main()
