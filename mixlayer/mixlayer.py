import sys

from attrdict import AttrDict
import yaml

import numpy as np
import h5py
from numba import jit, float64, prange

from .derivatives import BoundaryType
from .grid import SinhGrid
from .timestepping import RK4
from .filtering import filter5
from .models.eos import IdealGasEOS

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

def add_forcing(p, f, g):

    fx = np.zeros_like(g.x, dtype=np.float64)
    fy = np.zeros_like(g.x, dtype=np.float64)
    fx_max = 0

    vort = np.zeros_like(g.x, dtype=np.float64)
    stream = np.zeros([p.N+2, p.N+2], dtype=np.float64)

    amplitudes = [1, 0.5, 0.35, 0.35]
    for i in range(4):
        fx += amplitudes[i]*np.abs(np.sin(np.pi*g.x/(2**i*p.disturbance_wavelength)))
        fx_max = np.max([np.max(fx), fx_max])
    
    fx = fx/fx_max
    fy = np.exp(-np.pi*g.y**2/p.vorticity_thickness**2)
    
    vort[...] = fx*fy
    circ = np.sum(g.dy*g.dx*vort)

    vort[...] = (vort*p.Famp_2d*p.disturbance_wavelength*p.U_ref) / (circ/p.nperiod)

    for i in range(50000):
        err = jacobi_step(stream, p.dx, p.dn, -vort, g.dndy, g.d2ndy2)
        if err <= 1e-5:
            break

    u_pert = (stream[2:,1:-1] - stream[0:-2,1:-1])*g.dndy/(2*g.dn)
    v_pert = -(stream[1:-1,2:] - stream[1:-1,0:-2])/(2*g.dx)
       
    vort[...] = ((stream[1:-1, 2:] - 2*stream[1:-1, 1:-1] + stream[1:-1, 0:-2])/(g.dx**2) +
            ((stream[2:, 1:-1] - stream[0:-2, 1:-1])*g.d2ndy2/(2*g.dn)) +
            ((stream[2:, 1:-1] - 2*stream[1:-1, 1:-1] + stream[0:-2, 1:-1])/(g.dn**2))*g.dndy**2)

    circ = np.sum(g.dy*g.dx*vort)

    u_pert = u_pert*p.Famp_2d*p.disturbance_wavelength*p.U_ref / (circ/p.nperiod)
    v_pert = v_pert*p.Famp_2d*p.disturbance_wavelength*p.U_ref / (circ/p.nperiod)

    return u_pert, v_pert

def calculate_timestep(params, fields, eos, grid):

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
    dx = grid.dx
    dy = grid.dy

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

    p = AttrDict()
    f = AttrDict()

    with open(paramfile) as f:
        p.update(yaml.load(f))

    p.Ly = p.Lx*((p.N-1)/p.N)*2.

    # reference temperature
    p.T_ref = max([p.T_inf2, 344.6])

    # eos parameters
    p.Rspecific = 287

    # reference density
    p.rho_ref1 = p.P_inf/(p.Rspecific*p.T_inf1)
    p.rho_ref2 = p.P_inf/(p.Rspecific*p.T_inf2)
    p.rho_ref = (p.rho_ref1+p.rho_ref2)/2.0

    # reference velocities 
    p.C_sound1 = np.sqrt((p.Cp/p.Cv)*(p.Rspecific)*p.T_inf1)
    p.C_sound2 = np.sqrt((p.Cp/p.Cv)*(p.Rspecific)*p.T_inf2)
    p.U_inf1 = 2*p.Ma*p.C_sound1/(1+np.sqrt(p.rho_ref1/p.rho_ref2)*(p.C_sound1/p.C_sound2))
    p.U_inf2 = -np.sqrt(p.rho_ref1/p.rho_ref2)*p.U_inf1
    p.U_ref = p.U_inf1-p.U_inf2

    # grid parameters
    p.dx = p.Lx/p.N
    p.dn = 1./(p.N-1)

    # geometric parameters
    p.disturbance_wavelength = p.Lx/p.nperiod
    p.vorticity_thickness = p.disturbance_wavelength/7.29

    # reference viscosity; thermal and molecular diffusivities
    p.rho_ref = (p.rho_ref1+p.rho_ref2)/2.0
    p.mu_ref = (p.rho_ref*(p.U_inf1-p.U_inf2)*p.vorticity_thickness)/p.Re
    p.kappa_ref = 0.5*(p.Cp+p.Cp)*p.mu_ref/p.Pr 
    p.gamma_ref = p.mu_ref/(p.rho_ref*p.Pr)

    # fields
    dims = [p.N, p.N]
    f.rho = np.zeros(dims, dtype=np.float64)
    f.rho_u = np.zeros(dims, dtype=np.float64)
    f.rho_v = np.zeros(dims, dtype=np.float64)
    f.tmp = np.zeros(dims, dtype=np.float64)
    f.prs = np.zeros(dims, dtype=np.float64)
    f.egy = np.zeros(dims, dtype=np.float64)
    f.rho_rhs = np.zeros(dims, dtype=np.float64)
    f.rho_u_rhs = np.zeros(dims, dtype=np.float64)
    f.rho_v_rhs = np.zeros(dims, dtype=np.float64)
    f.egy_rhs = np.zeros(dims, dtype=np.float64)
    f.stream = np.zeros(dims, dtype=np.float64)
    f.vort = np.zeros(dims, dtype=np.float64)

    # make grid
    g = SinhGrid(p.N, p.N, p.Lx, p.Ly, p.grid_beta, BoundaryType.PERIODIC, BoundaryType.INNER)

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
        
        dt = calculate_timestep(p, f, eos, g)

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
