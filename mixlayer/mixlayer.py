import sys

from attrdict import AttrDict
import yaml

import numpy as np
import h5py
from numba import jit, float64, prange

from .derivatives import BoundaryType
from .grid.mapped import SinhGrid
from .timestepping import RK4
from .filtering import filter5
from .models.eos import IdealGasEOS
from .poisson import PoissonSolver
from .equation import Equation

def add_forcing(p, f, g):

    fx = np.zeros_like(g.x, dtype=np.float64)
    fy = np.zeros_like(g.x, dtype=np.float64)
    fx_max = 0

    vort = np.zeros_like(g.x, dtype=np.float64)
    stream = np.zeros([p.N, p.N], dtype=np.float64)

    amplitudes = [1, 0.5, 0.35, 0.35]
    for i in range(4):
        fx += amplitudes[i]*np.abs(np.sin(np.pi*g.x/(2**i*p.disturbance_wavelength)))
        fx_max = np.max([np.max(fx), fx_max])
    
    fx = fx/fx_max
    fy = np.exp(-np.pi*g.y**2/p.vorticity_thickness**2)
    
    vort[...] = fx*fy
    circ = np.sum(g.dy*g.dx*vort)

    vort[...] = (vort*p.Famp_2d*p.disturbance_wavelength*p.U_ref) / (circ/p.nperiod)

    ps = PoissonSolver(p.N, g.dx, g.dy)
    ps.solve(-vort, stream)

    u_pert = (np.roll(stream, -1, 1) - np.roll(stream, 1, 1))/(2*g.dy)
    v_pert = -(np.roll(stream, -1, 0) - np.roll(stream, 1, 0))/(2*g.dx)

    vort[...] = ((np.roll(stream, -1, 1) - 2*stream + np.roll(stream, 1, 1))/(g.dx**2) +
                 (np.roll(stream, -1, 0) - 2*stream + np.roll(stream, 1, 0))/(g.dy**2))

    circ = np.sum(g.dy*g.dx*vort)

    u_pert = u_pert*p.Famp_2d*p.disturbance_wavelength*p.U_ref / (circ/p.nperiod)
    v_pert = v_pert*p.Famp_2d*p.disturbance_wavelength*p.U_ref / (circ/p.nperiod)

    return u_pert, v_pert

def calculate_timestep(p, f, g, eos):

    f.tmp[...] = (
            f.egy -
            0.5*(f.rho_u**2 + f.rho_v**2)/f.rho
        ) / (f.rho*eos.Cv)

    eos.pressure(f.tmp, f.rho, f.prs)

    dxmin = np.minimum(g.dx, g.dy)

    # calculate diffusivities:
    alpha_1 = p.gamma_ref
    alpha_2 = p.mu_ref/f.rho
    alpha_3 = p.kappa_ref/(eos.Cp*f.rho)
    alpha_max = np.maximum(np.maximum(alpha_1, alpha_2), alpha_3)

    # calculate C_sound
    C_sound = np.sqrt(eos.Cp/eos.Cv*eos.R*f.tmp)
    test_1 = p.cfl_vel*g.dx/(C_sound + abs(f.rho_u/f.rho))
    test_2 = p.cfl_vel*g.dy/(C_sound + abs(f.rho_v/f.rho))
    test_3 = p.cfl_visc*(dxmin**2)/alpha_max

    dt = np.min(np.minimum(np.minimum(test_1, test_2), test_3))
    return dt

def rho_rhs(p, f, g, out=None):
    out[...] = -g.dfdy(f.rho_v)
    out[[0,-1], :] = 0
    out[...] += -g.dfdx(f.rho_u)

def rho_u_rhs(p, f, g, out=None):
    out[...] = -g.dfdy(f.rho_u * f.rho_v/f.rho)
    out[[0,-1], :] = 0
    out += -g.dfdx(f.rho_u*f.rho_u/f.rho + f.prs) 

    div_vel = g.dfdx(f.rho_u / f.rho) + g.dfdy(f.rho_v / f.rho)

    tau_11 = -(2./3)*p.mu_ref*div_vel + 2*p.mu_ref*g.dfdx(f.rho_u / f.rho) 
    tau_22 = -(2./3)*p.mu_ref*div_vel + 2*p.mu_ref*g.dfdy(f.rho_v / f.rho)
    tau_12 = p.mu_ref*(g.dfdx(f.rho_v / f.rho) + g.dfdy(f.rho_u / f.rho))
    
    tau_12[0, :] = (18.*tau_12[1, :] - 9*tau_12[2, :] + 2*tau_12[3, :]) / 11
    tau_12[-1, :] = (18.*tau_12[-2, :] - 9*tau_12[-3, :] + 2*tau_12[-4, :]) / 11

    out += g.dfdx(tau_11) + g.dfdy(tau_12)

def rho_v_rhs(p, f, g, out=None):
    out[...] = -g.dfdy(f.rho_v*f.rho_v/f.rho + f.prs)
    out[[0,-1], :] = 0 
    out += -g.dfdx(f.rho_v*f.rho_u/f.rho )

    div_vel = g.dfdx(f.rho_u / f.rho) + g.dfdy(f.rho_v / f.rho)

    tau_11 = -(2./3)*p.mu_ref*div_vel + 2*p.mu_ref*g.dfdx(f.rho_u / f.rho) 
    tau_22 = -(2./3)*p.mu_ref*div_vel + 2*p.mu_ref*g.dfdy(f.rho_v / f.rho)
    tau_12 = p.mu_ref*(g.dfdx(f.rho_v / f.rho) + g.dfdy(f.rho_u / f.rho))
    
    tau_12[0, :] = (18.*tau_12[1, :] - 9*tau_12[2, :] + 2*tau_12[3, :]) / 11
    tau_12[-1, :] = (18.*tau_12[-2, :] - 9*tau_12[-3, :] + 2*tau_12[-4, :]) / 11

    out += g.dfdx(tau_12) + g.dfdy(tau_22)

def egy_rhs(p, f, g, out=None):
    egy_rhs_x = -g.dfdx((f.egy + f.prs)*(f.rho_u/f.rho))
    egy_rhs_y = -g.dfdy((f.egy + f.prs)*(f.rho_v/f.rho))
    egy_rhs_y[[0,-1], :] = 0

    out[...] = egy_rhs_x + egy_rhs_y

    div_vel = g.dfdx(f.rho_u / f.rho) + g.dfdy(f.rho_v / f.rho)

    tau_11 = -(2./3)*p.mu_ref*div_vel + 2*p.mu_ref*g.dfdx(f.rho_u / f.rho) 
    tau_22 = -(2./3)*p.mu_ref*div_vel + 2*p.mu_ref*g.dfdy(f.rho_v / f.rho)
    tau_12 = p.mu_ref*(g.dfdx(f.rho_v / f.rho) + g.dfdy(f.rho_u / f.rho))
    
    tau_12[0, :] = (18.*tau_12[1, :] - 9*tau_12[2, :] + 2*tau_12[3, :]) / 11
    tau_12[-1, :] = (18.*tau_12[-2, :] - 9*tau_12[-3, :] + 2*tau_12[-4, :]) / 11

    out += (g.dfdx(f.rho_u/f.rho * tau_11) +
                  g.dfdx(f.rho_v/f.rho * tau_12) +
                  g.dfdy(f.rho_u/f.rho * tau_12) +
                  g.dfdy(f.rho_v/f.rho * tau_22) + 
                  p.kappa_ref*(
                    g.dfdx(g.dfdx(f.tmp)) +
                    g.dfdy(g.dfdy(f.tmp))))

def apply_filter(p, f):
    for f in f.rho, f.rho_u, f.rho_v, f.egy:
        filter5(f)

def non_reflecting_boundary_conditions(p, f, g, eos, equations):
    
    rho_rhs = equations[0].rhs
    rho_u_rhs = equations[1].rhs
    rho_v_rhs = equations[2].rhs
    egy_rhs = equations[3].rhs

    C_sound = np.sqrt(eos.Cp/eos.Cv * eos.R*f.tmp)

    dpdy = g.dfdy(f.prs)
    drhody = g.dfdy(f.rho)
    dudy = g.dfdy(f.rho_u/f.rho)
    dvdy = g.dfdy(f.rho_v/f.rho)
    
    L_1 = (f.rho_v/f.rho - C_sound) * (dpdy - f.rho*C_sound*dvdy)
    L_2 = f.rho_v/f.rho * (C_sound**2 * drhody - dpdy)
    L_3 = f.rho_v/f.rho * dudy
    L_4 = 0.4*(1 - p.Ma**2) * C_sound/p.Ly * (f.prs - p.P_inf)

    d_1 = (1. / C_sound**2) * (L_2 + 0.5*(L_4 + L_1))
    d_2 = 0.5*(L_1 + L_4)
    d_3 = L_3
    d_4 = 1./(2*f.rho*C_sound) * (L_4 - L_1)

    rho_rhs[0, :] = (rho_rhs - d_1)[0, :]
    rho_u_rhs[0, :] = (rho_u_rhs - f.rho_u/f.rho*d_1 - f.rho*d_3)[0, :]
    rho_v_rhs[0, :] = (rho_v_rhs - f.rho_v/f.rho*d_1 - f.rho*d_4)[0, :]
    egy_rhs[0, :] = (egy_rhs -
        0.5*np.sqrt((f.rho_u/f.rho)**2 + (f.rho_v/f.rho)**2)*d_1 -
        d_2 * (f.prs + f.egy) / (f.rho*C_sound**2) -
        f.rho * (f.rho_v/f.rho * d_4 + f.rho_u/f.rho * d_3))[0, :]

    L_1 = 0.4 * (1 - p.Ma**2) * C_sound/p.Ly * (f.prs - p.P_inf)
    L_2 = f.rho_v/f.rho * (C_sound**2 * drhody - dpdy)
    L_3 = f.rho_v/f.rho * dudy
    L_4 = (f.rho_v/f.rho + C_sound) * (dpdy + f.rho*C_sound*dvdy)

    d_1 = (1./C_sound**2) * (L_2 + 0.5*(L_4 + L_1))
    d_2 = 0.5*(L_1 + L_4)
    d_3 = L_3
    d_4 = 1/(2*f.rho*C_sound) * (L_4 - L_1)

    rho_rhs[-1, :] = (rho_rhs - d_1)[-1, :]
    rho_u_rhs[-1, :] = (rho_u_rhs - f.rho_u/f.rho*d_1 - f.rho*d_3)[-1, :]
    rho_v_rhs[-1, :] = (rho_v_rhs - f.rho_v/f.rho*d_1 - f.rho*d_4)[-1, :]
    egy_rhs[-1, :] = (egy_rhs-
        0.5*np.sqrt((f.rho_u/f.rho)**2 + (f.rho_v/f.rho)**2)*d_1 -
        d_2 * (f.prs + f.egy) / (f.rho*C_sound**2) -
        f.rho * (f.rho_v/f.rho * d_4 + f.rho_u/f.rho * d_3))[-1, :]

def update_temperature_and_pressure(f, eos):
    f.tmp[...] = (f.egy - 0.5*(f.rho_u**2 + f.rho_v**2)/f.rho) / (f.rho*eos.Cv)
    eos.pressure(f.tmp, f.rho, f.prs)

def rhs(eqvars, equations):

    for eq in equations:
        eq.compute_rhs()

    return equations[0].rhs, equations[1].rhs, equations[2].rhs, equations[3].rhs

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

    # make equations
    rho_eq = Equation(f.rho)
    rho_u_eq = Equation(f.rho_u)
    rho_v_eq = Equation(f.rho_v)
    egy_eq = Equation(f.egy)
    equations = [rho_eq, rho_u_eq, rho_v_eq, egy_eq]

    rho_eq.set_rhs_func(rho_rhs, p, f, g)
    rho_u_eq.set_rhs_func(rho_u_rhs, p, f, g)
    rho_v_eq.set_rhs_func(rho_v_rhs, p, f, g)
    egy_eq.set_rhs_func(egy_rhs, p, f, g)

    rho_eq.set_rhs_pre_func(update_temperature_and_pressure, f, eos)
    egy_eq.set_rhs_post_func(non_reflecting_boundary_conditions, p, f, g, eos, equations)

    # make time stepper
    stepper = RK4(equations)
    
    # run simulation
    import timeit

    for i in range(p.timesteps):
        
        dt = calculate_timestep(p, f, g, eos)

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
