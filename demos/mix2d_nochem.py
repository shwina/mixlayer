import numpy as np

import h5py
from mixlayer.solvers.nochem import NoChemistrySolver
from mixlayer.derivatives import BoundaryType
from mixlayer.grid.mapped import SinhGrid
from mixlayer.timestepping import RK4
from mixlayer.filtering import filter5
from mixlayer.models.eos import IdealGasEOS
from mixlayer.poisson import PoissonSolver

def add_forcing():

    x, y, dx, dy = grid.x, grid.y, grid.dx, grid.dy
    dfdx, dfdy = grid.dfdx, grid.dfdy
        
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

    u_pert =  (np.roll(stream, -1, 1) - np.roll(stream, 1, 1))/(2*dy)
    v_pert = -(np.roll(stream, -1, 0) - np.roll(stream, 1, 0))/(2*dx)

    vort[...] = ((np.roll(stream, -1, 1) - 2*stream + np.roll(stream, 1, 1))/(dx**2) +
                 (np.roll(stream, -1, 0) - 2*stream + np.roll(stream, 1, 0))/(dy**2))

    circ = np.sum(dy*dx*vort)

    u_pert = u_pert*Famp_2d*disturbance_wavelength*U_ref / (circ/nperiod)
    v_pert = v_pert*Famp_2d*disturbance_wavelength*U_ref / (circ/nperiod)

    return u_pert, v_pert

def calculate_timestep():

    cfl_vel = 0.5
    cfl_visc = 0.1

    tmp[...] = (egy -
            0.5*(rho_u**2 + rho_v**2)/rho
        ) / (rho*eos.Cv)

    eos.get_pressure(tmp, rho, prs)

    dxmin = np.minimum(grid.dx, grid.dy)

    # calculate diffusivities:
    alpha_1 = gamma
    alpha_2 = mu/rho
    alpha_3 = kappa/(eos.Cp*rho)
    alpha_max = np.maximum(np.maximum(alpha_1, alpha_2), alpha_3)

    # calculate C_sound
    C_sound = np.sqrt(eos.Cp/eos.Cv*eos.R*tmp)
    test_1 = cfl_vel*grid.dx/(C_sound + abs(rho_u/rho))
    test_2 = cfl_vel*grid.dy/(C_sound + abs(rho_v/rho))
    test_3 = cfl_visc*(dxmin**2)/alpha_max

    dt = np.min(np.minimum(np.minimum(test_1, test_2), test_3))
    return dt

def apply_filter():
    for f in U:
        filter5(f)

def non_reflecting_boundary_conditions():

    dfdx, dfdy = grid.dfdx, grid.dfdy
    C_sound = np.sqrt(eos.Cp/eos.Cv * eos.R*tmp)

    dpdy = dfdy(prs)
    drhody = dfdy(rho)
    dudy = dfdy(rho_u/rho)
    dvdy = dfdy(rho_v/rho)
    
    L_1 = (rho_v/rho - C_sound) * (dpdy - rho*C_sound*dvdy)
    L_2 = rho_v/rho * (C_sound**2 * drhody - dpdy)
    L_3 = rho_v/rho * dudy
    L_4 = 0.4*(1 - Ma**2) * C_sound/Ly * (prs - P_inf)

    d_1 = (1. / C_sound**2) * (L_2 + 0.5*(L_4 + L_1))
    d_2 = 0.5*(L_1 + L_4)
    d_3 = L_3
    d_4 = 1./(2*rho*C_sound) * (L_4 - L_1)

    rho_rhs[0, :] = (rho_rhs - d_1)[0, :]
    rho_u_rhs[0, :] = (rho_u_rhs - rho_u/rho*d_1 - rho*d_3)[0, :]
    rho_v_rhs[0, :] = (rho_v_rhs - rho_v/rho*d_1 - rho*d_4)[0, :]
    egy_rhs[0, :] = (egy_rhs -
        0.5*np.sqrt((rho_u/rho)**2 + (rho_v/rho)**2)*d_1 -
        d_2 * (prs + egy) / (rho*C_sound**2) -
        rho * (rho_v/rho * d_4 + rho_u/rho * d_3))[0, :]

    L_1 = 0.4 * (1 - Ma**2) * C_sound/Ly * (prs - P_inf)
    L_2 = rho_v/rho * (C_sound**2 * drhody - dpdy)
    L_3 = rho_v/rho * dudy
    L_4 = (rho_v/rho + C_sound) * (dpdy + rho*C_sound*dvdy)

    d_1 = (1./C_sound**2) * (L_2 + 0.5*(L_4 + L_1))
    d_2 = 0.5*(L_1 + L_4)
    d_3 = L_3
    d_4 = 1/(2*rho*C_sound) * (L_4 - L_1)

    rho_rhs[-1, :] = (rho_rhs - d_1)[-1, :]
    rho_u_rhs[-1, :] = (rho_u_rhs - rho_u/rho*d_1 - rho*d_3)[-1, :]
    rho_v_rhs[-1, :] = (rho_v_rhs - rho_v/rho*d_1 - rho*d_4)[-1, :]
    egy_rhs[-1, :] = (egy_rhs-
        0.5*np.sqrt((rho_u/rho)**2 + (rho_v/rho)**2)*d_1 -
        d_2 * (prs + egy) / (rho*C_sound**2) -
        rho * (rho_v/rho * d_4 + rho_u/rho * d_3))[-1, :]

def update_temperature_and_pressure():
    tmp[...] = (egy - 0.5*(rho_u**2 + rho_v**2)/rho) / (rho*eos.Cv)
    eos.get_pressure(tmp, rho, prs)

# grid dimensions
N = 144
Lx = 1
Ly = Lx*((N-1)/N)*2.
grid_beta = 5
grid = SinhGrid(N, (Lx, Ly), (BoundaryType.PERIODIC, BoundaryType.INNER), grid_beta)

# simulation control
Ma = 0.35
Re = 400
Pr = 0.697
nperiod = 8 # number of perturbation wavelengths        
timesteps = 10000
writer = True
Famp_2d = 0.4

# reference temperature and pressure
P_inf = 101325.
T_inf1 = 300.
T_inf2 = 300.

# reference temperature
T_ref = max([T_inf2, 344.6])

# equation of state
eos = IdealGasEOS()

# reference density
rho_ref1 = P_inf/(eos.R*T_inf1)
rho_ref2 = P_inf/(eos.R*T_inf2)
rho_ref = (rho_ref1+rho_ref2)/2.0

# reference velocities 
C_sound1 = np.sqrt((eos.Cp/eos.Cv)*(eos.R)*T_inf1)
C_sound2 = np.sqrt((eos.Cp/eos.Cv)*(eos.R)*T_inf2)
U_inf1 = 2*Ma*C_sound1/(1+np.sqrt(rho_ref1/rho_ref2)*(C_sound1/C_sound2))
U_inf2 = -np.sqrt(rho_ref1/rho_ref2)*U_inf1
U_ref = U_inf1-U_inf2

# viscosity; thermal and molecular diffusivities
disturbance_wavelength = Lx/nperiod
vorticity_thickness = disturbance_wavelength/7.29
rho_ref = (rho_ref1+rho_ref2)/2.0
mu = (rho_ref*(U_inf1-U_inf2)*vorticity_thickness)/Re
kappa = 0.5*(eos.Cp+eos.Cp)*mu/Pr 
gamma = mu/(rho_ref*Pr)

# fields
dims = (N, N)
U = np.zeros((4,)+dims, dtype=np.float64)
rhs = np.copy(U)
rho, rho_u, rho_v, egy = U
rho_rhs, rho_u_rhs, rho_v_rhs, egy_rhs = rhs
tmp = np.zeros(dims, dtype=np.float64)
prs = np.zeros(dims, dtype=np.float64)

# initialize fields
weight = np.tanh(np.sqrt(np.pi)*grid.y/vorticity_thickness)

tmp[:, :] = T_inf2 + (weight+1)/2.*(T_inf1-T_inf2)
rho[:, :] = P_inf/(eos.R*tmp[:, :])

rho_u[:, :] = rho*(U_inf2+(weight+1)/2.*(U_inf1-U_inf2))
rho_v[:, :] = 0.0

u_pert, v_pert = add_forcing()

rho_u += rho*u_pert
rho_v += rho*v_pert

egy[:, :] = 0.5*(rho_u**2 + rho_v**2)/rho + rho*eos.Cv*tmp

# make solver
solver = NoChemistrySolver(grid, U, rhs, tmp, prs, mu, kappa, RK4)
solver.set_rhs_pre_func(update_temperature_and_pressure)
solver.set_rhs_post_func(non_reflecting_boundary_conditions)

# run simulation
import timeit

for i in range(timesteps):
    
    dt = calculate_timestep()

    print("Iteration: {:10d}    Time: {:15.10e}    Total energy: {:15.10e}".format(i, dt*i, np.sum(egy)))

    solver.step(dt)
    apply_filter()

    if writer:
        if i%200 == 0:
            outfile = h5py.File("{:05d}.hdf5".format(i))
            outfile.create_group("fields")
            outfile.create_dataset("fields/rho", data=rho)
            outfile.create_dataset("fields/rho_u", data=rho_u)
            outfile.create_dataset("fields/rho_v", data=rho_v)
            outfile.create_dataset("fields/tmp", data=tmp)
