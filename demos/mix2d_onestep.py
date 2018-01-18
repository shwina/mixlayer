import numpy as np
import matplotlib.pyplot as plt
import h5py

from mixlayer.constants import *
from mixlayer.solvers.onestep import OneStepSolver
from mixlayer.derivatives import BoundaryType
from mixlayer.grid.mapped import SinhGrid
from mixlayer.timestepping import RK4
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

    u_pert =  (np.roll(stream, -1, 0) - np.roll(stream, 1, 0))/(2*dy)
    v_pert = -(np.roll(stream, -1, 1) - np.roll(stream, 1, 1))/(2*dx)

    vort[...] = ((np.roll(stream, -1, 1) - 2*stream + np.roll(stream, 1, 1))/(dx**2) +
                 (np.roll(stream, -1, 0) - 2*stream + np.roll(stream, 1, 0))/(dy**2))

    circ = np.sum(dy*dx*vort)

    u_pert = u_pert*Famp_2d*disturbance_wavelength*U_ref / (circ/nperiod)
    v_pert = v_pert*Famp_2d*disturbance_wavelength*U_ref / (circ/nperiod)

    return u_pert, v_pert

def calculate_timestep():

    cfl_vel = 0.5
    cfl_visc = 0.1

    update_temperature_and_pressure()

    dxmin = np.minimum(grid.dx, grid.dy)

    # calculate diffusivities:
    alpha_1 = gamma
    alpha_2 = mu/rho
    alpha_3 = kappa/(Cp*rho)
    alpha_max = np.maximum(np.maximum(alpha_1, alpha_2), alpha_3)

    # calculate C_sound
    C_sound = np.sqrt(Cp/Cv*R*tmp)
    test_1 = cfl_vel*grid.dx/(C_sound + abs(rho_u/rho))
    test_2 = cfl_vel*grid.dy/(C_sound + abs(rho_v/rho))
    test_3 = cfl_visc*(dxmin**2)/alpha_max

    dt = np.min(np.minimum(np.minimum(test_1, test_2), test_3))
    return dt

def non_reflecting_boundary_conditions():

    dfdx, dfdy = grid.dfdx, grid.dfdy
    C_sound = np.sqrt(Cp/Cv * R*tmp)

    dpdy = dfdy(prs)
    drhody = dfdy(rho)
    dudy = dfdy(rho_u/rho)
    dvdy = dfdy(rho_v/rho)
    dy1dy = dfdy(rho_y1/rho)
    dy2dy = dfdy(rho_y2/rho)
    dy3dy = dfdy(rho_y3/rho)
    
    L_1 = (rho_v/rho - C_sound) * (dpdy - rho*C_sound*dvdy)
    L_2 = rho_v/rho * (C_sound**2 * drhody - dpdy)
    L_3 = rho_v/rho * dudy
    L_4 = 0.4*(1 - Ma**2) * C_sound/Ly * (prs - P_inf)
    L_5 = (rho_v/rho)*dy1dy
    L_6 = (rho_v/rho)*dy2dy
    L_7 = (rho_v/rho)*dy3dy

    d_1 = (1. / C_sound**2) * (L_2 + 0.5*(L_4 + L_1))
    d_2 = 0.5*(L_1 + L_4)
    d_3 = L_3
    d_4 = 1./(2*rho*C_sound) * (L_4 - L_1)
    d_5 = L_5
    d_6 = L_6
    d_7 = L_7

    rho_rhs[0, :] = (rho_rhs - d_1)[0, :]
    rho_u_rhs[0, :] = (rho_u_rhs - rho_u/rho*d_1 - rho*d_3)[0, :]
    rho_v_rhs[0, :] = (rho_v_rhs - rho_v/rho*d_1 - rho*d_4)[0, :]
    egy_rhs[0, :] = (egy_rhs -
        0.5*np.sqrt((rho_u/rho)**2 + (rho_v/rho)**2)*d_1 -
        d_2 * (prs + egy) / (rho*C_sound**2) -
        rho * (rho_v/rho * d_4 + rho_u/rho * d_3))[0, :]
    rho_y1_rhs[0, :] = (rho_y1_rhs - rho_y1/rho*d_1 - rho*d_5)[0, :]
    rho_y2_rhs[0, :] = (rho_y2_rhs - rho_y2/rho*d_1 - rho*d_6)[0, :]
    rho_y3_rhs[0, :] = (rho_y3_rhs - rho_y3/rho*d_1 - rho*d_7)[0, :]

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
    rho_y1_rhs[-1, :] = (rho_y1_rhs - rho_y1/rho*d_1 - rho*d_5)[-1, :]
    rho_y2_rhs[-1, :] = (rho_y2_rhs - rho_y2/rho*d_1 - rho*d_6)[-1, :]
    rho_y3_rhs[-1, :] = (rho_y3_rhs - rho_y3/rho*d_1 - rho*d_7)[-1, :]

def update_temperature_and_pressure():
    y1 = U[4]/U[0]
    y2 = U[5]/U[0]
    y3 = U[6]/U[0]
    Cp[...] = y1*Cp1 + y2*Cp2 + y3*Cp3
    Cv[...] = y1*Cv1 + y2*Cv2 + y3*Cv3
    R[...] = y1*R1 + y2*R2 + y3*R3
    tmp[...] = (egy - 0.5*(rho_u**2 + rho_v**2)/rho) / (rho*Cv)
    eos.get_pressure(tmp, rho, prs)

    xy = 1 - y1 - y2 - y3
    U[4][ y1 > 0.5 ] += (rho*xy) [ y1 > 0.5 ]
    U[5][ y1 < 0.5 ] += (rho*xy) [ y1 < 0.5 ]

    # ensure that mass fractions are 0 <= y <= 1
    for rho_yi in U[4:]:
        rho_yi [rho_yi < 0] = 0.
        rho_yi [rho_yi > rho] = rho [rho_yi > rho]

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
timesteps = 20000
writer = True
Famp_2d = 0.4

# free stream temperature and pressure
P_inf = 101325.
T_inf1 = 300.
T_inf2 = 300.

# free stream mass fractions
y1_inf1 = 1 # mass fraction of fuel in fuel stream
y2_inf1 = 0
y1_inf2 = 0 # mass fraction of fuel in air stream
y2_inf2 = 1

# reference temperature
T_ref = T_inf2

# fuel, air and product properties:
rratio = 1

# hexane
T_prop = 345.

molwt_1 = 86.178
R1 = universal_gas_constant/molwt_1
Cp1 = -51.31 + 6.767*T_prop - 3.623e-3*T_prop**2
Cv1 = Cp1 - R1

# air
molwt_2 = 28.97
R2 = universal_gas_constant/molwt_2
Cp2 = Pr*(3.227e-3 + 8.389e-5*T_prop - 1.985e-8*T_prop**2)/(
        6.109e-6 + 4.606e-8*T_prop - 1.051e-11*T_prop**2)
Cv2 = Cp2 - R2

# product
molwt_3 = (molwt_1 + rratio*molwt_2)/(1+rratio)
R3 = universal_gas_constant/molwt_3
Cv3 = (Cv1*molwt_1 + rratio*Cv2*molwt_2)/(
        (1+rratio)*molwt_3)
Cp3 = Cv3 + R3

# free stream thermodynamic properties
Cp_inf1 = y1_inf1*Cp1 + y2_inf1*Cp2
Cv_inf1 = y1_inf1*Cv1 + y2_inf1*Cv2
R_inf1 =  y1_inf1*R1  + y2_inf1*R2
Cp_inf2 = y1_inf2*Cp1 + y2_inf2*Cp2
Cv_inf2 = y1_inf2*Cv1 + y2_inf2*Cv2
R_inf2 =  y1_inf2*R1 + y2_inf2*R2

# reference density
rho_inf1 = P_inf/(R_inf1*T_inf1)
rho_inf2 = P_inf/(R_inf2*T_inf2)
rho_ref = (rho_inf1+rho_inf2)/2.0

# reference velocities 
C_sound_inf1 = np.sqrt(Cp_inf1/Cv_inf1 * R_inf1*T_inf1)
C_sound_inf2 = np.sqrt(Cp_inf2/Cv_inf2 * R_inf2*T_inf2)
U_inf1 = 2*Ma*C_sound_inf1/(
        1+np.sqrt(rho_inf1/rho_inf2)*(C_sound_inf1/C_sound_inf2))
U_inf2 = -np.sqrt(rho_inf1/rho_inf2)*U_inf1
print(U_inf1, U_inf2)
U_ref = U_inf1-U_inf2

# viscosity; thermal and molecular diffusivities
disturbance_wavelength = Lx/nperiod
vorticity_thickness = disturbance_wavelength/7.29
mu = (rho_ref*U_ref*vorticity_thickness)/Re
kappa = 0.5*(Cp1+Cp2)*mu/Pr 
gamma = mu/(rho_ref*Pr)

# fields
dims = (N, N)
U = np.zeros((4+3,)+dims, dtype=np.float64)
rhs = np.copy(U)
rho, rho_u, rho_v, egy, rho_y1, rho_y2, rho_y3 = U
rho_rhs, rho_u_rhs, rho_v_rhs, egy_rhs, rho_y1_rhs, rho_y2_rhs, rho_y3_rhs = rhs
tmp = np.zeros(dims, dtype=np.float64)
prs = np.zeros(dims, dtype=np.float64)
Cp = np.zeros(dims, dtype=np.float64)
Cv = np.zeros(dims, dtype=np.float64)
R = np.zeros(dims, dtype=np.float64)

T_flame_0 = 600
# initialize fields
weight = np.tanh(np.sqrt(np.pi)*grid.y/vorticity_thickness)
tmp[...] = T_inf2 + (weight+1)/2.*(T_inf1-T_inf2)
tmp[ grid.y < 0 ] += ((weight+1)*(T_flame_0 - 0.5*(T_inf1+T_inf2)))[ grid.y < 0 ]
tmp[ grid.y >= 0 ] += ((1-weight)*(T_flame_0 - 0.5*(T_inf1+T_inf2)))[ grid.y >= 0]

ymix = y2_inf2 + (weight+1)/2*(y2_inf1-y2_inf2)
y1 = (-1+2*ymix) * (ymix > 0.5)
y2 = ( 1-2*ymix) * (ymix <= 0.5)
y3 = 1 - y1 - y2

Cp[...] = y1*Cp1 + y2*Cp2 + y3*Cp3
Cv[...] = y1*Cv1 + y2*Cv2 + y3*Cv3
R[...] = y1*R1 + y2*R2 + y3*R3

rho[:, :] = P_inf/(R*tmp[:, :])

print(P_inf)

rho_u[:, :] = rho*(U_inf2+(weight+1)/2.*(U_inf1-U_inf2))

rho_v[:, :] = 0.0

u_pert, v_pert = add_forcing()

rho_u += rho*u_pert
rho_v += rho*v_pert

egy[:, :] = 0.5*(rho_u**2 + rho_v**2)/rho + rho*Cv*tmp

rho_y1[...] = rho * y1
rho_y2[...] = rho * y2
rho_y3[...] = rho * y3

# equation of state:
eos = IdealGasEOS(Cp, Cv, R)


# reaction parameters
Da = 10.
Da_T_flame = 1.0
Da_inf = 0.0001
heat_release_parameter = 3.0
rratio = 1.0

T_flame = T_inf2*(1+heat_release_parameter)
Ze = (1+heat_release_parameter)/heat_release_parameter*np.log(
        Da/Da_inf)
activation_energy = Ze * universal_gas_constant * T_ref
arrhenius_coefficient = Da * (U_ref/vorticity_thickness) / np.exp(
        -activation_energy/(universal_gas_constant*T_flame))
enthalpy_of_formation = -heat_release_parameter*Cp_inf2*T_ref

# make solver
solver = OneStepSolver(grid, U, rhs, tmp, prs, mu, kappa, gamma,
        arrhenius_coefficient, activation_energy, rratio, enthalpy_of_formation,
        molwt_1, molwt_2, molwt_3, RK4)
solver.set_rhs_pre_func(update_temperature_and_pressure)
solver.set_rhs_post_func(non_reflecting_boundary_conditions)

# run simulation
import timeit

for i in range(timesteps):
 
    dt = calculate_timestep()

    print("Iteration: {:10d}    Time: {:15.10e}    Total energy: {:15.10e}".format(i, dt*i, np.sum(egy)))

    solver.step(dt)

    ytest = (U[5]/U[0])

    print(ytest.min(), ytest.max())

    if writer:
        if i%50 == 0:
            outfile = h5py.File("{:05d}.hdf5".format(i))
            outfile.create_group("fields")
            outfile.create_dataset("fields/rho", data=rho)
            outfile.create_dataset("fields/rho_u", data=rho_u)
            outfile.create_dataset("fields/rho_v", data=rho_v)
            outfile.create_dataset("fields/tmp", data=tmp)
            outfile.create_dataset("fields/rho_y1", data=rho_y1)
            outfile.create_dataset("fields/rho_y2", data=rho_y2)
            outfile.create_dataset("fields/rho_y3", data=rho_y3)
