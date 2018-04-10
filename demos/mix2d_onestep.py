
import matplotlib.pyplot as plt
import h5py

from mixlayer.constants import *
from mixlayer.solvers.onestep import OneStepSolver
from mixlayer.derivatives import BoundaryConditionType
from mixlayer.grid.mapped import SinhGrid
from mixlayer.timestepping import RK4
from mixlayer.models.species import *
from mixlayer.models.mixture import *
from mixlayer.models.thermodynamics import *
from mixlayer.models.transport import *
from mixlayer.models.eos import *
from mixlayer.models.reaction import *

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

# grid dimensions
N = 144
Lx = 1
Ly = Lx*((N-1)/N)*2.
grid_beta = 5
grid = SinhGrid(N, (Lx, Ly), (BoundaryConditionType.PERIODIC, BoundaryConditionType.INNER), grid_beta)

# simulation control
Ma = 0.35
Re = 500
Pr =0.697
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

# equation of state:
T_prop = 345.

# thermal conductivity, viscosity and mass diffusivity:
hexane = Species(
    'hexane',
    86.178,
    specific_heat_model=SpeciesSpecificHeatModelConstant(
        -51*31 + 6.767*T_prop - 3.623e-3*T_prop**2),
    gas_model = SpeciesEOSIdealGas(),
    viscosity_model = SpeciesViscosityModelConstant(None)
)

air = Species(
    'air',
    29.87,
    specific_heat_model=SpeciesSpecificHeatModelConstant(
        Pr*(3.227e-3 + 8.389e-5*T_prop - 1.985e-8*T_prop**2)/(
            6.109e-6 + 4.606e-8*T_prop - 1.051e-11*T_prop**2)
    ),
    gas_model = SpeciesEOSIdealGas(),
    viscosity_model = SpeciesViscosityModelConstant(None)
)

products = Species(
    'products',
    (hexane.molecular_weight + rratio*air.molecular_weight)/(1+rratio),
    specific_heat_model=SpeciesSpecificHeatModelConstant(
        (1*hexane.Cp(0, 0)*hexane.molecular_weight +
         rratio*air.Cp(0, 0)*air.molecular_weight) / (
         hexane.molecular_weight+rratio*air.molecular_weight)
    ),
    gas_model = SpeciesEOSIdealGas(),
    viscosity_model = SpeciesViscosityModelConstant(None)
)

# free stream thermodynamic properties
Cp_inf1 = y1_inf1*hexane.Cp(0, 0) + y2_inf1*air.Cp(0, 0)
Cv_inf1 = y1_inf1*hexane.Cv(0, 0) + y2_inf1*air.Cv(0, 0)
R_inf1 =  y1_inf1*hexane.R  + y2_inf1*air.R
Cp_inf2 = y1_inf2*hexane.Cp(0, 0) + y2_inf2*air.Cp(0, 0)
Cv_inf2 = y1_inf2*hexane.Cv(0, 0) + y2_inf2*air.Cv(0, 0)
R_inf2 =  y1_inf2*hexane.R + y2_inf2*air.R

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
U_ref = U_inf1-U_inf2

# viscosity; thermal and molecular diffusivities
disturbance_wavelength = Lx/nperiod
vorticity_thickness = disturbance_wavelength/7.29
mu = (rho_ref*U_ref*vorticity_thickness)/Re
kappa = 0.5*(hexane.Cp(P_inf, T_ref) + air.Cp(P_inf, T_ref)) * mu / Pr
mass_diffusivity = mu/(rho_ref*Pr) # here Pr is the same as the Schmidt number

# fields
fields = {}
dims = (N, N)
rho = np.zeros(dims, dtype=np.float64)
rho_u = np.zeros(dims, dtype=np.float64)
rho_v = np.zeros(dims, dtype=np.float64)
egy = np.zeros(dims, dtype=np.float64)
rho_y1 = np.zeros(dims, dtype=np.float64)
rho_y2 = np.zeros(dims, dtype=np.float64)
rho_y3 = np.zeros(dims, dtype=np.float64)
tmp = np.zeros(dims, dtype=np.float64)
prs = np.zeros(dims, dtype=np.float64)

fields['rho'] = rho
fields['rho_u'] = rho_u
fields['rho_v'] = rho_v
fields['egy'] = egy
fields['rho_y1'] = rho_y1
fields['rho_y2'] = rho_y2
fields['rho_y3'] = rho_y3
fields['tmp'] = tmp
fields['prs'] = prs

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

mixture = Mixture([hexane, air, products], [y1, y2, y3],
    viscosity_model=MixtureViscosityModelConstant(mu),
    thermal_conductivity_model=MixtureThermalConductivityModelConstant(kappa),
    mass_diffusivity_model=MassDiffusivityModelConstant(mass_diffusivity),
    gas_model=MixtureEOSIdealGas(),
    specific_heat_model=MixtureSpecificHeatModelMassWeighted())

rho[:, :] = P_inf/(mixture.R*tmp[:, :])

rho_u[:, :] = rho*(U_inf2+(weight+1)/2.*(U_inf1-U_inf2))

rho_v[:, :] = 0.0

u_pert, v_pert = add_forcing()

rho_u += rho*u_pert
rho_v += rho*v_pert

egy[:, :] = 0.5*(rho_u**2 + rho_v**2)/rho + rho*mixture.Cv(prs, tmp)*tmp

rho_y1[...] = rho * y1
rho_y2[...] = rho * y2
rho_y3[...] = rho * y3

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

products.enthalpy_of_formation = -heat_release_parameter*Cp_inf2*T_ref

reaction = OneStepReaction(arrhenius_coefficient, activation_energy)

# make solver
solver = OneStepSolver(mixture,
        grid,
        fields,
        reaction,
        rratio,
        RK4,
        Ma,
        P_inf,
        T_ref)

# run simulation
import timeit

outfile = h5py.File("results.hdf5", "w")
outfile.create_dataset("grid/x", data=grid.x)
outfile.create_dataset("grid/y", data=grid.y)

for i in range(timesteps):

    print("Iteration: {:10d}    Total energy: {:15.10e}".format(i, np.sum(egy)))

    solver.step()

    if i%50 == 0:
        outfile.create_group(str(i))
        outfile.create_dataset("{}/rho".format(i), data=rho)
        outfile.create_dataset("{}/rho_u".format(i), data=rho_u)
        outfile.create_dataset("{}/rho_v".format(i), data=rho_v)
        outfile.create_dataset("{}/tmp".format(i), data=tmp)
        outfile.create_dataset("{}/rho_y1".format(i), data=rho_y1)
        outfile.create_dataset("{}/rho_y2".format(i), data=rho_y2)
        outfile.create_dataset("{}/rho_y3".format(i), data=rho_y3)
