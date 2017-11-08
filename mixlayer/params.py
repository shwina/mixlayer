import yaml
from attrdict import AttrDict
import numpy as np

p = AttrDict({})
with open('params.yaml') as f:
    p.update(yaml.load(f))

# dimensions
p.Ly = p.Lx*((p.N-1)/p.N)*2.

# reference temperature
p.T_ref = max([p.T_inf2, 344.6])

# eos parameters
p.Rspecific = p.Rgas/p.molwt
p.Cp = p.Pr*(3.227e-3 + 8.3894e-5*p.T_ref - 1.958e-8*p.T_ref**2)/(
        6.109e-6 + 4.604e-8*p.T_ref - 1.051e-11*p.T_ref**2)
p.Cv = p.Cp - p.Rspecific

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

