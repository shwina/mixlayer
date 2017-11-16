import numpy as np
import yaml
from attrdict import AttrDict

class Params(object):

    def __init__(self, paramfile):
        self._paramfile = paramfile
        with open(paramfile) as f:
            p = AttrDict({})
            p.update(yaml.load(f))

        # dimensions
        self.N = p.N
        self.Lx = p.Lx

        # simulation control
        self.Ma = p.Ma
        self.Re = p.Re 
        self.Pr = p.Pr 
        self.nperiod = p.nperiod 
        self.timesteps = p.timesteps 
        self.writer = p.writer

        # constants
        self.Rgas = p.Rgas 
        self.Cp = p.Cp 
        self.Cv = p.Cv 

        # initial conditions
        self.P_inf = p.P_inf 
        self.T_inf1 = p.T_inf1 
        self.T_inf2 = p.T_inf2 

        # forcing
        self.Famp_2d = p.Famp_2d 

        # air (2) and reactant (1) properties
        self.molwt = p.molwt 

        # clustering parameters:
        self.grid_beta = p.grid_beta

        # time stepping criteria:
        self.cfl_vel = p.cfl_vel
        self.cfl_visc = p.cfl_visc

        # filtering:
        self.filter_amplitude = p.filter_amplitude

        self.Ly = self.Lx*((self.N-1)/self.N)*2.

        # reference temperature
        self.T_ref = max([self.T_inf2, 344.6])

        # eos parameters
        self.Rspecific = 287

        # reference density
        self.rho_ref1 = self.P_inf/(self.Rspecific*self.T_inf1)
        self.rho_ref2 = self.P_inf/(self.Rspecific*self.T_inf2)
        self.rho_ref = (self.rho_ref1+self.rho_ref2)/2.0

        # reference velocities 
        self.C_sound1 = np.sqrt((self.Cp/self.Cv)*(self.Rspecific)*self.T_inf1)
        self.C_sound2 = np.sqrt((self.Cp/self.Cv)*(self.Rspecific)*self.T_inf2)
        self.U_inf1 = 2*self.Ma*self.C_sound1/(1+np.sqrt(self.rho_ref1/self.rho_ref2)*(self.C_sound1/self.C_sound2))
        self.U_inf2 = -np.sqrt(self.rho_ref1/self.rho_ref2)*self.U_inf1
        self.U_ref = self.U_inf1-self.U_inf2

        # grid parameters
        self.dx = self.Lx/self.N
        self.dn = 1./(self.N-1)

        # geometric parameters
        self.disturbance_wavelength = self.Lx/self.nperiod
        self.vorticity_thickness = self.disturbance_wavelength/7.29

        # reference viscosity; thermal and molecular diffusivities
        self.rho_ref = (self.rho_ref1+self.rho_ref2)/2.0
        self.mu_ref = (self.rho_ref*(self.U_inf1-self.U_inf2)*self.vorticity_thickness)/self.Re
        self.kappa_ref = 0.5*(self.Cp+self.Cp)*self.mu_ref/self.Pr 
        self.gamma_ref = self.mu_ref/(self.rho_ref*self.Pr)

