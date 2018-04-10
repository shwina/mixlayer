import numpy as np
from mixlayer.constants import *

class Mixture:
    def __init__(self, species_list, Y,
                gas_model=None,
                specific_heat_model=None,
                viscosity_model=None,
                thermal_conductivity_model=None,
                mass_diffusivity_model=None):
        self.species_list = species_list
        self.Y = Y
        self.gas_model = gas_model
        self.specific_heat_model = specific_heat_model
        self.viscosity_model = viscosity_model
        self.thermal_conductivity_model = thermal_conductivity_model
        self.mass_diffusivity_model = mass_diffusivity_model

        self.gas_model.mixture = self

    @property
    def molecular_weight(self):
        M_inv = 0
        for specie, Yi in zip(self.species_list, self.Y):
            M_inv += Yi/(specie.molecular_weight)
        return 1/M_inv

    def Cp(self, p, T):
        return self.specific_heat_model.Cp(p, T)
    
    def Cv(self, p, T):
        return self.Cp(p, T) - self.gas_model.CpMCv(p, T)

    def mu(self, p, T):
        return self.viscosity_model.mu(p, T)

    def kappa(self, p, T):
        return self.thermal_conductivity_model.kappa(p, T)

    def D(self, specie, p, T):
        return self.mass_diffusivity_model.D(specie, p, T)

    @property
    def R(self):
        return universal_gas_constant / self.molecular_weight
