import numpy as np
"""
Models for evaluating pure-component specific heat
and mixture specific heat.
"""
class SpeciesSpecificHeatModelConstant(object):
    def __init__(self, Cp):
        self._Cp = Cp

    def Cp(self, p, T):
        return self._Cp

class MixtureSpecificHeatModelMassWeighted(object):
    def __init__(self, mixture):
        self.mixture = mixture
        grid_dims = mixture.Y[0].shape
        self._Cp = np.zeros(grid_dims, dtype=np.float64)

    def Cp(self, p, T):
        self._Cp[...] = 0
        for Yi, specie in zip(self.mixture.Y,
                              self.mixture.species_list):
            self._Cp += Yi * specie.Cp(p, T)
        return self._Cp

    def Cv(self, p, T):
        self._Cv[...] = 0
        for Yi, specie in zip(self.mixture.Y,
                              self.mixture.species_list):
            self._Cv += Yi * specie.Cv(p, T)
        return self._Cv
