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
    def __init__(self):
        self.mixture = None

    def Cp(self, p, T):
        out = 0
        for Yi, specie in zip(self.mixture.Y,
                              self.mixture.species_list):
            out += Yi * specie.Cp(p, T)
        return out

    def Cv(self, p, T):
        out = 0
        for Yi, specie in zip(self.mixture.Y,
                              self.mixture.species_list):
            out += Yi * specie.Cv(p, T)
        return out
