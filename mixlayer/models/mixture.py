import numpy as np

class Mixture:
    def __init__(self, species_list, Y):
        self.species_list = species_list
        self.Y = Y
        grid_dims = Y[0].shape
        self._Cp = np.zeros(grid_dims)
        self._Cv = np.zeros(grid_dims)
        self._R = np.zeros(grid_dims)

    def Cp(self, p, T):
        self._Cp[...] = 0
        for Yi, specie in zip(self.Y, self.species_list):
            self._Cp += Yi*specie.Cp(p, T)
        return self._Cp
    
    def Cv(self, p, T):
        self._Cv[...] = 0
        for Yi, specie in zip(self.Y, self.species_list):
            self._Cv += Yi*specie.Cv(p, T)
        return self._Cv

    @property
    def R(self):
        self._R[...] = 0
        for Yi, specie in zip(self.Y, self.species_list):
            self._R += Yi*specie.R
        return self._R
