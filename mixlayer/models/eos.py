import numpy as np

class IdealGasEOS(object):

    def __init__(self, Cp, Cv, R):
        self.Cp = Cp
        self.Cv = Cv
        self.R = R

    def get_pressure(self, temperature, density, out):
        out[...] = density*self.R*temperature
