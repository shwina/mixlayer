import numpy as np

class IdealGasEOS(object):

    def __init__(self):
        self.Cp = 1005.0
        self.Cv = 718.0
        self.R = 287.0

    def pressure(self, temperature, density, out):
        out[...] = density*self.R*temperature
