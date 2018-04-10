import numpy as np

class SpeciesEOSIdealGas(object):
    def __init__(self):
        self.specie = None

    def CpMCv(self, p, T):
        return self.specie.R

    def P(self, rho, T):
        return rho * self.specie.R * T

class MixtureEOSIdealGas(object):
    def __init__(self):
        self.mixture = None

    def CpMCv(self, p, T):
        return self.mixture.R

    def P(self, rho, T):
        return rho * self.mixture.R * T
