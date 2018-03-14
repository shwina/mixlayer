import numpy as np

class IdealGasEOS(object):
    def __init__(self):
        pass

    @classmethod
    def CpMCv(self, species, p, T):
        return species.R

    def P(self, species, rho, T):
        return rho * species.R * T

class SolidSpecies(object):
    def __init__(self, species):
        pass
