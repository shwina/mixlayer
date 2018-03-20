class SpeciesViscosityModelConstant:
    def __init__(self, mu):
        self._mu = mu

    def mu(self, p, T):
        return self._mu

class SpeciesThermalConductivityModelConstant:
    def __init__(self, kappa):
        self._kappa = kappa

    def kappa(self, p, T):
        return self._kappa

class MassDiffusivityModelConstant:
    def __init__(self, D):
        self._D = D

    def D(self, specie, p, T):
        return self._D

MixtureViscosityModelConstant = SpeciesViscosityModelConstant
MixtureThermalConductivityModelConstant = SpeciesThermalConductivityModelConstant
