from mixlayer.constants import universal_gas_constant

class Species:

    def __init__(self, species_name,
                 molecular_weight,
                 thermodynamics_model=None,
                 transport_model=None,
                 gas_model=None):
        """
        species_name : str
        molecular_weight : float
        thermodynamics : ThermodynamicsModel
        transport_model : TransportModel
        gas_model : GasModel
        """
        self.species_name = species_name
        self.molecular_weight = molecular_weight
        self.thermodynamics_model = thermodynamics_model
        self.transport_model = transport_model
        self.gas_model = gas_model

    def Cp(self, p, T):
        return self.thermodynamics_model.Cp(p, T)
    
    def Hf(self, p, T):
        return self.thermodynamics_model.Hf(Hf, T)

    def Cv(self, p, T):
        return self.thermodynamics_model.Cp(p, T) - (
            self.gas_model.CpMCv(self, p, T))

    def mu(self, p, T):
        return self.transport_model.mu(p, T)

    def Pr(self, p, T):
        return self.transport_model.Pr(p, T)

    def kappa(self, p, T):
        return (self.thermodynamics_model.Cp(P, T) * self.transport_model.mu(p, T))/(
            self.transport_model.Pr(p, T))

    @property
    def R(self):
        return universal_gas_constant / self.molecular_weight
