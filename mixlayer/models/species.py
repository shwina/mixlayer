from mixlayer.constants import universal_gas_constant
class Species:
    def __init__(self, species_name,
                 molecular_weight,
                 gas_model=None,
                 specific_heat_model=None,
                 thermal_conductivity_model=None,
                 viscosity_model=None,
                 enthalpy_of_formation=0):
        """
        species_name : str
        molecular_weight : float
        gas_model : GasModel
        specific_heat_model: ThermodynamicsModel
        viscosity_mode: TransportModel
        """
        self.species_name = species_name
        self.molecular_weight = molecular_weight
        self.gas_model = gas_model
        self.specific_heat_model=specific_heat_model
        self.viscosity_model = viscosity_model
        self.enthalpy_of_formation = 0

    def Cp(self, p, T):
        return self.specific_heat_model.Cp(p, T)

    def Cv(self, p, T):
        return self.specific_heat_model.Cp(p, T) - (
            self.gas_model.CpMCv(p, T))

    def mu(self, p, T):
        return self.viscosity.mu(p, T)

    def kappa(self, p, T):
        self.thermal_conductivity_model.kappa(p, T)

    @property
    def R(self):
        return universal_gas_constant / self.molecular_weight
