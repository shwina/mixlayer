from mixlayer.constants import *

class OneStepReaction:
    """
    Single step arrhenius reaction
    """
    def __init__(self, arrhenius_coefficient, activation_energy):
        self.arrhenius_coefficient = arrhenius_coefficient
        self.activation_energy = activation_energy

    def reaction_rate(self, T):
        return arrhenius_coefficient * np.exp(
            -activation_enery/(universal_gas_constant * T))
