class hConstThermodynamicsModel:
    """
    Constant values for Cp and Hf
    """
    def __init__(self, Cp, Hf):
        self._Cp = Cp
        self._Hf = Hf

    def Cp(self, p, T):
        return self._Cp

    def Hf(self, p, T):
        return self._Hf
