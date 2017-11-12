import numpy as np

class IdealGasEOS(object):

    def __init__(self, params, fields):
        self.fields = fields
        self.params = params

    def update_pressure(self):
        self.fields.tmp[:, :] = (
                self.fields.egy -
                0.5*(self.fields.rho_u**2 + self.fields.rho_v**2)/
                     self.fields.rho
                )/(self.fields.rho*self.params.Cv)
        self.fields.prs[:, :] = self.fields.rho*self.params.Rspecific*self.fields.tmp
