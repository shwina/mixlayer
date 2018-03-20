"""
2-d compressible flow with one-step chemistry.
"""
import numpy as np

from mixlayer.constants import *
from mixlayer.filtering.explicit import filter5

class OneStepSolver:

    def __init__(self, mixture, grid, U, rhs, tmp, prs,
            arrhenius_coefficient, activation_energy, rratio, enthalpy_of_formation,
            molwt_1, molwt_2, molwt_3,
            timestepping_scheme):

        self.mixture = mixture
        self.grid = grid
        self.U = U
        self.rhs = rhs
        self.tmp = tmp
        self.prs = prs
        self.arrhenius_coefficient = arrhenius_coefficient
        self.activation_energy = activation_energy
        self.rratio = rratio
        self.enthalpy_of_formation = enthalpy_of_formation
        self.molwt_1 = molwt_1
        self.molwt_2 = molwt_2
        self.molwt_3 = molwt_3
        self.timestepping_scheme = timestepping_scheme
 
        self.rhs_pre_func = None
        self.rhs_post_func = None

        self.stepper = timestepping_scheme(U, rhs, self.compute_rhs)

    def compute_rhs(self):

        if self.rhs_pre_func : self.rhs_pre_func(*self.rhs_pre_func_args)

        dfdx, dfdy = self.grid.dfdx, self.grid.dfdy
        rho, rho_u, rho_v, egy, rho_y1, rho_y2, rho_y3 = self.U
        rho_rhs, rho_u_rhs, rho_v_rhs, egy_rhs, rho_y1_rhs, rho_y2_rhs, rho_y3_rhs = self.rhs
        tmp, prs = self.tmp, self.prs
        arrhenius_coefficient = self.arrhenius_coefficient
        activation_energy = self.activation_energy
        rratio = self.rratio
        enthalpy_of_formation = self.enthalpy_of_formation
        molwt_1 = self.molwt_1
        molwt_2 = self.molwt_2
        molwt_3 = self.molwt_3

        mu = self.mixture.mu(prs, tmp)
        kappa = self.mixture.kappa(prs, tmp)

        # euler terms:
        rho_rhs[...] = -dfdy(rho_v)
        rho_rhs[[0,-1], :] = 0
        rho_rhs[...] += -dfdx(rho_u)

        rho_u_rhs[...] = -dfdy(rho_u*rho_v/rho)
        rho_u_rhs[[0,-1], :] = 0
        rho_u_rhs += -dfdx(rho_u*rho_u/rho + prs) 
        
        rho_v_rhs[...] = -dfdy(rho_v*rho_v/rho + prs)
        rho_v_rhs[[0,-1], :] = 0 
        rho_v_rhs += -dfdx(rho_v*rho_u/rho )
        
        egy_rhs_x = -dfdx((egy+prs) * (rho_u/rho))
        egy_rhs_y = -dfdy((egy+prs) * (rho_v/rho))
        egy_rhs_y[[0,-1], :] = 0

        egy_rhs[...] = egy_rhs_x + egy_rhs_y

        # viscous terms:
        div_vel = dfdx(rho_u/rho) + dfdy(rho_v/rho)

        tau_11 = -(2./3)*mu*div_vel + 2*mu*dfdx(rho_u/rho) 
        tau_22 = -(2./3)*mu*div_vel + 2*mu*dfdy(rho_v/rho)
        tau_12 = mu*(dfdx(rho_v/rho) + dfdy(rho_u/rho))
        
        tau_12[0, :] = ( 18.*tau_12[ 1, :] - 9*tau_12[ 2, :] + 2*tau_12[ 3, :]) / 11
        tau_12[-1, :] =( 18.*tau_12[-2, :] - 9*tau_12[-3, :] + 2*tau_12[-4, :]) / 11

        rho_u_rhs += dfdx(tau_11) + dfdy(tau_12)
        rho_v_rhs += dfdx(tau_12) + dfdy(tau_22)
        egy_rhs += (dfdx(rho_u/rho * tau_11) + dfdx(rho_v/rho * tau_12) +
                    dfdy(rho_u/rho * tau_12) + dfdy(rho_v/rho * tau_22) + 
                    kappa*(
                        dfdx(dfdx(tmp)) +
                        dfdy(dfdy(tmp))))

        # species equation convection and diffusion terms:
        for i, (rho_yi, rho_yi_rhs) in enumerate(zip(self.U[4:], self.rhs[4:])):
            D = self.mixture.D(i, tmp, prs)
            rho_yi_rhs_x = dfdx(-rho_u*rho_yi/rho +
                    rho*D*dfdx(-rho_yi/rho))
            rho_yi_rhs_y = dfdy(-rho_v*rho_yi/rho +
                    rho*D*dfdy(-rho_yi/rho))
            rho_yi_rhs_y[[0, -1], :] = 0
            rho_yi_rhs[...] = rho_yi_rhs_x + rho_yi_rhs_y

        # species equation source terms:
        reaction_rate = arrhenius_coefficient * rho * np.exp(
                -activation_energy/(universal_gas_constant*tmp))
        rho_y1_rhs[...] -= molwt_1 * reaction_rate*(rho_y1*rho_y2/(molwt_1*molwt_2))
        rho_y2_rhs[...] -= molwt_2 * rratio*reaction_rate*(rho_y1*rho_y2/(molwt_1*molwt_2))
        rho_y3_rhs[...] += molwt_3 * (1+rratio)*reaction_rate*(rho_y1*rho_y2/(molwt_1*molwt_2))

        # energy equation source term:
        egy_rhs[...] -= enthalpy_of_formation*(1+rratio)*reaction_rate*(rho_y1*rho_y2/(molwt_1*molwt_2))

        if self.rhs_post_func : self.rhs_post_func(*self.rhs_post_func_args)

    def set_rhs_pre_func(self, func, *args):
        self.rhs_pre_func = func
        self.rhs_pre_func_args = args

    def set_rhs_post_func(self, func, *args):
        self.rhs_post_func = func
        self.rhs_post_func_args = args

    def step(self, dt):
        self.stepper.step(dt)

        rho = self.U[0]

        # apply filter
        for f in self.U:
            filter5(f)

        # ensure that mass fractions are 0 <= y <= 1
        for rho_yi in self.U[4:]:
            rho_yi [rho_yi < 0] = 0.
            rho_yi [rho_yi > rho] = rho [rho_yi > rho]
