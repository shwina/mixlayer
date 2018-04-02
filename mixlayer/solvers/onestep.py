"""
2-d compressible flow with one-step chemistry.
"""
import numpy as np

from mixlayer.constants import *
from mixlayer.filtering.explicit import filter5

class OneStepSolver:

    def __init__(self, mixture, grid, U, tmp, prs,
            reaction, rratio,
            timestepping_scheme,
            Ma,
            P_inf,
            T_ref):

        self.mixture = mixture
        self.grid = grid
        self.U = U
        self.rhs = np.zeros_like(self.U)
        self.tmp = tmp
        self.prs = prs
        self.arrhenius_coefficient = reaction.arrhenius_coefficient
        self.activation_energy = reaction.activation_energy
        self.rratio = rratio
        self.molwt_1 = mixture.species_list[0].molecular_weight
        self.molwt_2 = mixture.species_list[1].molecular_weight 
        self.molwt_3 = mixture.species_list[2].molecular_weight 
        self.enthalpy_of_formation = mixture.species_list[2].enthalpy_of_formation
        self.timestepping_scheme = timestepping_scheme
        self.Ma = Ma
        self.P_inf = P_inf
        self.T_ref = T_ref
 
        self.stepper = timestepping_scheme(self.U, self.rhs, self.compute_rhs)

    def compute_rhs(self):

        self.correct()

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

        self.non_reflecting_boundary_conditions()

    def correct(self):
        U = self.U
        mixture = self.mixture
        rho, rho_u, rho_v, egy, rho_y1, rho_y2, rho_y3 = self.U
        tmp = self.tmp
        prs = self.prs
        
        y1 = U[4]/U[0]
        y2 = U[5]/U[0]
        y3 = U[6]/U[0]
        tmp[...] = (egy - 0.5*(rho_u**2 + rho_v**2)/rho) / (rho*mixture.Cv(prs, tmp))
        prs[...] = mixture.gas_model.P(rho, tmp)

        xy = 1 - y1 - y2 - y3
        U[4][ y1 > 0.5 ] += (rho*xy) [ y1 > 0.5 ]
        U[5][ y1 < 0.5 ] += (rho*xy) [ y1 < 0.5 ]

        # ensure that mass fractions are 0 <= y <= 1
        for rho_yi in U[4:]:
            rho_yi [rho_yi < 0] = 0.
            rho_yi [rho_yi > rho] = rho [rho_yi > rho]

        mixture.Y = [rho_y1/rho, rho_y2/rho, rho_y3/rho]

    def timestep(self):
        rho, rho_u, rho_v, egy, rho_y1, rho_y2, rho_y3 = self.U
        cfl_vel = 0.5
        cfl_visc = 0.1
        dxmin = np.minimum(self.grid.dx, self.grid.dy)

        alpha_1 = self.mixture.D(0, self.P_inf, self.T_ref)
        alpha_2 = self.mixture.mu(self.P_inf, self.T_ref)/rho
        alpha_3 = self.mixture.kappa(self.P_inf, self.T_ref)/(self.mixture.Cp(self.P_inf, self.T_ref)*rho)
        alpha_max = np.maximum(np.maximum(alpha_1, alpha_2), alpha_3)

        C_sound = np.sqrt(self.mixture.Cp(self.P_inf, self.T_ref)/self.mixture.Cv(self.P_inf, self.T_ref)*self.mixture.R*self.tmp)
        test_1 = cfl_vel*self.grid.dx/(C_sound + abs(rho_u/rho))
        test_2 = cfl_vel*self.grid.dy/(C_sound + abs(rho_v/rho))
        test_3 = cfl_visc*(dxmin**2)/alpha_max

        dt = np.min(np.minimum(np.minimum(test_1, test_2), test_3))
        return dt

    def non_reflecting_boundary_conditions(self):

        dfdx, dfdy = self.grid.dfdx, self.grid.dfdy
        Ly = self.grid.Ly
        mixture = self.mixture
        rho, rho_u, rho_v, egy, rho_y1, rho_y2, rho_y3 = self.U
        rho_rhs, rho_u_rhs, rho_v_rhs, egy_rhs, rho_y1_rhs, rho_y2_rhs, rho_y3_rhs = self.rhs
        tmp = self.tmp
        prs = self.prs
 
        C_sound = np.sqrt(mixture.Cp(prs, tmp)/mixture.Cv(prs, tmp) * mixture.R *tmp)

        dpdy = dfdy(prs)
        drhody = dfdy(rho)
        dudy = dfdy(rho_u/rho)
        dvdy = dfdy(rho_v/rho)
        dy1dy = dfdy(rho_y1/rho)
        dy2dy = dfdy(rho_y2/rho)
        dy3dy = dfdy(rho_y3/rho)
        
        L_1 = (rho_v/rho - C_sound) * (dpdy - rho*C_sound*dvdy)
        L_2 = rho_v/rho * (C_sound**2 * drhody - dpdy)
        L_3 = rho_v/rho * dudy
        L_4 = 0.4*(1 - self.Ma**2) * C_sound/Ly * (prs - self.P_inf)
        L_5 = (rho_v/rho)*dy1dy
        L_6 = (rho_v/rho)*dy2dy
        L_7 = (rho_v/rho)*dy3dy

        d_1 = (1. / C_sound**2) * (L_2 + 0.5*(L_4 + L_1))
        d_2 = 0.5*(L_1 + L_4)
        d_3 = L_3
        d_4 = 1./(2*rho*C_sound) * (L_4 - L_1)
        d_5 = L_5
        d_6 = L_6
        d_7 = L_7

        rho_rhs[0, :] = (rho_rhs - d_1)[0, :]
        rho_u_rhs[0, :] = (rho_u_rhs - rho_u/rho*d_1 - rho*d_3)[0, :]
        rho_v_rhs[0, :] = (rho_v_rhs - rho_v/rho*d_1 - rho*d_4)[0, :]
        egy_rhs[0, :] = (egy_rhs -
            0.5*np.sqrt((rho_u/rho)**2 + (rho_v/rho)**2)*d_1 -
            d_2 * (prs + egy) / (rho*C_sound**2) -
            rho * (rho_v/rho * d_4 + rho_u/rho * d_3))[0, :]
        rho_y1_rhs[0, :] = (rho_y1_rhs - rho_y1/rho*d_1 - rho*d_5)[0, :]
        rho_y2_rhs[0, :] = (rho_y2_rhs - rho_y2/rho*d_1 - rho*d_6)[0, :]
        rho_y3_rhs[0, :] = (rho_y3_rhs - rho_y3/rho*d_1 - rho*d_7)[0, :]

        L_1 = 0.4 * (1 - self.Ma**2) * C_sound/Ly * (prs - self.P_inf)
        L_2 = rho_v/rho * (C_sound**2 * drhody - dpdy)
        L_3 = rho_v/rho * dudy
        L_4 = (rho_v/rho + C_sound) * (dpdy + rho*C_sound*dvdy)

        d_1 = (1./C_sound**2) * (L_2 + 0.5*(L_4 + L_1))
        d_2 = 0.5*(L_1 + L_4)
        d_3 = L_3
        d_4 = 1/(2*rho*C_sound) * (L_4 - L_1)

        rho_rhs[-1, :] = (rho_rhs - d_1)[-1, :]
        rho_u_rhs[-1, :] = (rho_u_rhs - rho_u/rho*d_1 - rho*d_3)[-1, :]
        rho_v_rhs[-1, :] = (rho_v_rhs - rho_v/rho*d_1 - rho*d_4)[-1, :]
        egy_rhs[-1, :] = (egy_rhs-
            0.5*np.sqrt((rho_u/rho)**2 + (rho_v/rho)**2)*d_1 -
            d_2 * (prs + egy) / (rho*C_sound**2) -
            rho * (rho_v/rho * d_4 + rho_u/rho * d_3))[-1, :]
        rho_y1_rhs[-1, :] = (rho_y1_rhs - rho_y1/rho*d_1 - rho*d_5)[-1, :]
        rho_y2_rhs[-1, :] = (rho_y2_rhs - rho_y2/rho*d_1 - rho*d_6)[-1, :]
        rho_y3_rhs[-1, :] = (rho_y3_rhs - rho_y3/rho*d_1 - rho*d_7)[-1, :]

    def step(self):
        dt = self.timestep()
        self.stepper.step(dt)

        rho = self.U[0]

        # apply filter
        for f in self.U:
            filter5(f)

        # ensure that mass fractions are 0 <= y <= 1
        for rho_yi in self.U[4:]:
            rho_yi [rho_yi < 0] = 0.
            rho_yi [rho_yi > rho] = rho [rho_yi > rho]
