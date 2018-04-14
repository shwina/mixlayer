"""
2-d compressible flow with one-step chemistry.
"""
import numpy as np

from mixlayer.constants import *
from mixlayer.filtering.explicit import filter5

class OneStepSolver:

    def __init__(
            self,
            operators,
            mixture, grid, fields,
            reaction, rratio,
            timestepping_scheme,
            Ma,
            P_inf,
            T_ref):
        
        self.operators = operators
        self.mixture = mixture
        self.grid = grid
        self.fields = fields
        self.reaction = reaction
        self.rratio = rratio
        self.timestepping_scheme = timestepping_scheme
        self.Ma = Ma
        self.P_inf = P_inf
        self.T_ref = T_ref
 
        # primary fields
        self.U = [fields['rho'],
                  fields['rho_u'],
                  fields['rho_v'],
                  fields['egy'],
                  fields['rho_y']]
        self.tmp = fields['tmp']
        self.prs = fields['prs']

        self.rhs = [np.zeros_like(u) for u in
            self.U]

        self.stepper = timestepping_scheme(self.U, self.rhs, self.compute_rhs)


    def compute_rhs(self):
        self.correct()

        dfdx, dfdy = self.operators.dfdx, self.operators.dfdy
        divergence, laplacian = self.operators.divergence, self.operators.laplacian
        rho, rho_u, rho_v, egy, rho_y = self.U
        rho_rhs, rho_u_rhs, rho_v_rhs, egy_rhs, rho_y_rhs = self.rhs
        tmp, prs = self.tmp, self.prs
        molwt_1 = self.mixture.species_list[0].molecular_weight
        molwt_2 = self.mixture.species_list[1].molecular_weight
        molwt_3 = self.mixture.species_list[2].molecular_weight

        mu = self.mixture.mu(prs, tmp)
        kappa = self.mixture.kappa(prs, tmp)

        activation_energy = self.reaction.activation_energy
        arrhenius_coefficient = self.reaction.arrhenius_coefficient
        rratio = self.rratio
        enthalpy_of_formation = self.mixture.species_list[2].enthalpy_of_formation
        reaction_rate = arrhenius_coefficient * rho * np.exp(
                -activation_energy/(universal_gas_constant*tmp))

        # euler and viscous terms:
        tau_11, tau_12, tau_22 = self.stress_tensor()

        rho_rhs[...] = -divergence(rho_u, rho_v)
        rho_u_rhs[...] = (-divergence(rho_u*rho_u/rho + prs, rho_u*rho_v/rho)
                          + dfdx(tau_11)
                          + dfdy(tau_12))
        rho_v_rhs[...] = (-divergence(rho_v*rho_u/rho, rho_v*rho_v/rho + prs)
                          + dfdx(tau_12)
                          + dfdy(tau_22))
        egy_rhs[...] = (-divergence((egy+prs) * rho_u/rho,
                                    (egy+prs) * rho_v/rho)
                        + dfdx(rho_u/rho * tau_11) + dfdx(rho_v/rho * tau_12)
                        + dfdy(rho_u/rho * tau_12) + dfdy(rho_v/rho * tau_22)
                        + kappa*(laplacian(tmp))
                        - enthalpy_of_formation*(1+rratio)*reaction_rate*(
                            rho_y[0]*rho_y[1]/(molwt_1*molwt_2)))

        # species equation convection and diffusion terms:
        for i, (yi, rho_yi_rhs) in enumerate(zip(self.mixture.Y, rho_y_rhs)):
            D = self.mixture.D(i, tmp, prs)
            rho_yi_rhs[...] = (
                - divergence(rho_u*yi, rho_v*yi)
                - rho*D*divergence(
                    yi,
                    yi))

        # species equation source terms:
        rho_y_rhs[0, ...] -= molwt_1 * reaction_rate*(rho_y[0]*rho_y[1]/(molwt_1*molwt_2))
        rho_y_rhs[1, ...] -= molwt_2 * rratio*reaction_rate*(rho_y[0]*rho_y[1]/(molwt_1*molwt_2))
        rho_y_rhs[2, ...] += molwt_3 * (1+rratio)*reaction_rate*(rho_y[0]*rho_y[1]/(molwt_1*molwt_2))

        self.non_reflecting_boundary_conditions()

    def correct(self):
        U = self.U
        mixture = self.mixture
        rho, rho_u, rho_v, egy, rho_y = self.U
        tmp = self.tmp
        prs = self.prs
        
        y1 = rho_y[0]/rho
        y2 = rho_y[1]/rho
        y3 = rho_y[2]/rho
        tmp[...] = (egy - 0.5*(rho_u**2 + rho_v**2)/rho) / (rho*mixture.Cv(prs, tmp))
        prs[...] = mixture.gas_model.P(rho, tmp)

        xy = 1 - y1 - y2 - y3
        rho_y[0][ y1 > 0.5 ] += (rho*xy) [ y1 > 0.5 ]
        rho_y[1][ y1 < 0.5 ] += (rho*xy) [ y1 < 0.5 ]

        # ensure that mass fractions are 0 <= y <= 1
        for rho_yi in rho_y:
            rho_yi [rho_yi < 0] = 0.
            rho_yi [rho_yi > rho] = rho [rho_yi > rho]

        mixture.Y = [rho_y[0]/rho, rho_y[1]/rho, rho_y[2]/rho]

    def timestep(self):
        rho, rho_u, rho_v, egy, rho_y = self.U
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

        dfdx, dfdy = self.operators.dfdx, self.operators.dfdy
        divergence, laplacian = self.operators.divergence, self.operators.laplacian
        Ly = self.grid.Ly
        mixture = self.mixture
        rho, rho_u, rho_v, egy, rho_y = self.U
        rho_rhs, rho_u_rhs, rho_v_rhs, egy_rhs, rho_y_rhs = self.rhs
        tmp = self.tmp
        prs = self.prs
        molwt_1 = self.mixture.species_list[0].molecular_weight
        molwt_2 = self.mixture.species_list[1].molecular_weight
        molwt_3 = self.mixture.species_list[2].molecular_weight

        mu = self.mixture.mu(prs, tmp)
        kappa = self.mixture.kappa(prs, tmp)

        activation_energy = self.reaction.activation_energy
        arrhenius_coefficient = self.reaction.arrhenius_coefficient
        rratio = self.rratio
        enthalpy_of_formation = self.mixture.species_list[2].enthalpy_of_formation
        reaction_rate = arrhenius_coefficient * rho * np.exp(
                -activation_energy/(universal_gas_constant*tmp))

        tau_11, tau_12, tau_22 = self.stress_tensor()

        C_sound = np.sqrt(mixture.Cp(prs, tmp)/mixture.Cv(prs, tmp) * mixture.R *tmp)

        dpdy = dfdy(prs)
        drhody = dfdy(rho)
        dudy = dfdy(rho_u/rho)
        dvdy = dfdy(rho_v/rho)
        dy1dy = dfdy(rho_y[0]/rho)
        dy2dy = dfdy(rho_y[1]/rho)
        dy3dy = dfdy(rho_y[2]/rho)
        
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

        rho_rhs[0, :] = (
            - d_1[0, :]
            - dfdx(np.atleast_2d(rho_u[0, :]))
        )
        rho_u_rhs[0, :] = (
            ( - rho_u/rho*d_1 - rho*d_3)[0, :]
            - dfdx(np.atleast_2d((rho_u*rho_u/rho + prs)[0, :]))
            + (dfdx(tau_11) + dfdy(tau_12))[0, :]
        )
        rho_v_rhs[0, :] = (
            ( - rho_v/rho*d_1 - rho*d_4)[0, :]
            - dfdx(np.atleast_2d((rho_v*rho_u/rho)[0, :]))
            + (dfdx(tau_12) + dfdy(tau_22))[0, :]
        )
        egy_rhs[0, :] = (
            (   - 0.5*np.sqrt((rho_u/rho)**2 + (rho_v/rho)**2)*d_1
                - d_2 * (prs + egy) / (rho*C_sound**2)
                - rho * (rho_v/rho * d_4 + rho_u/rho * d_3))[0, :]
            - dfdx(np.atleast_2d(((egy+prs) * rho_u/rho)[0, :]))
            + (dfdx(rho_u/rho * tau_11) + dfdx(rho_v/rho * tau_12) +
                    dfdy(rho_u/rho * tau_12) + dfdy(rho_v/rho * tau_22) + 
                    kappa*(laplacian(tmp)))[0, :]
            - (enthalpy_of_formation*(1+rratio)*reaction_rate*(
                rho_y[0]*rho_y[1]/(molwt_1*molwt_2)))[0, :]
        )

        for i, (rho_yi, rho_yi_rhs) in enumerate(zip(rho_y, rho_y_rhs)):
            D = self.mixture.D(i, tmp, prs)
            rho_yi_rhs[0, :] = (
                - dfdx(rho_u*rho_yi/rho)
                - rho*D*dfdx(rho_yi/rho))[0, :]

        # species equation source terms:
        rho_y_rhs[0, 0, :] -= molwt_1 * (reaction_rate*(rho_y[0]*rho_y[1]/(molwt_1*molwt_2)))[0, :]
        rho_y_rhs[1, 0, :] -= molwt_2 * (rratio*reaction_rate*(rho_y[0]*rho_y[1]/(molwt_1*molwt_2)))[0, :]
        rho_y_rhs[2, 0, :] += molwt_3 * ((1+rratio)*reaction_rate*(rho_y[0]*rho_y[1]/(molwt_1*molwt_2)))[0, :]

        rho_y_rhs[0, 0, :] = (rho_y_rhs[0] - rho_y[0]/rho*d_1 - rho*d_5)[0, :]
        rho_y_rhs[1, 0, :] = (rho_y_rhs[1] - rho_y[1]/rho*d_1 - rho*d_6)[0, :]
        rho_y_rhs[2, 0, :] = (rho_y_rhs[2] - rho_y[2]/rho*d_1 - rho*d_7)[0, :]

        L_1 = 0.4 * (1 - self.Ma**2) * C_sound/Ly * (prs - self.P_inf)
        L_2 = rho_v/rho * (C_sound**2 * drhody - dpdy)
        L_3 = rho_v/rho * dudy
        L_4 = (rho_v/rho + C_sound) * (dpdy + rho*C_sound*dvdy)

        d_1 = (1./C_sound**2) * (L_2 + 0.5*(L_4 + L_1))
        d_2 = 0.5*(L_1 + L_4)
        d_3 = L_3
        d_4 = 1/(2*rho*C_sound) * (L_4 - L_1)

        rho_rhs[-1, :] = (
            - d_1[-1, :]
            - dfdx(np.atleast_2d(rho_u[-1, :]))
        )

        rho_u_rhs[-1, :] = (
            ( - rho_u/rho*d_1 - rho*d_3)[-1, :]
            - dfdx(np.atleast_2d((rho_u*rho_u/rho + prs)[-1, :]))
            + (dfdx(tau_11) + dfdy(tau_12))[-1, :]
        )

        rho_v_rhs[-1, :] = (
            ( - rho_v/rho*d_1 - rho*d_4)[-1, :]
            - dfdx(np.atleast_2d((rho_v*rho_u/rho)[-1, :]))
            + (dfdx(tau_12) + dfdy(tau_22))[-1, :]
        )

        egy_rhs[-1, :] = (
            (   - 0.5*np.sqrt((rho_u/rho)**2 + (rho_v/rho)**2)*d_1 -
                d_2 * (prs + egy) / (rho*C_sound**2) -
                rho * (rho_v/rho * d_4 + rho_u/rho * d_3))[-1, :]
            - dfdx(np.atleast_2d(((egy+prs) * rho_u/rho)[-1, :]))
            + (dfdx(rho_u/rho * tau_11) + dfdx(rho_v/rho * tau_12) +
                    dfdy(rho_u/rho * tau_12) + dfdy(rho_v/rho * tau_22) + 
                    kappa*(laplacian(tmp)))[-1, :]
            - (enthalpy_of_formation*(1+rratio)*reaction_rate*(
                rho_y[0]*rho_y[1]/(molwt_1*molwt_2)))[-1, :]
        )

        for i, (rho_yi, rho_yi_rhs) in enumerate(zip(rho_y, rho_y_rhs)):
            D = self.mixture.D(i, tmp, prs)
            rho_yi_rhs[-1, :] = (
                - dfdx(rho_u*rho_yi/rho)
                - rho*D*dfdx(rho_yi/rho))[-1, :]

        # species equation source terms:
        rho_y_rhs[0, -1, :] -= molwt_1 * (reaction_rate*(rho_y[0]*rho_y[1]/(molwt_1*molwt_2)))[-1, :]
        rho_y_rhs[1, -1, :] -= molwt_2 * (rratio*reaction_rate*(rho_y[0]*rho_y[1]/(molwt_1*molwt_2)))[-1, :]
        rho_y_rhs[2, -1, :] += molwt_3 * ((1+rratio)*reaction_rate*(rho_y[0]*rho_y[1]/(molwt_1*molwt_2)))[-1, :]

        rho_y_rhs[0, -1, :] = (rho_y_rhs[0] - rho_y[0]/rho*d_1 - rho*d_5)[-1, :]
        rho_y_rhs[1, -1, :] = (rho_y_rhs[1] - rho_y[1]/rho*d_1 - rho*d_6)[-1, :]
        rho_y_rhs[2, -1, :] = (rho_y_rhs[2] - rho_y[2]/rho*d_1 - rho*d_7)[-1, :]
        
    def stress_tensor(self):
        dfdx, dfdy = self.operators.dfdx, self.operators.dfdy
        mixture = self.mixture
        rho, rho_u, rho_v, egy, rho_y = self.U
        tmp = self.tmp
        prs = self.prs
        mu = self.mixture.mu(prs, tmp)
        kappa = self.mixture.kappa(prs, tmp)

        div_vel = dfdx(rho_u/rho) + dfdy(rho_v/rho)

        tau_11 = -(2./3)*mu*div_vel + 2*mu*dfdx(rho_u/rho) 
        tau_22 = -(2./3)*mu*div_vel + 2*mu*dfdy(rho_v/rho)
        tau_12 = mu*(dfdx(rho_v/rho) + dfdy(rho_u/rho))
        
        tau_12[0, :] = ( 18.*tau_12[ 1, :] - 9*tau_12[ 2, :] + 2*tau_12[ 3, :]) / 11
        tau_12[-1, :] =( 18.*tau_12[-2, :] - 9*tau_12[-3, :] + 2*tau_12[-4, :]) / 11

        return tau_11, tau_12, tau_22

    def filter(self):
        # all other equations:
        for f in self.U[:-1]:
            filter5(f)

        # species equations:
        for f in self.U[-1]:
            filter5(f)

    def step(self):
        dt = self.timestep()
        self.stepper.step(dt)

        rho = self.U[0]

        # apply filter
        self.filter()

        # ensure that mass fractions are 0 <= y <= 1
        for rho_yi in self.U[-1]:
            rho_yi [rho_yi < 0] = 0.
            rho_yi [rho_yi > rho] = rho [rho_yi > rho]
