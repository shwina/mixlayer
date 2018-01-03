import numpy as np

class NoChemistrySolver:

    def __init__(self, grid, U, rhs, tmp, prs, mu, kappa, timestepping_scheme):
        self.grid = grid
        self.U = U
        self.rhs = rhs
        self.tmp = tmp
        self.prs = prs
        self.mu = mu
        self.kappa = kappa
        self.timestepping_scheme = timestepping_scheme
 
        self.rhs_pre_func = None
        self.rhs_post_func = None

        self.stepper = timestepping_scheme(U, rhs, self.compute_rhs)

    def compute_rhs(self):

        if self.rhs_pre_func : self.rhs_pre_func(*self.rhs_pre_func_args)

        dfdx, dfdy = self.grid.dfdx, self.grid.dfdy
        rho, rho_u, rho_v, egy = self.U
        rho_rhs, rho_u_rhs, rho_v_rhs, egy_rhs = self.rhs
        tmp, prs = self.tmp, self.prs
        mu, kappa = self.mu, self.kappa

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

        if self.rhs_post_func : self.rhs_post_func(*self.rhs_post_func_args)

    def set_rhs_pre_func(self, func, *args):
        self.rhs_pre_func = func
        self.rhs_pre_func_args = args

    def set_rhs_post_func(self, func, *args):
        self.rhs_post_func = func
        self.rhs_post_func_args = args

    def step(self, dt):
        self.stepper.step(dt)
