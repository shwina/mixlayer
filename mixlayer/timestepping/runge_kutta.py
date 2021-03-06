"""
Time steppers for ODEs

Classes
-------

- `RK4` -- 4th order Runge-Kutta time-stepping

"""

import numpy as np

class RK4:
    """
    4th order Runge-Kutta time-stepping. Evolves the system of equations
        
        d(u_1)/dt = f1(...)
        d(u_2)/dt = f2(...)
        
    by one time step.

    Parameters
    ----------

    U : sequence of array_like ([u1, u2 ...])
        Initial values of u_1, u_2, etc.,
    
    rhs : sequence of array_like ([rhs1, rhs2, ...])
        Space for storing right-hand sides. Each array must be of the same
        shape as the corresponding array in `U`.

    rhs_func : array_like
        Function that computes the right-hand sides and stores them in RHS

    *rhs_func_args
        Inputs to rhs_func
    """

    def __init__(self, U, rhs, rhs_func, *rhs_func_args):
        self.U = U
        self.rhs = rhs
        self.rhs_func = rhs_func
        self.rhs_func_args = rhs_func_args
        self._allocate_arrays()

    def step(self, dt):
        """
        Take a time step of "dt" and update fields.

        Parameters
        ----------

        dt : float
            Time step length
        """

        for f, f0 in zip(self.U, self.U0):
            f0[...] = f

        ki = [dt/6, dt/3, dt/3]
        hi = [dt/2, dt/2, dt]

        for h, k in zip(hi, ki):

            # first compute RHS
            self.rhs_func(*self.rhs_func_args)

            # then update U, U1
            for u, u0, u1, r in zip(self.U, self.U0, self.U1, self.rhs):
                u[...] = u0 + h*r
                u1[...] = k*r

        h = (dt/6)

        self.rhs_func(*self.rhs_func_args)

        for u, u0, u1, r in zip(self.U, self.U0, self.U1, self.rhs):
            u[...] = u0 + u1 + h*r

    def _allocate_arrays(self):
        self.U0 = []
        self.U1 = []
        for u in self.U:
            self.U0.append(np.copy(u))
            self.U1.append(np.copy(u))
