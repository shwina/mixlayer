"""
Time steppers for ODEs

Classes
-------

- `RK4` -- 4th order Runge-Kutta time-stepping
"""

import numpy as np

class RK4:
    """
    4th order Runge-Kutta time-stepping.

    Parameters
    ----------

    """

    def __init__(self, U, rhs, rhs_func, *rhs_func_args):
        self.U = U
        self.rhs = rhs
        self.rhs_func = rhs_func
        self.rhs_func_args = rhs_func_args
        self._allocate_arrays()

    def step(self, dt):
        """
        Take a time-step of "dt" and update fields.

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
            self.U[...] = self.U0 + h*self.rhs
            self.U1[...] = k*self.rhs

        h = (dt/6)

        self.rhs_func(*self.rhs_func_args)

        self.U[...] = self.U0 + self.U1 + h*self.rhs

    def _allocate_arrays(self):
        """
        Allocate extra storage for field vectors
        """
        self.U0 = np.copy(self.U)
        self.U1 = np.copy(self.U)
