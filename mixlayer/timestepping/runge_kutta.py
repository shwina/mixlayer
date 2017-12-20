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

    equations : Equation
        List of Equation objects representing a system of ODEs that will
        be evolved in a time step
    """

    def __init__(self, equations):
        self.equations = equations
        self._allocate_arrays()

    def step(self, dt):
        """
        Take a time-step of "dt" and update fields.

        Parameters
        ----------

        dt : float
            Time step length
        """

        for eq, f0_ in zip(self.equations, self.f0):
            f = eq.f
            f0_[...] = f

        ki = [dt/6, dt/3, dt/3]
        hi = [dt/2, dt/2, dt]

        for h, k in zip(hi, ki):

            # first compute all right hand sides
            for eq in self.equations:
                eq.compute_rhs()

            # then update f, f1
            for eq, f0_, f1_ in zip(self.equations, self.f0, self.f1):
                f = eq.f
                rhs = eq.rhs
                f[...] = f0_ + h*rhs
                f1_[...] = k*rhs

        h = (dt/6)

        for eq in self.equations:
            eq.compute_rhs()

        for eq, f0_, f1_ in zip(self.equations, self.f0, self.f1):
            f = eq.f
            rhs = eq.rhs
            f[...] = f0_ + f1_ + h*rhs

    def _allocate_arrays(self):
        """
        Allocate extra storage for field vectors
        """
        self.f0 = []
        self.f1 = []
        for eq in self.equations:
            self.f0.append(np.copy(eq.f))
            self.f1.append(np.copy(eq.f))
