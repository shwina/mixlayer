import numpy as np

class RK4:

    def __init__(self, fields, rhs_func, *rhs_func_extra_args):
        """

        Runge-Kutta time-stepper.

        Parameters
        ----------
        fields : list or tuple
            List or tuple of field vectors that will be advanced
            in a time step.

        rhs_func : function
            Function that computes the right-hand sides for all equations.
            `fields` is assumed to be the first argument to the function.
        
        rhs_func_extra_args : *tuple
            Any additional arguments to be passed to `rhs_func`.
        """
        self.fields = fields
        self.rhs_func = rhs_func
        self.rhs_func_extra_args = rhs_func_extra_args
        self._allocate_arrays()

    def step(self, dt):
        """

        Take a time-step of "dt" and update fields.

        Parameters
        ----------

        dt : float
            Time step length
        """
        
        for f, f0 in zip(self.fields, self.fields_0):
            f0[...] = f

        ki = [dt/6, dt/3, dt/3]
        hi = [dt/2, dt/2, dt]

        for h, k in zip(hi, ki):
            rhss = self.rhs_func(self.fields, *self.rhs_func_extra_args)
            for f, f0, f1, rhs in zip(self.fields, self.fields_0, self.fields_1, rhss):
                f[...] = f0 + h*rhs
                f1[...] = k*rhs

        h = (dt/6)

        rhss = self.rhs_func(self.fields, *self.rhs_func_extra_args)
        for f, f0, f1, rhs in zip(self.fields, self.fields_0, self.fields_1, rhss):
            f[...] = f0 + f1 + h*rhs

    def _allocate_arrays(self):
        """
        Allocate extra storage.
        """
        self.fields_0 = []
        self.fields_1 = []
        for field in self.fields:
            self.fields_0.append(np.copy(field))
            self.fields_1.append(np.copy(field))

