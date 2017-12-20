import numpy as np

class Equation(object):
    """
    Class for storing the ODE df/dt = rhs

    Attributes
    ----------

    f : array_like
        Field that is evolved in time

    rhs : array_like
        Right-hand side

    rhs_func : function
        Function for computing the right-hand side

    rhs_pre_func : function
        Function that must execute before rhs is computed.

    rhs_post_func : function
        Function that must execute after rhs is computed
    """

    def __init__(self, f):
        self.f = f
        self.rhs_pre_func = None
        self.rhs_post_func = None
        self._allocate_arrays()

    def set_rhs_func(self, rhs_func, *args):
        self.rhs_func = rhs_func
        self.rhs_func_args = args

    def set_rhs_pre_func(self, rhs_pre_func, *args):
        self.rhs_pre_func = rhs_pre_func
        self.rhs_pre_func_args = args

    def set_rhs_post_func(self, rhs_post_func, *args):
        self.rhs_post_func = rhs_post_func
        self.rhs_post_func_args = args

    def compute_rhs(self):
        if self.rhs_pre_func:
            self.rhs_pre_func(*self.rhs_pre_func_args)

        self.rhs_func(*self.rhs_func_args, out=self.rhs)

        if self.rhs_post_func:
            self.rhs_post_func(*self.rhs_post_func_args)

    def _allocate_arrays(self):
        self.rhs = np.zeros_like(self.f)
