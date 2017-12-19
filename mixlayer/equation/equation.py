import numpy as np

class Equation(object):
    
    def __init__(self, f):
        self.f = f
        self._allocate_arrays()
        self.rhs_pre_func = None
        self.rhs_post_func = None

    def set_rhs_func(self, rhs_func, *args):
        """
        Must support a kwarg 'out' for storing the
        right-hand side
        """
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
