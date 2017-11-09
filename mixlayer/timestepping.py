class RK4:

    def __init__(self, params, fields):
        """

        Runge-Kutta time-stepping with filtering.

        Parameters
        ----------
        params : attrdict.AttrDict
            Dictionary containing simulation parameters

        fields : attrdict.AttrDict
            Dictionary containing simulation field vectors

        """
        self.p = params
        self.f = fields
        self._allocate_arrays()

    def set_rhs_func(self, rhs_func, *rhs_func_args):
        """

        Set function for computing the right-hand sides of the equations.

        Parameters
        ----------
        rhs_func : function
            Function that computes right-hand sides. Takes
            `params` and `fields` as its first two arguments.

        *rhs_func_args (optional)
            Any additional arguments to `rhs_func`.

        """
        self.rhs_func = rhs_func
        self.rhs_func_args = rhs_func_args

    def set_filter_func(self, filter_func, *filter_func_args):
        """

        Set function for filtering the fields at each sub-step

        Parameters
        ----------
        filter_func : function
            Function that applies the filter to the fields. Takes
            `params` and `fields` as its first two arguments.

        *filter_func_args (optional)
             Any additional arguments to `filter_func`.

        """

        self.filter_func = filter_func
        self.filter_func_args = filter_func_args

    def step(self, dt):
        """

        Take a time-step of "dt" and update fields.

        Parameters
        ----------

        dt : float
            Time step length
        """

        self.rho_0[...] = self.f.rho
        self.rho_u_0[...] = self.f.rho_u
        self.rho_v_0[...] = self.f.rho_v
        self.egy_0[...] = self.f.egy

        ki = [dt/6, dt/3, dt/3]
        hi = [dt/2, dt/2, dt]
    
        for h, k in zip(hi, ki):

            self.rhs_func(self.p, self.f, *self.rhs_func_args)

            self.f.rho = self.rho_0 + h*self.f.rho_rhs
            self.f.rho_u = self.rho_u_0 + h*self.f.rho_u_rhs
            self.f.rho_v = self.rho_v_0 + h*self.f.rho_v_rhs
            self.f.egy = self.egy_0 + h*self.f.egy_rhs

            self.rho_next = k*self.f.rho_rhs
            self.rho_u_next = k*self.f.rho_u_rhs
            self.rho_v_next = k*self.f.rho_v_rhs
            self.egy_next = k*self.f.egy_rhs

            #self.filter_func(self.p, self.f, *self.filter_func_args)

        h = (dt/6)

        self.rhs_func(self.p, self.f, *self.rhs_func_args)

        self.f.rho = self.rho_0 + self.rho_next + h*self.f.rho_rhs
        self.f.rho_u = self.rho_u_0 + self.rho_u_next + h*self.f.rho_u_rhs
        self.f.rho_v = self.rho_v_0 + self.rho_v_next + h*self.f.rho_v_rhs
        self.f.egy = self.egy_0 + self.egy_next + h*self.f.egy_rhs

        self.filter_func(self.p, self.f, *self.filter_func_args)

    def _allocate_arrays(self):
        """
        Allocate extra storage.
        """
        self.rho_0 = self.f.rho.copy()
        self.rho_u_0 = self.f.rho_u.copy()
        self.rho_v_0 = self.f.rho_v.copy()
        self.egy_0 = self.f.egy.copy()

        self.rho_next = self.f.rho.copy()
        self.rho_u_next = self.f.rho_u.copy()
        self.rho_v_next = self.f.rho_v.copy()
        self.egy_next = self.f.egy.copy()

