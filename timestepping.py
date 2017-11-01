
class RK4:

    def __init__(self, params, fields):
        self.p = params
        self.f = fields
        self.allocate_arrays()

    def allocate_arrays(self):
        self.rho_0 = self.f.rho.copy()
        self.rho_u_0 = self.f.rho_u.copy()
        self.rho_v_0 = self.f.rho_v.copy()
        self.egy_0 = self.f.egy.copy()

        self.rho_next = self.f.rho.copy()
        self.rho_u_next = self.f.rho_u.copy()
        self.rho_v_next = self.f.rho_v.copy()
        self.egy_next = self.f.egy.copy()

    def set_rhs_func(self, rhs_func, *rhs_func_args):
        self.rhs_func = rhs_func
        self.rhs_func_args = rhs_func_args

    def set_filter_func(self, filter_func, *filter_func_args):
        self.filter_func = filter_func
        self.filter_func_args = filter_func_args

    def step(self, dt):

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

            self.filter_func(self.p, self.f, *self.filter_func_args)

        h = (dt/6)

        self.rhs_func(self.p, self.f, *self.rhs_func_args)

        self.f.rho = self.rho_0 + self.rho_next + h*self.f.rho_rhs
        self.f.rho_u = self.rho_u_0 + self.rho_u_next + h*self.f.rho_u_rhs
        self.f.rho_v = self.rho_v_0 + self.rho_v_next + h*self.f.rho_v_rhs
        self.f.egy = self.egy_0 + self.egy_next + h*self.f.egy_rhs

        self.filter_func(self.p, self.f, *self.filter_func_args)
