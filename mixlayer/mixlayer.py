import numpy as np
from numba import jit, float64, prange
import h5py

import sys

@jit(nopython=True, nogil=True)
def jacobi_step(f, dx, dn, rhs, dndy, d2ndy2):
    f_old = f.copy()
    denominator = 2./dx**2 + (2./dn**2)*dndy**2

    ny, nx = f.shape

    f[0, :] = 0
    f[-1, :] = 0
    f[:, 0] = f[:, -2]
    f[:, -1] = f[:, 1]

    for i in range(1, nx-1):
        for j in range(1, nx-1):
            fnew = (-rhs[i-1, j-1] +
                    (f[i,j+1] + f[i,j-1])/dx**2 +
                    (f[i+1,j] - f[i-1,j])/(2*dn)*d2ndy2[i-1,j-1] +
                    (f[i+1,j] + f[i-1,j])/(dn**2)*dndy[i-1,j-1]**2)/denominator[i-1,j-1]
            f[i,j] = f[i,j] + 1.6*(fnew - f[i,j])
    return np.linalg.norm(f-f_old)

def add_forcing(params, fields, x, y, dndy, d2ndy2):
    N = params.N
    dn = params.dn
    stream = fields.stream
    vort = fields.vort

    U_ref = params.U_ref
    Famp_2d = params.Famp_2d
    vorticity_thickness = params.vorticity_thickness
    disturbance_wavelength = params.disturbance_wavelength
    nperiod = params.nperiod

    dx = 1./N
    dy = x.copy()
    dy[:-1, :] = y[1:, :] - y[:-1, :]
    dy[-1, :] = y[-1, :] - y[-2, :]

    fx = np.zeros_like(x, dtype=np.float64)
    fy = np.zeros_like(x, dtype=np.float64)
    fx_max = 0

    vort = np.zeros_like(x, dtype=np.float64)
    stream = np.zeros([N+2, N+2], dtype=np.float64)

    amplitudes = [1, 0.5, 0.35, 0.35]
    for i in range(4):
        fx += amplitudes[i]*np.abs(np.sin(np.pi*x/(2**i*disturbance_wavelength)))
        fx_max = np.max([np.max(fx), fx_max])
    
    fx = fx/fx_max
    fy = np.exp(-np.pi*y**2/vorticity_thickness**2)
    
    vort[...] = fx*fy
    circ = np.sum(dy*dx*vort)

    vort[...] = (vort*Famp_2d*disturbance_wavelength*U_ref) / (circ/nperiod)

    for i in range(50000):
        err = jacobi_step(stream, dx, dn, -vort, dndy, d2ndy2)
        if err <= 1e-5:
            print("Jacobi steps: {}".format(i))
            break

    u_pert = (stream[2:,1:-1] - stream[0:-2,1:-1])*dndy/(2*dn)
    v_pert = -(stream[1:-1,2:] - stream[1:-1,0:-2])/(2*dx)
       
    vort[...] = ((stream[1:-1, 2:] - 2*stream[1:-1, 1:-1] + stream[1:-1, 0:-2])/(dx**2) +
            ((stream[2:, 1:-1] - stream[0:-2, 1:-1])*d2ndy2/(2*dn)) +
            ((stream[2:, 1:-1] - 2*stream[1:-1, 1:-1] + stream[0:-2, 1:-1])/(dn**2))*dndy**2)

    circ = np.sum(dy*dx*vort)

    u_pert = u_pert*Famp_2d*disturbance_wavelength*U_ref / (circ/nperiod)
    v_pert = v_pert*Famp_2d*disturbance_wavelength*U_ref / (circ/nperiod)

    return u_pert, v_pert

def eos(params, fields):

    rho = fields.rho
    rho_u = fields.rho_u
    rho_v = fields.rho_v
    egy = fields.egy
    tmp = fields.tmp
    prs = fields.prs

    Rspecific = params.Rspecific
    Cv = params.Cv

    tmp[:, :] = (egy - 0.5*(rho_u**2 + rho_v**2)/rho)/(rho*Cv)
    prs[:, :] = rho*Rspecific*tmp

def calculate_timestep(params, fields, x, y):

    eos(params, fields)
 
    rho = fields.rho
    rho_u = fields.rho_u
    rho_v = fields.rho_v
    egy = fields.egy
    tmp = fields.tmp
    prs = fields.prs

    gamma_ref = params.gamma_ref
    mu_ref = params.mu_ref
    kappa_ref = params.kappa_ref
    Cp = params.Cp
    Cv = params.Cv
    Rspecific = params.Rspecific
    cfl_vel = params.cfl_vel
    cfl_visc = params.cfl_visc

    # dx, dy
    N = np.shape(x)[0]
    dx = 1./N
    dy = x.copy()
    dy[:-1, :] = y[1:, :] - y[:-1, :]
    dy[-1, :] = y[-1, :] - y[-2, :]

    dxmin = np.minimum(dx, dy)

    # calculate diffusivities:
    alpha_1 = gamma_ref
    alpha_2 = mu_ref/rho
    alpha_3 = kappa_ref/(Cp*rho)
    alpha_max = np.maximum(np.maximum(alpha_1, alpha_2), alpha_3)

    # calculate C_sound
    C_sound = np.sqrt(Cp/Cv*Rspecific*tmp)
    test_1 = cfl_vel*dx/(C_sound + abs(rho_u/rho))
    test_2 = cfl_vel*dy/(C_sound + abs(rho_v/rho))
    test_3 = cfl_visc*(dxmin**2)/alpha_max

    dt = np.min(np.minimum(np.minimum(test_1, test_2), test_3))
    return dt

def rhs_euler_terms(params, fields, dndy):
 
    rho = fields.rho
    rho_u = fields.rho_u
    rho_v = fields.rho_v
    egy = fields.egy
    tmp = fields.tmp
    prs = fields.prs
    rho_rhs = fields.rho_rhs
    rho_u_rhs = fields.rho_u_rhs
    rho_v_rhs = fields.rho_v_rhs
    egy_rhs = fields.egy_rhs

    dx = params.dx
    dn = params.dn

    rho_rhs[...] = -dfdy(rho_v, dn)*dndy
    rho_rhs[[0,-1], :] = 0
    rho_rhs[...] += -dfdx(rho_u, dx)

    rho_u_rhs[...] = -dfdy(rho_u*rho_v/rho, dn)*dndy
    rho_u_rhs[[0,-1], :] = 0
    rho_u_rhs += -dfdx(rho_u*rho_u/rho + prs, dx) 
    
    rho_v_rhs[...] = -dfdy(rho_v*rho_v/rho + prs, dn)*dndy
    rho_v_rhs[[0,-1], :] = 0 
    rho_v_rhs += -dfdx(rho_v*rho_u/rho , dx)
    
    egy_rhs_x = -dfdx((egy + prs)*(rho_u/rho), dx)
    egy_rhs_y = -dfdy((egy + prs)*(rho_v/rho), dn)*dndy
    
    egy_rhs_y[[0,-1], :] = 0
    egy_rhs[...] = egy_rhs_x + egy_rhs_y

def rhs_viscous_terms(params, fields, dndy):

    rho = fields.rho
    rho_u = fields.rho_u
    rho_v = fields.rho_v
    egy = fields.egy
    tmp = fields.tmp
    prs = fields.prs
    rho_rhs = fields.rho_rhs
    rho_u_rhs = fields.rho_u_rhs
    rho_v_rhs = fields.rho_v_rhs
    egy_rhs = fields.egy_rhs
    
    dx = params.dx
    dn = params.dn
    mu = params.mu_ref
    kappa = params.kappa_ref

    div_vel = dfdx(rho_u/rho, dx) + dfdy(rho_v/rho, dn)*dndy
    tau_11 = -(2./3)*mu*div_vel + 2*mu*dfdx(rho_u/rho, dx) 
    tau_12 = mu*(dfdx(rho_v/rho, dx) + dfdy(rho_u/rho, dn)*dndy)
    tau_22 = -(2./3)*mu*div_vel + 2*mu*dfdy(rho_v/rho, dn)*dndy
    
    tau_12[0, :] = (18.*tau_12[1, :] - 9*tau_12[2, :] + 2*tau_12[3, :])/11.
    tau_12[-1, :] = (18.*tau_12[-2, :] - 9*tau_12[-3, :] + 2*tau_12[-4, :])/11.

    rho_u_rhs += (dfdx(tau_11, dx) + dfdy(tau_12,dn)*dndy)
    rho_v_rhs += (dfdx(tau_12, dx) + dfdy(tau_22,dn)*dndy)
    egy_rhs += (dfdx(rho_u/rho*tau_11, dx) + dfdx(rho_v/rho*tau_12, dx) + dfdy(rho_u/rho*tau_12, dn)*dndy + dfdy(rho_v/rho*tau_22, dn)*dndy) + kappa*(dfdx(dfdx(tmp, dx), dx) + dfdy(dfdy(tmp, dn)*dndy, dn)*dndy)

def apply_boundary_filter(params, f):
    f[0, :] = f[0, :] - (f[0, :] - 5*f[1, :]
                + 10*f[2, :] - 10*f[3, :] + 5*f[4, :]
                - f[5, :])*params.filter_amplitude

    f[-1, :] = f[-1, :] - (f[-1, :] - 5*f[-2, :]
                + 10*f[-3, :] - 10*f[-4, :] + 5*f[-5, :]
                - f[-6, :])*params.filter_amplitude
    
def apply_inner_filter(params, fields):

    for f in fields.rho, fields.rho_u, fields.rho_v, fields.egy:
        filter_amplitude = params.filter_amplitude/10

        ny, nx = f.shape
        inner_filter = np.empty_like(f, dtype=np.float64)
        
        inner_filter[...] = (252*f[...] -210*(np.roll(f,-1,1)+np.roll(f,+1,1)) +
                120*(np.roll(f,-2,1)+np.roll(f,+2,1)) +
                -45*(np.roll(f,-3,1)+np.roll(f,+3,1)) +
                10*(np.roll(f,-4,1)+np.roll(f,+4,1)) +
                -1*(np.roll(f,-5,1)+np.roll(f,+5,1)))

        inner_filter[5:-5, :] += (252*f[...]
                -210*(np.roll(f,-1,0)+np.roll(f,+1,0)) + 
                120*(np.roll(f,-2,0)+np.roll(f,+2,0)) +
                -45*(np.roll(f,-3,0)+np.roll(f,+3,0)) +
                10*(np.roll(f,-4,0)+np.roll(f,+4,0)) +
                -1*(np.roll(f,-5,0)+np.roll(f,+5,0)))[5:-5, :]

        inner_filter[0, :] += (f[0,:]-5*f[1,:]+10*f[2,:]-10*f[3,:]+5*f[4,:]-1*f[5,:])
        
        inner_filter[1, :] += (-5*f[0,:]+26*f[1,:]-55*f[2,:]+60*f[3,:]-35*f[4,:]+10*f[5,:]
            -1*f[6,:])

        inner_filter[2, :] += (10*f[0,:]-55*f[1,:]+126*f[2,:]-155*f[3,:]+110*f[4,:]-45*f[5,:]
             +10*f[6,:]-1*f[7,:])

        inner_filter[3, :] += (-10*f[0,:]+60*f[1,:]-155*f[2,:]+226*f[3,:]-205*f[4,:]+120*f[5,:]
                -45*f[6,:]+10*f[7,:]-1*f[8,:])

        inner_filter[4, :] += (5*f[0,:]-35*f[1,:]+110*f[2,:]-205*f[3,:]+251*f[4,:]-210*f[5,:]
                +120*f[6,:]-45*f[7,:]+10*f[8,:]-1*f[9,:])

        inner_filter[-1, :] += (f[-1,:]-5*f[-2,:]+10*f[-3,:]-10*f[-4,:]+5*f[-5,:]-1*f[-6,:])
        
        inner_filter[-2, :] += (-5*f[-1,:]+26*f[-2,:]-55*f[-3,:]+60*f[-4,:]-35*f[-5,:]+10*f[-6,:]
            -1*f[-7,:])

        inner_filter[-3, :] += (10*f[-1,:]-55*f[-2,:]+126*f[-3,:]-155*f[-4,:]+110*f[-5,:]-45*f[-6,:]
             +10*f[-7,:]-1*f[-8,:])

        inner_filter[-4, :] += (-10*f[-1,:]+60*f[-2,:]-155*f[-3,:]+226*f[-4,:]-205*f[-5,:]+120*f[-6,:]
                -45*f[-7,:]+10*f[-8,:]-1*f[-9,:])

        inner_filter[-5, :] += (5*f[-1,:]-35*f[-2,:]+110*f[-3,:]-205*f[-4,:]+251*f[-5,:]-210*f[-6,:]
                +120*f[-7,:]-45*f[-8,:]+10*f[-9,:]-1*f[-10,:])
        
        f[...] = f[...] - filter_amplitude*inner_filter

def non_reflecting_boundary_conditions(params, fields, dndy):
    
    rho = fields.rho
    rho_u = fields.rho_u
    rho_v = fields.rho_v
    egy = fields.egy
    tmp = fields.tmp
    prs = fields.prs
    rho_rhs = fields.rho_rhs
    rho_u_rhs = fields.rho_u_rhs
    rho_v_rhs = fields.rho_v_rhs
    egy_rhs = fields.egy_rhs

    dx = params.dx
    dn = params.dn
    filter_amplitude = params.filter_amplitude
    Ma = params.Ma
    Ly = params.Ly
    P_inf = params.P_inf

    C_sound = np.sqrt(params.Cp/params.Cv*params.Rspecific*tmp)

    dpdy = dfdy(prs, dn)*dndy
    drhody = dfdy(rho, dn)*dndy
    dudy = dfdy(rho_u/rho, dn)*dndy
    dvdy = dfdy(rho_v/rho, dn)*dndy

    for dxdy in dpdy, drhody, dudy, dvdy:
        apply_boundary_filter(params, dxdy)
    
    L_1 = (rho_v/rho - C_sound) * (dpdy - rho*C_sound*dvdy)
    L_2 = rho_v/rho* (C_sound**2 * drhody - dpdy)
    L_3 = rho_v/rho* (dudy)
    L_4 = 0.4 * (1 - Ma**2)*C_sound/Ly*(prs - P_inf)

    d_1 = (1./C_sound**2)*(L_2 + 0.5*(L_4 + L_1))
    d_2 = 0.5*(L_1 + L_4)
    d_3 = L_3
    d_4 = 1./(2*rho*C_sound) * (L_4 - L_1)

    rho_rhs[0, :] = (rho_rhs - d_1)[0, :]
    rho_u_rhs[0, :] = (rho_u_rhs - rho_u/rho*d_1 - rho*d_3)[0, :]
    rho_v_rhs[0, :] = (rho_v_rhs - rho_v/rho*d_1 - rho*d_4)[0, :]
    egy_rhs[0, :] = (egy_rhs-
        0.5*np.sqrt((rho_u/rho)**2 + (rho_v/rho)**2)*d_1 -
        d_2*(prs + egy)/(rho*C_sound**2) -
        rho*(rho_v/rho*d_4+rho_u/rho*d_3))[0, :]

    L_1 = 0.4 * (1 - Ma**2)*C_sound/Ly*(prs - P_inf)
    L_2 = rho_v/rho* (C_sound**2 * drhody - dpdy)
    L_3 = rho_v/rho* (dudy)
    L_4 = (rho_v/rho + C_sound) * (dpdy + rho*C_sound*dvdy)

    d_1 = (1./C_sound**2)*(L_2 + 0.5*(L_4 + L_1))
    d_2 = 0.5*(L_1 + L_4)
    d_3 = L_3
    d_4 = 1./(2*rho*C_sound) * (L_4 - L_1)

    rho_rhs[-1, :] = (rho_rhs - d_1)[-1, :]
    rho_u_rhs[-1, :] = (rho_u_rhs - rho_u/rho*d_1 - rho*d_3)[-1, :]
    rho_v_rhs[-1, :] = (rho_v_rhs - rho_v/rho*d_1 - rho*d_4)[-1, :]
    egy_rhs[-1, :] = (egy_rhs-
        0.5*np.sqrt((rho_u/rho)**2 + (rho_v/rho)**2)*d_1 -
        d_2*(prs + egy)/(rho*C_sound**2) -
        rho*(rho_v/rho*d_4+rho_u/rho*d_3))[-1, :]

    # apply rhs filter
    for rhs in rho_rhs, rho_u_rhs, rho_v_rhs, egy_rhs:
        apply_boundary_filter(params, rhs)

@jit(nopython=True, nogil=True, parallel=True)
def dfdx(f, dx):
    out = np.empty_like(f, dtype=np.float64)
    ny, nx = f.shape
    for i in prange(ny):
        for j in prange(4):
            out[i,j] = (1./dx)*((4./5)*(f[i,j+1] - f[i,j-1]) + 
                    (-1./5)*(f[i,j+2] - f[i,j-2]) +
                    (4./105)*(f[i,j+3] - f[i,j-3]) +
                    (-1./280)*(f[i,j+4] - f[i,j-4]))

        for j in range(4, nx-4):
            out[i,j] = (1./dx)*((4./5)*(f[i,j+1] - f[i,j-1]) + 
                    (-1./5)*(f[i,j+2] - f[i,j-2]) +
                    (4./105)*(f[i,j+3] - f[i,j-3]) +
                    (-1./280)*(f[i,j+4] - f[i,j-4]))
            
        for j in range(-1, -5, -1):
            out[i,j] = (1./dx)*((4./5)*(f[i,j+1] - f[i,j-1]) + 
                    (-1./5)*(f[i,j+2] - f[i,j-2]) +
                    (4./105)*(f[i,j+3] - f[i,j-3]) +
                    (-1./280)*(f[i,j+4] - f[i,j-4]))
    return out

@jit(nopython=True, nogil=True, parallel=True)
def dfdy(f, dy):
    ny, nx = f.shape
    out = np.empty_like(f, dtype=np.float64)
    for i in prange(4, ny-4):
        for j in prange(nx):
            out[i,j] =  (1./dy)*((4./5)*(f[i+1,j] - f[i-1,j]) +
                    (-1./5)*(f[i+2,j] - f[i-2,j]) +
                    (4./105)*(f[i+3,j] - f[i-3,j]) +
                    (-1./280)*(f[i+4,j] - f[i-4,j]))
            out[0,j] = (-11*f[0,j]+18*f[1,j]-9*f[2,j]+2*f[3,j])/(6.*dy)
            out[1,j] = (-2*f[0,j]-3*f[1,j]+6*f[2,j]-1*f[3,j])/(6.*dy)
            out[2,j] = 2*(f[3,j]-f[1,j])/(3*dy) - 1*(f[4,j]-f[0,j])/(12*dy)
            out[3,j] = 3*(f[4,j]-f[2,j])/(4*dy) - 3*(f[5,j]-f[1,j])/(20*dy) + 1*(f[6,j]-f[0,j])/(60*dy) 

            out[-1,j] = -((-11*f[-1,j]+18*f[-2,j]-9*f[-3,j]+2*f[-4,j])/(6.*dy))
            out[-2,j] = -((-2*f[-1,j]-3*f[-2,j]+6*f[-3,j]-1*f[-4,j])/(6.*dy))
            out[-3,j] = -(2*(f[-4,j]-f[-2,j])/(3*dy) - 1*(f[-5,j]-f[-1,j])/(12*dy))
            out[-4,j] = -(3*(f[-5,j]-f[-3,j])/(4*dy) - 3*(f[-6,j]-f[-2,j])/(20*dy) + 1*(f[-7,j]-f[-1,j])/(60*dy))
    return out

def rhs(params, fields, dndy):

    eos(params, fields)

    rho = fields.rho
    rho_u = fields.rho_u
    rho_v = fields.rho_v
    egy = fields.egy
    tmp = fields.tmp
    prs = fields.prs
    rho_rhs = fields.rho_rhs
    rho_u_rhs = fields.rho_u_rhs
    rho_v_rhs = fields.rho_v_rhs
    egy_rhs = fields.egy_rhs

    rhs_euler_terms(params, fields, dndy)

    rhs_viscous_terms(params, fields, dndy)

    non_reflecting_boundary_conditions(params, fields, dndy)

def main():
    from .params import Params
    from .fields import Fields
    from .grid import asinh_grid
    from .timestepping import RK4
    
    paramfile = sys.argv[1]
    p = Params(paramfile)
    f = Fields(p)

    # make grid
    x, y, dndy, d2ndy2 = asinh_grid(p.N, p.N, p.Lx, p.Ly, p.grid_beta)

    # initialize fields
    weight = np.tanh(np.sqrt(np.pi)*y/p.vorticity_thickness)

    f.tmp[:, :] = p.T_inf2 + (weight+1)/2.*(p.T_inf1-p.T_inf2)
    f.rho[:, :] = p.P_inf/(p.Rspecific*f.tmp[:, :])
    
    f.rho_u[:, :] = f.rho*(p.U_inf2+(weight+1)/2.*(p.U_inf1-p.U_inf2))
    f.rho_v[:, :] = 0.0

    u_pert, v_pert = add_forcing(p, f, x, y, dndy, d2ndy2)

    f.rho_u += f.rho*u_pert
    f.rho_v += f.rho*v_pert

    f.egy[:, :] = 0.5*(f.rho_u**2 + f.rho_v**2)/f.rho + f.rho*p.Cv*f.tmp

    # make time stepper
    stepper = RK4(p, f)
    stepper.set_rhs_func(rhs, dndy)
    stepper.set_filter_func(apply_inner_filter)
    
    # run simulation
    import timeit

    t1 = timeit.default_timer()

    for i in range(p.timesteps):
        
        dt = calculate_timestep(p, f, x, y)

        print("Iteration: {}    Time: {}    Total energy: {}".format(i, dt*i, np.sum(f.egy)))

        stepper.step(dt)

        if i%200 == 0:
            outfile = h5py.File("{:05d}.hdf5".format(i))
            outfile.create_group("fields")
            outfile.create_dataset("fields/rho", data=f.rho)
            outfile.create_dataset("fields/rho_u", data=f.rho_u)
            outfile.create_dataset("fields/rho_v", data=f.rho_v)
            outfile.create_dataset("fields/tmp", data=f.tmp)

    t2 = timeit.default_timer()

    print(t2-t1)

if __name__ == "__main__":
    main()
