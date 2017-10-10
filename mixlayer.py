import numpy as np
import matplotlib.pyplot as plt
import yaml
from numba import jit, float64, prange
import h5py

def jacobi_step(f, dx, dn, rhs, dndy, d2ndy2):
    #f[...] = (((np.roll(f,-1,0) + np.roll(f,1,0))/dx**2) + (
    #            np.roll(f,-1,1) + np.roll(f,1,1))/dy**2 + rhs)*(dx**2 + dy**2)/2 
    f_new = np.copy(f)
    denominator = 2/dx**2 + (2/dn**2)*dndy**2

    ny, nx = f.shape
    #f_new[...] = (-rhs +
    #        (np.roll(f,-1,1) + np.roll(f,+1,1))/dx**2 +
    #        (np.roll(f,-1,0) - np.roll(f,+1,0))/(2*dn)*d2ndy2 +
    #        (np.roll(f,-1,0) + np.roll(f,+1,0))/(dn**2)*dndy**2)/denominator
    f[...] = f_new

def add_forcing(stream, vort, x, y, U_ref, Famp_2d, disturbance_wavelength, nperiod, dn, dndy, d2ndy2):
    N = np.shape(x)[0]
    dx = 1./N
    dy = x.copy()
    dy[:-1, :] = y[1:, :] - y[:-1, :]
    dy[-1, :] = y[-1, :] - y[-2, :]

    fx = np.zeros_like(x, dtype=np.float64)
    fy = np.zeros_like(x, dtype=np.float64)
    fx_max = 0

    vort = np.zeros_like(x, dtype=np.float64)
    stream = np.zeros_like(x, dtype=np.float64)

    amplitudes = [1, 0.5, 0.35, 0.35]
    for i in range(4):
        #vort += np.exp(-np.pi*y**2/vorticity_thickness**2)*np.abs(
        #    amplitudes[i]*np.sin(np.pi*x/(2**i*disturbance_wavelength)))
        fx += amplitudes[i]*np.abs(np.sin(np.pi*x/(2**i*disturbance_wavelength)))
        fx_max = np.max([np.max(fx), fx_max])
    
    fx = fx/fx_max
    fy = np.exp(-np.pi*y**2/vorticity_thickness**2)
    
    vort[...] = fx*fy
    circ = np.sum(dy*dx*vort)

    vort[...] = (vort*Famp_2d*disturbance_wavelength*U_ref) / (circ/nperiod)

    for i in range(10000):
        print(i)
        jacobi_step(stream, dx, dn, -vort, dndy, d2ndy2)

    u_pert = (np.roll(stream, -1, 0) -  np.roll(stream, +1, 0))*dndy/(2*dn)
    v_pert = -(np.roll(stream, -1, 1) - np.roll(stream, +1, 1))/(2*dx)
    
    vort[...] = ((np.roll(stream, -1, 1) - 2*stream + np.roll(stream, +1, 1))/(dx**2) +
            ((np.roll(stream, -1, 0) - np.roll(stream, +1, 0))/(2*dn))*d2ndy2 +
            ((np.roll(stream, -1, 0) - 2*stream + np.roll(stream, +1, 0))/(dn**2))*dndy**2)

    circ = np.sum(dy*dx*vort)

    u_pert = u_pert*Famp_2d*disturbance_wavelength*U_ref / (circ/nperiod)
    v_pert = v_pert*Famp_2d*disturbance_wavelength*U_ref / (circ/nperiod)

    return u_pert, v_pert

def eos(rho, rho_u, rho_v, egy, tmp, prs, Cv, Rspecific):
    tmp[:, :] = (egy - 0.5*(rho_u**2 + rho_v**2)/rho)/(rho*Cv)
    prs[:, :] = rho*Rspecific*tmp

def calculate_timestep(x, y, rho, rho_u, rho_v, tmp,
        gamma_ref, mu_ref, kappa_ref, Cp, Cv, Rspecific, cfl_vel, cfl_visc):

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

def rhs_euler_terms(rho, rho_u, rho_v, egy, rho_rhs, rho_u_rhs,
        rho_v_rhs, egy_rhs, prs, dx, dn, dndy):
 
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

def rhs_viscous_terms(rho, rho_u, rho_v, egy, rho_rhs, rho_u_rhs,
        rho_v_rhs, egy_rhs, prs, tmp, dx, dn, dndy, mu, kappa):
    
    div_vel = dfdx(rho_u/rho, dx) + dfdy(rho_v/rho, dn)*dndy
    tau_11 = -(2./3)*mu*div_vel + 2*mu*dfdx(rho_u/rho, dx) 
    tau_12 = mu*(dfdx(rho_v/rho, dx) + dfdy(rho_u/rho, dn)*dndy)
    tau_22 = -(2./3)*mu*div_vel + 2*mu*dfdy(rho_v/rho, dn)*dndy
    
    tau_12[0, :] = (18.*tau_12[1, :] - 9*tau_12[2, :] + 2*tau_12[3, :])/11.
    tau_12[-1, :] = (18.*tau_12[-2, :] - 9*tau_12[-3, :] + 2*tau_12[-4, :])/11.

    rho_u_rhs += (-dfdx(tau_11, dx) - dfdy(tau_12,dn)*dndy)
    rho_v_rhs += (-dfdx(tau_12, dx) - dfdy(tau_22,dn)*dndy)
    egy_rhs += (dfdx(rho_u/rho*tau_11, dx) + dfdx(rho_v/rho*tau_12, dx) + dfdy(rho_u/rho*tau_12, dn)*dndy + dfdy(rho_v/rho*tau_22, dn)*dndy) + kappa*(dfdx(dfdx(tmp, dx), dx) + dfdy(dfdy(tmp, dn)*dndy, dn)*dndy)

def apply_boundary_filter(f, filter_amplitude):
    f[0, :] = f[0, :] - (f[0, :] - 5*f[1, :]
                + 10*f[2, :] - 10*f[3, :] + 5*f[4, :]
                - f[5, :])*filter_amplitude

    f[-1, :] = f[-1, :] - (f[-1, :] - 5*f[-2, :]
                + 10*f[-3, :] - 10*f[-4, :] + 5*f[-5, :]
                - f[-6, :])*filter_amplitude
    
def apply_inner_filter(f, filter_amplitude):
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

def non_reflecting_boundary_conditions(rho, rho_u, rho_v, egy, rho_rhs, rho_u_rhs,
        rho_v_rhs, egy_rhs, prs, tmp, dx, dn, dndy, C_sound, filter_amplitude,
        Ma, Ly, P_inf):

    dpdy = dfdy(prs, dn)*dndy
    drhody = dfdy(rho, dn)*dndy
    dudy = dfdy(rho_u/rho, dn)*dndy
    dvdy = dfdy(rho_v/rho, dn)*dndy

    for dxdy in dpdy, drhody, dudy, dvdy:
        apply_boundary_filter(dxdy, filter_amplitude)
    
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
        0.5*((rho_u/rho)**2 + (rho_v/rho)**2)*d_1 -
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
        0.5*((rho_u/rho)**2 + (rho_v/rho)**2)*d_1 -
        d_2*(prs + egy)/(rho*C_sound**2) -
        rho*(rho_v/rho*d_4+rho_u/rho*d_3))[-1, :]

    # apply rhs filter
    for rhs in rho_rhs, rho_u_rhs, rho_v_rhs, egy_rhs:
        apply_boundary_filter(rhs, filter_amplitude)

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
            out[i,j] =  (1./dy)*((4./5)*(f[i-1,j] - f[i+1,j]) +
                    (-1./5)*(f[i-2,j] - f[i+2,j]) +
                    (4./105)*(f[i-3,j] - f[i+3,j]) +
                    (-1./280)*(f[i-4,j] - f[i+4,j]))
            out[0,j] = (-11*f[0,j]+18*f[1,j]-9*f[2,j]+2*f[3,j])/(6.*dy)
            out[1,j] = (-2*f[0,j]-3*f[1,j]+6*f[2,j]-1*f[3,j])/(6.*dy)
            out[2,j] = 2*(f[3,j]-f[1,j])/(3*dy) - 1*(f[4,j]-f[0,j])/(12*dy)
            out[3,j] = 3*(f[4,j]-f[2,j])/(4*dy) - 3*(f[5,j]-f[1,j])/(20*dy) + 1*(f[6,j]-f[0,j])/(60*dy) 

            out[-1,j] = -((-11*f[-1,j]+18*f[-2,j]-9*f[-3,j]+2*f[-4,j])/(6.*dy))
            out[-2,j] = -((-2*f[-1,j]-3*f[-2,j]+6*f[-3,j]-1*f[-4,j])/(6.*dy))
            out[-3,j] = -(2*(f[-4,j]-f[-2,j])/(3*dy) - 1*(f[-5,j]-f[-1,j])/(12*dy))
            out[-4,j] = -(3*(f[-5,j]-f[-3,j])/(4*dy) - 3*(f[-6,j]-f[-2,j])/(20*dy) + 1*(f[-7,j]-f[-1,j])/(60*dy))
    return out
    
if __name__ == "__main__":
    # read parameters
    with open('params.yaml') as f:
        params = yaml.load(f)

    N = params['N']
    Lx = params['Lx']
    Ly = Lx*((N-1)/N)*2.

    Famp_2d = params['Famp_2d']

    P_inf = params['P_inf']
    T_inf1 = params['T_inf1']
    T_inf2 = params['T_inf2']

    Ma = params['Ma']
    Re = params['Re']
    Pr = params['Pr']
    nperiod = params['nperiod']

    Rgas = params['Rgas']
    molwt = params['molwt']

    grid_beta = params['grid_beta']

    cfl_vel = params['cfl_vel']
    cfl_visc = params['cfl_visc']

    filter_amplitude = params['filter_amplitude']

    # get free stream velocities:
    Rspecific = Rgas/molwt
    T_ref = max([T_inf2, 344.6])
    Cp = Pr*(3.227e-3 + 8.3894e-5*T_ref - 1.958e-8*T_ref**2)/(
            6.109e-6 + 4.604e-8*T_ref - 1.051e-11*T_ref**2)
    Cv = Cp - Rspecific
    C_sound1 = np.sqrt((Cp/Cv)*(Rspecific)*T_inf1)
    C_sound2 = np.sqrt((Cp/Cv)*(Rspecific)*T_inf2)
    rho_ref1 = P_inf/(Rspecific*T_inf1)
    rho_ref2 = P_inf/(Rspecific*T_inf2)
    U_inf1 = 2*Ma*C_sound1/(1+np.sqrt(rho_ref1/rho_ref2)*(C_sound1/C_sound2))
    U_inf2 = -np.sqrt(rho_ref1/rho_ref2)*U_inf1
    U_ref = U_inf1-U_inf2

    print(Rspecific)
    print(Cp)
    print(Cv)
    print(C_sound1)
    print(C_sound2)
    print(rho_ref1)
    print(U_inf1)
    print(U_inf2)

    # construct grid
    dx = Lx/N
    dn = 1./(N-1)
    x = np.arange(N)*dx*np.ones([N, N])
    y = np.arange(0, 1+dn, dn)*np.ones([N, N])
    y = y.T
    grid_A = 1./(2*grid_beta)*np.log((1 + (np.exp(grid_beta) - 1)*((Ly/2)/Ly))/(
        1 + (np.exp(-grid_beta) - 1)*((Ly/2)/Ly)))
    y = (Ly/2)*(1 + np.sinh(grid_beta*(y - grid_A))/np.sinh(
        grid_beta*grid_A))
    dndy = np.sinh(grid_beta*grid_A)/(grid_beta*(Ly/2)*(1+((y/(Ly/2))-1)**2*np.sinh(grid_beta*grid_A)**2)**0.5)*Ly
    d2ndy2 = ((y/(Ly/2) - 1)*(np.sinh(grid_beta*grid_A))**3)/(
            grid_beta*((Ly/2)**2)*((y/(Ly/2)-1)**2 * np.sinh(grid_beta*grid_A)**2 + 1)**1.5)

    y = y-Ly/2
    dn = Ly*dn

    # geometric parameters
    disturbance_wavelength = Lx/nperiod
    vorticity_thickness = disturbance_wavelength/7.29

    # reference viscosity; thermal and molecular diffusivities
    rho_ref = (rho_ref1+rho_ref2)/2.0
    mu_ref = (rho_ref*(U_inf1-U_inf2)*vorticity_thickness)/Re
    kappa_ref = 0.5*(Cp+Cp)*mu_ref/Pr 
    gamma_ref = mu_ref/(rho_ref*Pr)

    # initialize fields
    rho = np.zeros([N, N], dtype=np.float64)
    rho_u = np.zeros([N, N], dtype=np.float64)
    rho_v = np.zeros([N, N], dtype=np.float64)
    tmp = np.zeros([N, N], dtype=np.float64)
    prs = np.zeros([N, N], dtype=np.float64)
    egy = np.zeros([N, N], dtype=np.float64)
    rho_rhs = np.zeros([N, N], dtype=np.float64)
    rho_u_rhs = np.zeros([N, N], dtype=np.float64)
    rho_v_rhs = np.zeros([N, N], dtype=np.float64)
    egy_rhs = np.zeros([N, N], dtype=np.float64)
    stream = np.zeros([N, N], dtype=np.float64)
    vort = np.zeros([N, N], dtype=np.float64)

    weight = np.tanh(np.sqrt(np.pi)*y/vorticity_thickness)
    tmp[:, :] = T_inf2 + (weight+1)/2.*(T_inf1-T_inf2)
    rho[:, :] = P_inf/(Rspecific*tmp[:, :])
    rho_u[:, :] = rho*(U_inf2+(weight+1)/2.*(U_inf1-U_inf2))
    rho_v[:, :] = 0.0

    # read values of rho_u and rho_v since we don't know how
    # to generate them yet

    u_pert, v_pert = add_forcing(stream, vort, x, y, U_ref, Famp_2d, disturbance_wavelength, nperiod, dn, dndy, d2ndy2)

    rho_u += rho*u_pert
    rho_v += rho*v_pert

    rho_u_actual = np.loadtxt('init/u.txt').reshape([N, N])
    rho_v_actual = np.loadtxt('init/v.txt').reshape([N, N])

    plt.imshow((rho_u - rho_u_actual)/rho_u_actual)
    plt.colorbar()
    plt.show()

    plt.subplot(121)
    plt.streamplot(np.arange(N), np.arange(N), rho_u, rho_v, density=5)
    plt.subplot(122)
    plt.streamplot(np.arange(N), np.arange(N), rho_u_actual, rho_v_actual, density=5)

    plt.show()
    egy[:, :] = 0.5*(rho_u**2 + rho_v**2)/rho + rho*Cv*tmp

    import timeit

    t1 = timeit.default_timer()
    for i in range(10000):
        print(i, egy.max())
        eos(rho, rho_u, rho_v, egy, tmp, prs, Cv, Rspecific)

        dt = calculate_timestep(x, y, rho, rho_u, rho_v, tmp, gamma_ref, mu_ref, kappa_ref,
            Cp, Cv, Rspecific, cfl_vel, cfl_visc)

        dt *= 0.05

        rhs_euler_terms(rho, rho_u, rho_v, egy, rho_rhs, rho_u_rhs, rho_v_rhs, egy_rhs, prs, dx, dn, dndy)

        rhs_viscous_terms(rho, rho_u, rho_v, egy, rho_rhs, rho_u_rhs, rho_v_rhs, egy_rhs, prs, tmp, dx, dn, dndy, mu_ref, kappa_ref)

        C_sound = np.sqrt(Cp/Cv*Rspecific*tmp)

        non_reflecting_boundary_conditions(rho, rho_u, rho_v, egy, rho_rhs, rho_u_rhs,
                rho_v_rhs, egy_rhs, prs, tmp, dx, dn, dndy, C_sound, filter_amplitude, Ma, Ly, P_inf)

        rho[...] = rho[...] + dt*rho_rhs
        rho_u[...] = rho_u[...] + dt*rho_u_rhs
        rho_v[...] = rho_v[...] + dt*rho_v_rhs
        egy[...] = egy[...] + dt*egy_rhs

        apply_inner_filter(rho, filter_amplitude/10)
        apply_inner_filter(rho_u, filter_amplitude/10)
        apply_inner_filter(rho_v, filter_amplitude/10)
        apply_inner_filter(egy, filter_amplitude/10)

        print(egy.min(), egy.max())

        if i%200 == 0:
            f = h5py.File("{:05d}.hdf5".format(i))
            f.create_group("fields")
            f.create_dataset("fields/rho", data=rho)
            f.create_dataset("fields/rho_u", data=rho_u)
            f.create_dataset("fields/rho_v", data=rho_v)
            f.create_dataset("fields/tmp", data=tmp)
    t2 = timeit.default_timer()

    print(t2-t1)
