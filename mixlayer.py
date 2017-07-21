import numpy as np
import matplotlib.pyplot as plt
import yaml


def eos(rho, rho_u, rho_v, egy, tmp, pressure, Cv, Rspecific):
    tmp[:, :] = (egy - 0.5*(rho_u**2 + rho_v**2)/rho)/(rho*Cv)
    pressure[:, :] = rho*Rspecific*tmp

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

def rhs_euler_terms():
    pass

def dfdx(f, dx):
    dfdx = (1./dx)*((4./5)*(np.roll(f,+1,1)-np.roll(f,-1,1)) + 
            (-1./5)*(np.roll(f,+2,1)-np.roll(f,-2,1)) +
            (4./105)*(np.roll(f,+3,1)-np.roll(f,-3,1)) +
            (-1./280)*(np.roll(f,+4,1)-np.roll(f,-4,1)))
    return dfdx

def dfdy(f, dy):
    N = f.shape[0]
    dfdy =  (1./dy)*((4./5)*(np.roll(f,+1,0)-np.roll(f,-1,0)) + 
            (-1./5)*(np.roll(f,+2,0)-np.roll(f,-2,0)) +
            (4./105)*(np.roll(f,+3,0)-np.roll(f,-3,0)) +
            (-1./280)*(np.roll(f,+4,0)-np.roll(f,-4,0)))
    
    dfdy[0, :] = (-11*f[0,:]+18*f[1,:]-9*f[2,:]+2*f[3,:])/(6.*dy)
    dfdy[1, :] = (-2*f[0,:]-3*f[1,:]+6*f[2,:]-1*f[3:])/(6.*dy)
    dfdy[2, :] = (2*f[3,:]-1*f[1,:])/(3*dy) - (1*f[4,:]-f[0,:])/(12*dy)
    dfdy[3, :] = (3*f[4,:]-3*f[2,:])/(4*dy) - (3*f[5,:]-f[1,:])/(20*dy) + (1*f[6,:]-f[0,:])/(60*dy) 

    dfdy[-1, :] = -((-11*f[-1,:]+18*f[-2,:]-9*f[-3,:]+2*f[-4,:])/(6.*dy))
    dfdy[-2, :] = -((-2*f[-1,:]-3*f[-2,:]+6*f[-3,:]-1*f[-4:])/(6.*dy))
    dfdy[-3, :] = -((2*f[-4,:]-1*f[-2,:])/(3*dy) - (1*f[-5,:]-f[-1,:])/(12*dy))
    dfdy[-4, :] = -((3*f[-5,:]-3*f[-3,:])/(4*dy) - (3*f[-6,:]-f[-2,:])/(20*dy) + (1*f[-7,:]-f[-1,:])/(60*dy))
    return dfdy
    

# read parameters
with open('params.yaml') as f:
    params = yaml.load(f)

N = params['N']
Lx = params['Lx']
Ly = params['Ly']

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


print(Rspecific)
print(Cp)
print(Cv)
print(C_sound1)
print(C_sound2)
print(rho_ref1)
print(U_inf1)
print(U_inf2)

# construct grid
x, y = np.meshgrid(np.arange(0, Lx+Lx/(N-1), Lx/(N-1)), np.arange(0, Ly+Ly/(N-1), Ly/(N-1)))
y = y - y/2
grid_A = 1./(2*grid_beta)*np.log((1 + (np.exp(grid_beta) - 1)*((Ly/2)/Ly))/(
    1 + (np.exp(-grid_beta) - 1)*((Ly/2)/Ly)))
y = (Ly/2)*(1 + np.sinh(grid_beta*(y - grid_A))/np.sinh(
    grid_beta*grid_A))

# geometric parameters
disturbance_wavelength = Lx/nperiod
vorticity_thickness = disturbance_wavelength/7.29

# reference viscosity; thermal and molecular diffusivities
rho_ref = (rho_ref1+rho_ref2)/2.0
mu_ref = (rho_ref*(U_inf1-U_inf2)*vorticity_thickness)/Re
kappa_ref = 0.5*(Cp+Cp)*mu_ref/Pr 
gamma_ref = mu_ref/(rho_ref*Pr)

#import matplotlib.pyplot as plt
#plt.pcolor(x, y, np.zeros_like(x), edgecolor='black', facecolor='none')
#plt.show()

# initialize fields
rho = np.zeros([N, N], dtype=np.float64)
rho_u = np.zeros([N, N], dtype=np.float64)
rho_v = np.zeros([N, N], dtype=np.float64)
tmp = np.zeros([N, N], dtype=np.float64)
pressure = np.zeros([N, N], dtype=np.float64)
egy = np.zeros([N, N], dtype=np.float64)
rho_rhs = np.zeros([N, N], dtype=np.float64)
rho_u_rhs = np.zeros([N, N], dtype=np.float64)
rho_v_rhs = np.zeros([N, N], dtype=np.float64)
egy_rhs = np.zeros([N, N], dtype=np.float64)

weight = (np.tanh(np.sqrt(np.pi*y/vorticity_thickness)) + 1.)/2.0
tmp[:, :] = T_inf2 + weight*(T_inf1-T_inf2)
rho[:, :] = P_inf/(Rspecific*tmp[:, :])
#rho_u[:, :] = rho*(U_inf2+(weight+1)/2.*(U_inf1-U_inf2))
#rho_v[:, :] = 0.0


# read values of rho_u and rho_v since we don't know how
# to generate them yet
rho_u = rho*np.loadtxt('init/u.txt')*(U_inf1-U_inf2)
rho_v = rho*np.loadtxt('init/v.txt')*(U_inf1-U_inf2)
egy[:, :] = 0.5*(rho_u**2 + rho_v**2)/rho + rho*Cv*tmp


eos(rho, rho_u, rho_v, egy, tmp, pressure, Cv, Rspecific)

dt = calculate_timestep(x, y, rho, rho_u, rho_v, tmp, gamma_ref, mu_ref, kappa_ref,
    Cp, Cv, Rspecific, cfl_vel, cfl_visc)
print(dt)

