import numpy as np
import yaml

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

grid_A = 1./(2*grid_beta)*np.log((1 + (np.exp(grid_beta) - 1)*((Ly/2)/Ly))/(
    1 + (np.exp(-grid_beta) - 1)*((Ly/2)/Ly)))
y = (Ly/2)*(1 + np.sinh(grid_beta*(y - grid_A))/np.sinh(
    grid_beta*grid_A))

# geometric parameters
disturbance_wavelength = Lx/nperiod
vorticity_thickness = disturbance_wavelength/7.29

#import matplotlib.pyplot as plt
#plt.pcolor(x, y_phys, np.zeros_like(x), edgecolor='black', facecolor='none')
#plt.show()

# initialize fields
rho = np.zeros([N, N], dtype=np.float64)
u_x = np.zeros([N, N], dtype=np.float64)
u_y = np.zeros([N, N], dtype=np.float64)
tmp = np.zeros([N, N], dtype=np.float64)
egy = np.zeros([N, N], dtype=np.float64)

weight = (np.tanh(np.sqrt(np.pi*y/vorticity_thickness)) + 1.)/2.0
tmp[:, :] = T_inf2 + weight*(T_inf1-T_inf2)
rho[:, :] = (P_inf/(Rspecific*tmp[:, :]))/((rho_ref1+rho_ref2)/2.)
u_x[:, :] = rho*(U_inf2+(weight+1)/2.*(U_inf1-U_inf2))
u_y[:, :] = 0.0
egy[:, :] = 0.5*(u_x**2 + u_y**2)/rho + rho*Cv*tmp
