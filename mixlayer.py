import numpy as np
import yaml

with open('params.yaml') as f:
    params = yaml.load(f)

N = params['N']
Lx = params['Lx']
Ly = params['Ly']

Pinf = params['Pinf']
Tinf_1 = params['Tinf_1']
Tinf_2 = params['Tinf_2']

Ma = params['Ma']
Re = params['Re']
Pr = params['Pr']

Rgas = params['Rgas']
molwt = params['molwt']

grid_beta = params['grid_beta']


Rspecific = Rgas/molwt
T_ref = max([Tinf_2, 344.6])
Cp = Pr*(3.227e-3 + 8.3894e-5*T_ref - 1.958e-8*T_ref**2)/(
        6.109e-6 + 4.604e-8*T_ref - 1.051e-11*T_ref**2)
Cv = Cp - Rspecific
C_sound1 = np.sqrt((Cp/Cv)*(Rspecific)*Tinf_1)
C_sound2 = np.sqrt((Cp/Cv)*(Rspecific)*Tinf_2)
rho_ref1 = Pinf/(Rspecific*Tinf_1)
rho_ref2 = Pinf/(Rspecific*Tinf_2)
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

x, y = np.meshgrid(np.arange(0, 1, 1./(N-1)), np.arange(0, 1, 1./(N-1)))
rho = np.zeros([N, N], dtype=np.float64)
u_x = np.zeros([N, N], dtype=np.float64)
u_y = np.zeros([N, N], dtype=np.float64)
egy = np.zeros([N, N], dtype=np.float64)

grid_A = 1./(2*grid_beta)*np.log((1 + (np.exp(grid_beta) - 1)*((Ly/2)/Ly))/(
    1 + (np.exp(-grid_beta) - 1)*((Ly/2)/Ly)))
y_phys = (Ly/2)*(1 + np.sinh(grid_beta*(y - grid_A))/np.sinh(
    grid_beta*grid_A))

import matplotlib.pyplot as plt
plt.pcolor(x, y_phys, np.zeros_like(x), edgecolor='black', facecolor='none')
plt.show()
