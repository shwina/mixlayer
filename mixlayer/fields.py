import yaml
from attrdict import AttrDict
import numpy as np

from params import p
dims = [p.N, p.N]

f = AttrDict({})
f.rho = np.zeros(dims, dtype=np.float64)
f.rho_u = np.zeros(dims, dtype=np.float64)
f.rho_v = np.zeros(dims, dtype=np.float64)
f.tmp = np.zeros(dims, dtype=np.float64)
f.prs = np.zeros(dims, dtype=np.float64)
f.egy = np.zeros(dims, dtype=np.float64)
f.rho_rhs = np.zeros(dims, dtype=np.float64)
f.rho_u_rhs = np.zeros(dims, dtype=np.float64)
f.rho_v_rhs = np.zeros(dims, dtype=np.float64)
f.egy_rhs = np.zeros(dims, dtype=np.float64)
f.stream = np.zeros(dims, dtype=np.float64)
f.vort = np.zeros(dims, dtype=np.float64)
