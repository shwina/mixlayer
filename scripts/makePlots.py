import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

max_val = []
min_val = []

for fname in sorted(glob.glob("*.hdf5")):
    try:
        f = h5py.File(fname)
    except OSError:
        print("unable to read file")
        continue
    field = f['fields/rho_y2']
    max_val.append(field[...].max())
    min_val.append(field[...].min())
    f.close()

for fname in sorted(glob.glob("*.hdf5")):
    try:
        f = h5py.File(fname)
    except OSError:
        print("unable to read file")
        continue
    field = f['fields/rho_y2']
    plt.imshow(field, cmap='hot', vmin=min(min_val), vmax=max(max_val))
    plt.colorbar()
    plt.savefig(fname.replace('hdf5', 'png'))
    plt.close()
    f.close()
