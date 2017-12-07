import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

for fname in sorted(glob.glob("*.hdf5")):
    f = h5py.File(fname)
    tmp = f['fields/tmp']
    plt.imshow(tmp)
    plt.savefig(fname.replace('hdf5', 'png'))
