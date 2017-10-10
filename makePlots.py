import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob

for fname in glob.glob("*.hdf5"):
    f = h5py.File(fname)
    egy = f['egy']
    plt.imshow(egy)
    plt.savefig(fname.replace('hdf5', 'png'))
