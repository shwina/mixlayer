import h5py
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
import sys

fname = sys.argv[1]

hf = h5py.File(fname, 'r')

dsets = [hf[str(i)]['tmp'] for i in sorted(hf.keys(), key=int)]
arrays = [da.from_array(dset, chunks=(100, 100)) for dset in dsets]
x = da.stack(arrays, axis=0)
min_val = x.min().compute()
max_val = x.max().compute()

fig, ax = plt.subplots()
h = ax.imshow(x[0], cmap='hot')
fig.colorbar(h)

def plot_frame(i):
    h.set_data(x[i])

from matplotlib.animation import FuncAnimation
anim = FuncAnimation(fig, plot_frame, frames=range(len(x)), interval=50)

plt.show()
