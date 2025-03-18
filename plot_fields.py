import sys
import os
import shutil
import numpy as np
import h5py as h5
import matplotlib as mpl
mpl.use('QtAgg')
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm.Barrier()
# infl = "out_py_DOP853_kap_1_0_C_1_0.h5"
# infl = "out_py_DOP853_gpu_kap_1_0_C_1_0.h5"
# infl = "out_jl_DP8_kap_1_0_C_1_0.h5"
# infl = "out_jl_Tsit5_kap_1_0_C_1_0.h5"
infl = "out_jl_DP8_gpu_kap_1_0_C_1_0.h5"
# infl = "out_jl_Tsit5_gpu_kap_1_0_C_1_0.h5"

with h5.File(infl, "r", libver='latest', swmr=True) as fl:
    if 'params' in fl:
        kap = fl['params/kap'][()]
        C = fl['params/C'][()]
        Lx = fl['params/Lx'][()]
        Ly = fl['params/Ly'][()]
        x = fl['data/x'][:] 
        y = fl['data/y'][:]
    else:
        raise KeyError("The 'params' group is missing in the HDF5 file.")

    t = fl['fields/t'][:]
    om = fl['fields/om'][0]
    n = fl['fields/n'][0]
    om_last = fl['fields/om'][100]
    n_last = fl['fields/n'][100]

fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

om_lim = np.max(np.abs(om_last))
n_lim = np.max(np.abs(om_last))

c1 = axs[0].pcolormesh(x,y, om_last, vmin=-om_lim,vmax=om_lim, shading='auto', cmap='seismic')
axs[0].set_title('$\\Omega$')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')  
axs[0].set_aspect('equal')
plt.colorbar(c1, ax=axs[0], fraction=0.046, pad=0.04)  

c2 = axs[1].pcolormesh(x,y, n_last, vmin=-n_lim, vmax=n_lim, shading='auto', cmap='seismic')
axs[1].set_title('$n$')
axs[1].set_xlabel('x')
axs[1].set_aspect('equal')
plt.colorbar(c2, ax=axs[1], fraction=0.046, pad=0.04) 

plt.tight_layout()
plt.show()