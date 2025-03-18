import sys
import os
import shutil
import numpy as np
import h5py as h5
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pylab as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm.Barrier()
# infl = "out_py_DOP853_kap_1_0_C_1_0.h5"
# infl = "out_py_DOP853_gpu_kap_1_0_C_1_0.h5"
# infl = "out_jl_DP8_kap_1_0_C_1_0.h5"
# infl = "out_jl_Tsit5_kap_1_0_C_1_0.h5"
infl = "out_jl_DP8_gpu_kap_1_0_C_1_0.h5"
# infl = "out_jl_Tsit5_gpu_kap_1_0_C_1_0.h5"

outfl = infl.replace(".h5", ".mp4")

with h5.File(infl, "r", libver='latest', swmr=True) as fl:
    if 'params' in fl:
        kap = fl['params/kap'][()]
        C = fl['params/C'][()]
        Lx = fl['params/Lx'][()]
        Ly = fl['params/Ly'][()]
    else:
        raise KeyError("The 'params' group is missing in the HDF5 file.")

    t = fl['fields/t'][:]
    om = fl['fields/om'][0]
    n = fl['fields/n'][0]
    om_last = fl['fields/om'][-1]
    n_last = fl['fields/n'][-1]

Nx, Ny = om.shape[-2], om.shape[-1]

om_max_last = max(abs(om_last.min()), abs(om_last.max()))
n_max_last = max(abs(n_last.min()), abs(n_last.max()))

w, h = 10.5, 5
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(w, h))
qd = []
qd.append(ax[0].pcolormesh(om.T, cmap='seismic', rasterized=True, vmin=-om_max_last, vmax=om_max_last, shading='auto'))
qd.append(ax[1].pcolormesh(n.T, cmap='seismic', rasterized=True, vmin=-n_max_last, vmax=n_max_last, shading='auto'))
# cbar1 = fig.colorbar(qd[0], ax=ax[0], orientation='vertical')
# cbar2 = fig.colorbar(qd[1], ax=ax[1], orientation='vertical')
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')

# Update the figure to ensure the aspect settings take effect before creating the colorbars
fig.canvas.draw()

# Create colorbars with appropriate sizing to match plot height
cbar1 = fig.colorbar(qd[0], ax=ax[0], orientation='vertical', fraction=0.05, pad=0.05, shrink=0.8)
cbar2 = fig.colorbar(qd[1], ax=ax[1], orientation='vertical', fraction=0.05, pad=0.05, shrink=0.8)

if (comm.rank == 0):
    print((Nx, Ny))
ax[0].set_title('$\\Omega$', pad=-1)
ax[1].set_title('$n$', pad=-1)
ax[0].tick_params('y', labelleft=False)
ax[0].tick_params('x', labelbottom=False)
ax[1].tick_params('y', labelleft=False)
ax[1].tick_params('x', labelbottom=False)

plt.subplots_adjust(wspace=0.2, hspace=0.2)

nt0 = 10
nt = t.shape[0]

ax[0].axis('off')
ax[1].axis('off')

tx = fig.text(0.515, 0.9, "t=0", ha='center')
if (comm.rank == 0):
    lt = np.arange(nt)
    lt_loc = np.array_split(lt, comm.size)
    if not os.path.exists('_tmpimg_folder'):
        os.makedirs('_tmpimg_folder')
else:
    lt_loc = None
lt_loc = comm.scatter(lt_loc, root=0)

for j in lt_loc:
    print(j)
    with h5.File(infl, "r", libver='latest', swmr=True) as fl:
        om = fl['fields/om'][j]
        n = fl['fields/n'][j]

    qd[0].set_array(om.T.ravel())
    qd[1].set_array(n.T.ravel())

    tx.set_text('t=' + str(int(t[j]) * 1.0))
    fig.savefig("_tmpimg_folder/tmpout%04i" % (j + nt0) + ".png", dpi=600)
comm.Barrier()

if comm.rank == 0:
    with h5.File(infl, "r", libver='latest', swmr=True) as fl:
        om = fl['fields/om'][0]
        T = fl['fields/n'][0]

    qd[0].set_array(om.T.ravel())
    qd[1].set_array(T.T.ravel())
    tx.set_text('')

    fig.savefig("_tmpimg_folder/tmpout%04i" % (0) + ".png", dpi=200)
    for j in range(1, nt0):
        os.system("cp _tmpimg_folder/tmpout%04i" % (0) + ".png _tmpimg_folder/tmpout%04i" % (j) + ".png")
    
    os.system("ffmpeg -framerate 30 -y -i _tmpimg_folder/tmpout%04d.png -c:v libx264 -pix_fmt yuv420p -vf fps=30 " + outfl)
    shutil.rmtree("_tmpimg_folder")