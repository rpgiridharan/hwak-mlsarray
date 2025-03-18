#%% Import libraries

import cupy as cp
import numpy as np
import gc
import os
from modules.mlsarray_gpu import mlsarray,slicelist,init_kspace_grid
from modules.mlsarray_gpu import irft2 as original_irft2, rft2 as original_rft2, irft as original_irft, rft as original_rft
from modules.gamma import gammax,kymax
import h5py as h5
from time import time
from functools import partial

#%% Define parameters

Npx,Npy=256,256
Nx,Ny=2*int(np.floor(Npx/3)),2*int(np.floor(Npy/3))
Lx,Ly=16*np.pi,16*np.pi
dkx,dky=2*np.pi/Lx,2*np.pi/Ly
sl=slicelist(Nx,Ny)
lkx,lky=init_kspace_grid(sl)
kx,ky=lkx*dkx,lky*dky
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Ny/2)] #Slice to access only zonal modes
ky0=ky[:int(Ny/2)-1] #ky at just kx=0
# Construct real space with padded resolution because it's needed in the RHS when going to real space
xl,yl=np.arange(0,Lx,Lx/Npx),np.arange(0,Ly,Ly/Npy)
x,y=np.meshgrid(np.array(xl),np.array(yl),indexing='ij')

# Physical parameters
kap=1.0
C=1.0
nu=1e-3*kymax(ky0,1.0,1.0)**2/kymax(ky0,kap,C)**2
D=1e-3*kymax(ky0,1.0,1.0)**2/kymax(ky0,kap,C)**2

output = 'out_py_DOP853_gpu_kap_' + f'{kap:.1f}'.replace('.', '_') + '_C_' + f'{C:.1f}'.replace('.', '_') + '.h5'

# All times needs to be in float for the solver
dtstep,dtshow,dtsave=0.1,1.0,1.0
gamma_time=50/gammax(ky0, kap, C)
t0,t1=0.0,int(round(gamma_time/dtstep))*dtstep
rtol,atol=1e-10,1e-12 
wecontinue=False

w=10.0
phik=1e-4*cp.exp(-lkx**2/2/w**2-lky**2/w**2)*cp.exp(1j*2*cp.pi*cp.random.rand(lkx.size).reshape(lkx.shape))
nk=1e-4*cp.exp(-lkx**2/w**2-lky**2/w**2)*cp.exp(1j*2*cp.pi*cp.random.rand(lkx.size).reshape(lkx.shape))
zk=cp.hstack((phik,nk))

del lkx,lky,xl,yl
gc.collect()

#%% Define functions and classes

irft2 = partial(original_irft2, Npx=Npx, Npy=Npy, sl=sl)
rft2 = partial(original_rft2, sl=sl)
irft = partial(original_irft, Npx=Npx, Nx=Nx)
rft = partial(original_rft, Nx=Nx)

# def save_last(t,y,fl):
#     zk=y.view(dtype=complex)
#     save_data(fl,'last',ext_flag=False,zk=zk.get(),t=t.get())

def save_callback(t,y):
    zk=y.view(dtype=complex)
    phik,nk=zk[:int(zk.size/2)],zk[int(zk.size/2):]
    om=irft2(-phik*(kx**2+ky**2))
    vx=irft2(-1j*ky*phik)
    vy=irft2(1j*kx*phik)
    n=irft2(nk)
    Gam=cp.mean(vx*n,1)
    Pi=cp.mean(vx*om,1)
    R=cp.mean(vx*vy,1)
    vbar=cp.mean(vy,1)
    ombar=cp.mean(om,1)
    nbar=cp.mean(n,1)
    save_data(fl,'fields',ext_flag=True,om=om.get(),n=n.get(),t=t.get())
    save_data(fl,'fluxes',ext_flag=True,Gam=Gam.get(),Pi=Pi.get(),R=R.get(),t=t.get())
    save_data(fl,'fields/zonal/',ext_flag=True,vbar=vbar.get(),ombar=ombar.get(),nbar=nbar.get(),t=t.get())

def fshow(t,y):
    zk=y.view(dtype=complex)
    phik,nk=zk[:int(zk.size/2)],zk[int(zk.size/2):]
    kpsq = kx**2 + ky**2
    dyphi=irft2(1j*ky*phik)             
    n=irft2(nk)

    Gam=-cp.mean(n*dyphi)    
    Ktot, Kbar = cp.sum(kpsq*abs(phik)**2), cp.sum(abs(kx[slbar] * phik[slbar])**2)
    print(f"Gam={Gam.get():.3g}, Ktot={Ktot:.3g}, Kbar/Ktot={Kbar/Ktot*100:.1f}%")
                
    del phik, nk, kpsq, dyphi, n

def rhs(t,y):
    #Get the fields and create the time derivative
    zk=y.view(dtype=complex)
    dzkdt=cp.zeros_like(zk)
    phik,nk=zk[:int(zk.size/2)],zk[int(zk.size/2):]
    dphikdt,dnkdt=dzkdt[:int(zk.size/2)],dzkdt[int(zk.size/2):]

    kpsq=kx**2+ky**2
    sigk = cp.sign(ky) # zero for ky=0, 1 for ky>0

    # Compute all the fields in real space that we need 
    dxphi=irft2(1j*kx*phik)
    dyphi=irft2(1j*ky*phik)
    n=irft2(nk)

    #Compute the non-linear terms
    dphikdt[:]=-1*(kx*ky*rft2(dxphi**2-dyphi**2) + (ky**2-kx**2)*rft2(dxphi*dyphi))/kpsq
    dnkdt[:]=1j*kx*rft2(dyphi*n)-1j*ky*rft2(dxphi*n)

    # Add the linear terms on non-zonal modes
    dphikdt[:] += (-C*(phik-nk)/kpsq)*sigk
    dnkdt[:] += (-kap*1j*ky*phik + C*(phik-nk))*sigk

    # Add the hyper viscosity terms on non-zonal modes
    dphikdt[:] += -nu*kpsq**2*phik*sigk
    dnkdt[:] += -D*kpsq**2*nk*sigk

    del phik, nk, dphikdt, dnkdt, kpsq, dxphi, dyphi, n
    return dzkdt.view(dtype=float)

def save_data(fl,grpname,ext_flag,**kwargs):
    if not (grpname in fl):
        grp=fl.create_group(grpname)
    else:
        grp=fl[grpname]
    for l,m in kwargs.items():
        if hasattr(m, 'get'):
            m = m.get()
        if not l in grp:
            if(not ext_flag):
                grp[l]=m
            else:
                if(np.isscalar(m)):
                    grp.create_dataset(l,(1,),maxshape=(None,),dtype=type(m))
                    if(not fl.swmr_mode):
                        fl.swmr_mode = True
                else:
                    grp.create_dataset(l,(1,)+m.shape,chunks=(1,)+m.shape,maxshape=(None,)+m.shape,dtype=m.dtype)
                    if(not fl.swmr_mode):
                        fl.swmr_mode = True
                lptr=grp[l]
                lptr[-1,]=m
                lptr.flush()
        elif(ext_flag):
            lptr=grp[l]
            lptr.resize((lptr.shape[0]+1,)+lptr.shape[1:])
            lptr[-1,]=m
            lptr.flush()
        fl.flush()

class Gensolver:    
    def __init__(self,solver,f,t0,y0,t1,fsave,fshow=None,fy=None,dtstep=0.1,dtshow=None,dtsave=None,dtfupdate=None,force_update=None,**kwargs):
        if(dtshow is None):
            dtshow=dtstep
        if(dtsave is None):
            dtsave=dtstep
        if isinstance(dtsave,float):
            dtsave=np.array([dtsave,])
        if isinstance(dtsave,list) or isinstance(dtsave,tuple):
            dtsave=np.array(dtsave)
        if solver=='scipy.DOP853':
            from scipy.integrate import DOP853
            print(kwargs)
            self.r=DOP853(f,t0,y0,t1,max_step=dtstep,**kwargs)
        if solver=='cupy_ivp.DOP853':
            from modules.cupy_ivp import DOP853
            print(kwargs)
            self.r=DOP853(f,t0,y0,t1,max_step=dtstep,**kwargs)
        elif solver=='scipy.RK45':
            from scipy.integrate import RK45
            self.r=RK45(f,t0,y0,t1,max_step=dtstep,**kwargs)
        elif solver=='cupy_ivp.RK45':
            from modules.cupy_ivp import RK45
            self.r=RK45(f,t0,y0,t1,max_step=dtstep,**kwargs)
        elif solver=='scipy.vode':
            from scipy.integrate import ode
            self.r=ode(f).set_integrator('vode',**kwargs)
            self.r.set_initial_value(y0,t0)
#            self.r.step = lambda : self.r.integrate(t=t1,step=True)
        if not hasattr(self.r, 'integrate'):
            def integrate(tnext):
                while(self.r.t < tnext):
                    self.r.step()
            self.r.integrate=integrate
        self.dtstep,self.dtshow,self.dtsave=dtstep,dtshow,dtsave
        self.t0,self.t1=t0,t1
        if(not(fy is None) and not(force_update is None)):
            self.fy=fy
            self.force_update=force_update
            if(dtfupdate is None):
                dtfupdate=dtstep
            self.dtfupdate=dtfupdate
        if(callable(fsave)):
            self.fsave=[fsave,]
        else:
            self.fsave=fsave
        self.fshow=fshow

    def run(self):
        dtstep,dtshow,dtsave=self.dtstep,self.dtshow,self.dtsave
        dtfupdate=None
        t0,t1=self.t0,self.t1
        r=self.r
        trnd=int(-np.log10(min(dtstep,dtshow,min(dtsave))/100))
        ct=time()
        t=t0

        if t0 < dtstep : #saving initial condition only if it's a new simulation staring at t0=0.0
            for l in range(len(self.fsave)): #save t=t0
                self.fsave[l](r.t,r.y)

        tnext=round(t0+dtstep,trnd)
        tshownext=round(t0+dtshow,trnd)
        tsavenext=np.array([round(t0+l,trnd) for l in dtsave])
        if('dtfupdate' in self.__dict__.keys()):
            dtfupdate=self.dtfupdate
            tnextfupdate=round(t0+dtfupdate,trnd)    
                
        while(t<t1):
            r.integrate(tnext)
            tnext=round(tnext+dtstep,trnd)
            t=r.t
            if(not(dtfupdate is None)):
                if(r.t>=tnextfupdate):
                    tnextfupdate=round(tnextfupdate+dtfupdate,trnd)
                    self.force_update(self.fy,t)
            if(r.t>=tshownext):
                print('t='+f'{t:.3g}'+', '+f'{time()-ct:.3g}'+" secs elapsed." , end=' ')
                if(callable(self.fshow)):
                    self.fshow(r.t,r.y)
                else:
                    print()
                tshownext=round(tshownext+dtshow,trnd)
            for l in range(len(dtsave)):
                if(r.t>=tsavenext[l]):
                    tsavenext[l]=round(tsavenext[l]+dtsave[l],trnd)
                    self.fsave[l](r.t,r.y)

#%% Run the simulation

if(wecontinue):
    fl=h5.File(output,'r+',libver='latest')
    fl.swmr_mode = True
    omk,nk=rft2(cp.array(fl['fields/om'][-1,])),rft2(cp.array(fl['fields/n'][-1,]))
    phik=-omk/(kx**2+ky**2)
    t0=fl['fields/t'][-1]
    zk=cp.hstack((phik,nk))
else:
    fl=h5.File(output,'w',libver='latest')
    fl.swmr_mode = True
    t=float(t0)
    save_data(fl,'data',ext_flag=False,x=x,y=y,kap=kap,C=C,nu=nu,D=D)

save_data(fl,'params',ext_flag=False,C=C,kap=kap,nu=nu,D=D,Lx=Lx,Ly=Ly,Npx=Npx, Npy=Npy)
r=Gensolver('cupy_ivp.DOP853',rhs,t0,zk.view(dtype=float),t1,fsave=save_callback,fshow=fshow,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,rtol=rtol,atol=atol)
r.run()
fl.close()
