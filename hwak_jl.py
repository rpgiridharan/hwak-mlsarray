#%% Import libraries

import cupy as cp
import numpy as np
import gc
from mlsarray import mlsarray,slicelist,init_kspace_grid,irft,rft
import h5py as h5
from time import time
from functools import partial

#%% Define parameters

Npx,Npy=512,512
t0,t1=0,1000
dtstep,dtshow,dtsave=0.01,0.01,0.1
wecontinue=False
Nx,Ny=2*int(np.floor(Npx/3)),2*int(np.floor(Npy/3))
Lx,Ly=16*np.pi,16*np.pi
dkx,dky=2*np.pi/Lx,2*np.pi/Ly
sl=slicelist(Nx,Ny)
lkx,lky=init_kspace_grid(sl)
kx,ky=lkx*dkx,lky*dky
w=10.0
phik=1e-4*cp.exp(-lkx**2/2/w**2-lky**2/w**2)*cp.exp(1j*2*np.pi*cp.random.rand(lkx.size).reshape(lkx.shape))
nk=1e-4*cp.exp(-lkx**2/w**2-lky**2/w**2)*cp.exp(1j*2*np.pi*cp.random.rand(lkx.size).reshape(lkx.shape))
zk=np.hstack((phik,nk))
del lkx,lky

gc.collect()
xl,yl=np.arange(0,Lx,Lx/Npx),np.arange(0,Ly,Ly/Npy)
x,y=cp.meshgrid(cp.array(xl),cp.array(yl),indexing='ij')

kap=1.0
C=1.0
nu=5e-2
D=5e-2
mu1=1e2
mu0=0.0
sig=Lx/100
u=mlsarray(Npx,Npy)

#%% Define functions and classes

def save_data(fl,grpname,ext_flag,**kwargs):
    if not (grpname in fl):
        grp=fl.create_group(grpname)
    else:
        grp=fl[grpname]
    for l,m in kwargs.items():
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

def save_callback(t,y):
    zk=y.view(dtype=complex)
    phik,nk=zk[:int(zk.size/2)],zk[int(zk.size/2):]
    om=irft(-phik*(kx**2+ky**2), Npx, Npy, sl)
    dyphi=irft(1j*ky*phik, Npx, Npy, sl)
    vy=irft(1j*kx*phik, Npx, Npy, sl)
    n=irft(nk, Npx, Npy, sl)
    Gam=-np.mean(n*dyphi,1)
    Pi=np.mean(-dyphi*om,1)
    vbar=cp.mean(vy,1)
    ombar=cp.mean(om,1)
    nbar=cp.mean(n,1)
    save_data(fl,'fields',ext_flag=True,om=om.get(),n=n.get(),t=t.get())
    save_data(fl,'fluxes',ext_flag=True,Gam=Gam.get(),Pi=Pi.get(),t=t.get())
    save_data(fl,'fields/zonal/',ext_flag=True,vbar=vbar.get(),ombar=ombar.get(),nbar=nbar.get(),t=t.get())

def fshow(t,y):
    zk=y.view(dtype=complex)
    phik,nk=zk[:int(zk.size/2)],zk[int(zk.size/2):]
    dyphi=irft(1j*ky*phik, Npx, Npy, sl)             
    n=irft(nk, Npx, Npy, sl)
    Gam=-np.mean(n*dyphi)
    print(' Gam=', f'{Gam.get():.3g}')

def rhs(t,y):
    zk=y.view(dtype=complex)
    dzkdt=cp.zeros_like(zk)
    phik,nk=zk[:int(zk.size/2)],zk[int(zk.size/2):]
    dphikdt,dnkdt=dzkdt[:int(zk.size/2)],dzkdt[int(zk.size/2):]
    ksqr=kx**2+ky**2
    dxphi=irft(1j*kx*phik, Npx, Npy, sl)
    dyphi=irft(1j*ky*phik, Npx, Npy, sl)
    om=irft(-ksqr*phik, Npx, Npy, sl)
    n=irft(nk, Npx, Npy, sl)
    phi=irft(phik, Npx, Npy, sl)
    d2n=irft(-ksqr*nk, Npx, Npy, sl)
    sigk = cp.sign(ky)
    dphikdt[:]=-1j*kx*rft(dyphi*om, sl)/ksqr+1j*ky*rft(dxphi*om, sl)/ksqr
    dnkdt[:]=1j*kx*rft(dyphi*n, sl)-1j*ky*rft(dxphi*n, sl)
    dphikdt[:]+=-C*sigk*(phik-nk)/ksqr-nu*ksqr*phik
    dnkdt[:]+=-kap*1j*ky*phik-C*sigk*(phik-nk)/ksqr-D*ksqr*nk
    return dzkdt.view(dtype=float)

def save_data(fl,grpname,ext_flag,**kwargs):
    if not (grpname in fl):
        grp=fl.create_group(grpname)
    else:
        grp=fl[grpname]
    for l,m in kwargs.items():
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
        else:
            lptr=grp[l]
            if(ext_flag):
                lptr.resize((lptr.shape[0]+1,)+lptr.shape[1:])
                lptr[-1,]=m
            else:
                lptr[...]=m
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
                while(self.r.t<tnext):
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
                print(f't={t:.3g}, {time()-ct:.3g} secs elapsed.', end='')
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
    fl=h5.File('out2.h5','r+',libver='latest')
    fl.swmr_mode = True
    omk,nk=rft(cp.array(fl['fields/om'][-1,]), sl),rft(cp.array(fl['fields/n'][-1,]), sl)
    phik=-omk/(kx**2+ky**2)
    t0=fl['fields/t'][-1]
    zk=np.hstack((phik,nk))
else:
    fl=h5.File('out.h5','w',libver='latest')
    fl.swmr_mode = True
    save_data(fl,'data',ext_flag=False,x=x.get(),y=y.get(),kap=kap,C=C,nu=nu,D=D)

r=Gensolver('cupy_ivp.DOP853',rhs,t0,zk.view(dtype=float),t1,fsave=save_callback,fshow=fshow,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,rtol=1e-9,atol=1e-10)
r.run()
fl.close()
