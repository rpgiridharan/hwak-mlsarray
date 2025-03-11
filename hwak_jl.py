#%% Import libraries

import numpy as np
import gc
import os
from modules.mlsarray_cpu import mlsarray,slicelist,init_kspace_grid
from modules.mlsarray_cpu import irft2 as original_irft2, rft2 as original_rft2, irft as original_irft, rft as original_rft
from gamma import gammax,kymax
import h5py as h5
from time import time
from functools import partial
from juliacall import Main as jl

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
nu=4e-3*kymax(ky0,1.0,1.0)**2/kymax(ky0,kap,C)**2
D=4e-3*kymax(ky0,1.0,1.0)**2/kymax(ky0,kap,C)**2

output = 'out_kap_' + f'{kap:.1f}'.replace('.', '_') + '_C_' + f'{C:.1f}'.replace('.', '_') + '.h5'

# All times needs to be in float for the solver
dtstep,dtshow,dtsave=0.1,1.0,1.0
gamma_time=50/gammax(ky0, kap, C)
t0,t1=0.0,int(round(gamma_time/dtstep))*dtstep
rtol,atol=1e-10,1e-12 
wecontinue=False

w=10.0
phik=1e-4*np.exp(-lkx**2/2/w**2-lky**2/w**2)*np.exp(1j*2*np.pi*np.random.rand(lkx.size).reshape(lkx.shape))
nk=1e-4*np.exp(-lkx**2/w**2-lky**2/w**2)*np.exp(1j*2*np.pi*np.random.rand(lkx.size).reshape(lkx.shape))
zk=np.hstack((phik,nk))

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
    Gam=np.mean(vx*n,1)
    Pi=np.mean(vx*om,1)
    R=np.mean(vx*vy,1)
    vbar=np.mean(vy,1)
    ombar=np.mean(om,1)
    nbar=np.mean(n,1)
    save_data(fl,'fields',ext_flag=True,om=om.get(),n=n.get(),t=t.get())
    save_data(fl,'fluxes',ext_flag=True,Gam=Gam.get(),Pi=Pi.get(),R=R.get(),t=t.get())
    save_data(fl,'fields/zonal/',ext_flag=True,vbar=vbar.get(),ombar=ombar.get(),nbar=nbar.get(),t=t.get())

def fshow(t,y):
    zk=y.view(dtype=complex)
    phik,nk=zk[:int(zk.size/2)],zk[int(zk.size/2):]
    kpsq = kx**2 + ky**2
    dyphi=irft2(1j*ky*phik)             
    n=irft2(nk)

    # Gam=-np.mean(n*dyphi)
    # print(' Gam=', f'{Gam.get():.3g}')
    
    Ktot, Kbar = np.sum(kpsq*abs(phik)**2), np.sum(abs(kx[slbar] * phik[slbar])**2)
    print(f"Ktot={Ktot:.6}, Kbar/Ktot={np.round(Kbar/Ktot*100,1)}%")
                
    del phik, nk, kpsq, dyphi, n

def rhs(t,y):
    #Get the fields and create the time derivative
    zk=y.view(dtype=complex)
    dzkdt=np.zeros_like(zk)
    phik,nk=zk[:int(zk.size/2)],zk[int(zk.size/2):]
    dphikdt,dnkdt=dzkdt[:int(zk.size/2)],dzkdt[int(zk.size/2):]
    kpsq=kx**2+ky**2

    # Compute all the fields in real space that we need 
    dxphi=irft2(1j*kx*phik)
    dyphi=irft2(1j*ky*phik)
    n=irft2(nk)

    #Compute the non-linear terms
    dphikdt[:]=-1*(kx*ky*rft2(dxphi**2-dyphi**2) + (ky**2-kx**2)*rft2(dxphi*dyphi))/kpsq
    dnkdt[:]=1j*kx*rft2(dyphi*n)-1j*ky*rft2(dxphi*n)

    #Add the linear terms on non-zonal modes
    sigk = np.sign(ky)
    NZ = ky>0
    # dphikdt[:]+=(-C*(phik-nk)/kpsq - nu*kpsq*phik)*sigk
    # dnkdt[:]+=(-kap*1j*ky*phik + C*(phik-nk) - D*kpsq*nk)*sigk
    dphikdt[:]+=(-C*(phik-nk)/kpsq - nu*kpsq*phik) * NZ
    dnkdt[:]+=(-kap*1j*ky*phik + C*(phik-nk) - D*kpsq*nk) * NZ

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

        # Convert numpy/cupy arrays to a standard numpy array
        if hasattr(y0, 'get'):
            y0_array = y0.get()
        else:
            y0_array = np.array(y0)

        # Store if this is a GPU array
        self.is_gpu_array = hasattr(y0, 'get')

        # Initialize Julia environment
        jl.seval("using OrdinaryDiffEq")
        jl.seval("using LinearAlgebra")

        # Store the Python RHS function
        self.py_rhs_func = f
        
        # Pass initial values to Julia
        jl.py_y0 = y0_array
        jl.py_t0 = float(t0)
        jl.py_t1 = float(t1)

        # Create a bridge function for Julia to call Python that ensures proper type conversion
        def bridge_function(t, u):
            # Convert to numpy array to avoid serialization issues
            u_np = np.array(u)
            
            # If original array was GPU-based, move computation to GPU
            if self.is_gpu_array:
                u_gpu = np.array(u_np)
                # Call the RHS function
                result = f(float(t), u_gpu)
                # Get result back from GPU and ensure it's a numpy array
                if hasattr(result, 'get'):
                    return result.get().astype(np.float64)
                return np.array(result, dtype=np.float64)
            else:
                # CPU path - ensure result is proper float64 array
                result = f(float(t), u_np)
                return np.array(result, dtype=np.float64)

        # Make the bridge function available to Julia
        jl.bridge_func = bridge_function

        # Define the wrapper in Julia - explicitly cast types
        jl.seval("""
        function safe_rhs_wrapper(du, u, p, t)
            # Call the Python function via the bridge
            result = bridge_func(convert(Float64, t), Array(u))
            
            # Ensure result is properly handled - assuming it's now a NumPy array
            # Copy directly to the output array
            for i in 1:length(du)
                du[i] = result[i]
            end
            
            return nothing
        end
        """)

        # Set up the Julia ODE problem and solver - AVOID KEYWORD ARGS
        jl.seval("prob = ODEProblem(safe_rhs_wrapper, convert(Vector{Float64}, py_y0), (py_t0, py_t1))")
        self.problem = jl.prob  # Store the Julia problem

        # Choose appropriate solver
        solver_name = "Tsit5()"  # Default
        if solver == 'julia.Tsit5':
            solver_name = "Tsit5()"
        elif solver == 'julia.Dopri8':
            solver_name = "DP8()"
        elif solver == 'julia.ROCK2':
            solver_name = "ROCK2()"
        elif solver == 'julia.ROCK4':
            solver_name = "ROCK4()"

        # Set up solver parameters - use the correct approach compatible with solve()
        jl.rtol_val = float(kwargs.get('rtol', 1e-3))
        jl.atol_val = float(kwargs.get('atol', 1e-6))
        jl.dtmax_val = float(dtstep)
        jl.solver_str = solver_name
        
        # Create the integrator - use solve() first, then init() if needed
        jl.seval("""
        # Create the solver with properly named kwargs
        solver = eval(Meta.parse(solver_str))
        
        # Use solve to create the solution and integrator
        sol = solve(prob, solver, 
                    abstol=atol_val, 
                    reltol=rtol_val,
                    dtmax=dtmax_val,
                    save_everystep=false, 
                    dense=false)    
                    
        # Get the integrator
        integrator = sol.integrator
        """)
        
        self.integrator = jl.integrator
        
        # Attach a Python-friendly interface
        self.r = JuliaIntegrator(self.integrator, jl, is_gpu=self.is_gpu_array)

        # Store other parameters
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
        self.jl = jl

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

class JuliaIntegrator:
    """Python wrapper for Julia integrator to mimic scipy/cupy interface"""
    
    def __init__(self, integrator, jl, is_gpu=False):
        self.integrator = integrator
        self.jl = jl
        self.is_gpu = is_gpu
        
        # Get current time as float
        self.t = float(self.jl.seval("convert(Float64, integrator.t)"))
        
        # Get solution array with proper type conversion
        y_numpy = np.array(self.jl.seval("convert(Array{Float64}, Array(integrator.u))"))
        
        # Convert to GPU array if needed
        if self.is_gpu:
            self.y = np.array(y_numpy)
        else:
            self.y = y_numpy
        
    def integrate(self, tnext):
        """Step the integrator to time tnext"""
        # Pass the next time value with explicit type
        self.jl.tnext_val = float(tnext)
        self.jl.seval("step!(integrator, convert(Float64, tnext_val))")
        
        # Update time with proper type
        self.t = float(self.jl.seval("convert(Float64, integrator.t)"))
        
        # Get updated solution with proper type
        y_numpy = np.array(self.jl.seval("convert(Array{Float64}, Array(integrator.u))"))
        
        # Convert to GPU if needed
        if self.is_gpu:
            self.y = np.array(y_numpy)
        else:
            self.y = y_numpy
        
    def step(self):
        """Take a single step"""
        self.jl.seval("step!(integrator)")
        
        # Update time with proper type
        self.t = float(self.jl.seval("convert(Float64, integrator.t)"))
        
        # Get updated solution with proper type
        y_numpy = np.array(self.jl.seval("convert(Array{Float64}, Array(integrator.u))"))
        
        # Convert to GPU if needed
        if self.is_gpu:
            self.y = np.array(y_numpy)
        else:
            self.y = y_numpy

#%% Run the simulation

if(wecontinue):
    fl=h5.File(output,'r+',libver='latest')
    fl.swmr_mode = True
    omk,nk=rft2(np.array(fl['fields/om'][-1,])),rft2(np.array(fl['fields/n'][-1,]))
    phik=-omk/(kx**2+ky**2)
    t0=fl['fields/t'][-1]
    zk=np.hstack((phik,nk))
else:
    fl=h5.File(output,'w',libver='latest')
    fl.swmr_mode = True
    t=float(t0)
    save_data(fl,'data',ext_flag=False,x=x.get(),y=y.get(),kap=kap,C=C,nu=nu,D=D)

save_data(fl,'params',ext_flag=False,C=C,kap=kap,nu=nu,D=D,Lx=Lx,Ly=Ly,Npx=Npx, Npy=Npy)
r=Gensolver('julia.Dopri8',rhs,t0,zk.view(dtype=float),t1,fsave=save_callback,fshow=fshow,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,rtol=rtol,atol=atol)
r.run()
fl.close()
