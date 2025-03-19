#%% Import libraries

import numpy as np
import cupy as cp
import gc
import os
from modules.mlsarray_gpu import mlsarray,slicelist,init_kspace_grid
from modules.mlsarray_gpu import irft2 as original_irft2, rft2 as original_rft2, irft as original_irft, rft as original_rft
from modules.gamma import gammax,kymax
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
nu=1e-3*kymax(ky0,1.0,1.0)**4/kymax(ky0,kap,C)**4
D=1e-3*kymax(ky0,1.0,1.0)**4/kymax(ky0,kap,C)**4

solver='jl.KenCarp4' # 'jl.KenCarp3', 'jl.KenCarp4', 'jl.KenCarp47', 'jl.KenCarp5', 'jl.KenCarp58'
output = 'out_'+solver.replace('.','_')+'_gpu_kap_'+f'{kap:.1f}'.replace('.', '_') + '_C_' + f'{C:.1f}'.replace('.', '_') + '.h5'

# All times needs to be in float for the solver
dtstep,dtshow,dtsave=0.1,1.0,1.0
gamma_time=50/gammax(ky0, kap, C)
t0,t1=0.0,int(round(gamma_time/dtstep))*dtstep
rtol,atol=1e-6,1e-8
wecontinue=False

w=10.0
phik=1e-3*cp.exp(-lkx**2/2/w**2-lky**2/w**2)*cp.exp(1j*2*cp.pi*cp.random.rand(lkx.size).reshape(lkx.shape))
nk=1e-3*cp.exp(-lkx**2/w**2-lky**2/w**2)*cp.exp(1j*2*cp.pi*cp.random.rand(lkx.size).reshape(lkx.shape))
zk=cp.hstack((phik,nk))

del lkx,lky,xl,yl
gc.collect()

#%% Define functions and classes

irft2 = partial(original_irft2, Npx=Npx, Npy=Npy, sl=sl)
rft2 = partial(original_rft2, sl=sl)
irft = partial(original_irft, Npx=Npx, Nx=Nx)
rft = partial(original_rft, Nx=Nx)

# def save_last(t,y,fl):
#     zk = cp.array(y, copy=False)
#     save_data(fl,'last',ext_flag=False,zk=zk,t=t)

def save_callback(t, zk):
    phik, nk = zk[:int(zk.size/2)], zk[int(zk.size/2):]
    om = irft2(-phik*(kx**2+ky**2))
    vx = irft2(-1j*ky*phik)
    vy = irft2(1j*kx*phik)
    n = irft2(nk)
    Gam = cp.mean(vx*n, 1)
    Pi = cp.mean(vx*om, 1)
    R = cp.mean(vx*vy, 1)
    vbar = cp.mean(vy, 1)
    ombar = cp.mean(om, 1)
    nbar = cp.mean(n, 1)
    save_data(fl, 'fields', ext_flag=True, om=om.get(), n=n.get(), t=t)
    save_data(fl, 'fluxes', ext_flag=True, Gam=Gam.get(), Pi=Pi.get(), R=R.get(), t=t)
    save_data(fl, 'fields/zonal/', ext_flag=True, vbar=vbar.get(), ombar=ombar.get(), nbar=nbar.get(), t=t)

def fshow(t, zk):
    phik, nk = zk[:int(zk.size/2)], zk[int(zk.size/2):]
    kpsq = kx**2 + ky**2
    vx = irft2(-1j*ky*phik)             
    n = irft2(nk)

    Gam = cp.mean(vx*n)
    Ktot, Kbar = cp.sum(kpsq*abs(phik)**2), cp.sum(abs(kx[slbar] * phik[slbar])**2)
    
    # Print without elapsed time
    print(f"Gam={Gam:.3g}, Ktot={Ktot:.3g}, Kbar/Ktot={Kbar/Ktot*100:.1f}%")

def rhs_nonstiff(dzkdt, zk, p, t):
    """Non-stiff part of the RHS (advection, linear coupling)"""
    # Split zk into phik and nk
    phik, nk = zk[:int(zk.size/2)], zk[int(zk.size/2):]
    
    # Split dzkdt into dphikdt and dnkdt
    dphikdt, dnkdt = dzkdt[:int(zk.size/2)], dzkdt[int(zk.size/2):]
    
    kpsq = kx**2 + ky**2
    sigk = np.sign(ky) # zero for ky=0, 1 for ky>0
    
    # Compute all the fields in real space that we need 
    dxphi = irft2(1j*kx*phik)
    dyphi = irft2(1j*ky*phik)
    n = irft2(nk)
    
    # Compute the non-linear terms
    dphikdt[:] = -1*(kx*ky*rft2(dxphi**2-dyphi**2) + (ky**2-kx**2)*rft2(dxphi*dyphi))/kpsq
    dnkdt[:] = 1j*kx*rft2(dyphi*n) - 1j*ky*rft2(dxphi*n)

def rhs_stiff(dzkdt, zk, p, t):
    """Stiff part of the RHS (viscosity terms)"""
    # Split zk into phik and nk
    phik, nk = zk[:int(zk.size/2)], zk[int(zk.size/2):]
    
    # Split dzkdt into dphikdt and dnkdt
    dphikdt, dnkdt = dzkdt[:int(zk.size/2)], dzkdt[int(zk.size/2):]
    
    kpsq = kx**2 + ky**2
    sigk = np.sign(ky) # zero for ky=0, 1 for ky>0

    # Add the linear terms on non-zonal modes
    dphikdt[:] += (-C*(phik-nk)/kpsq)*sigk
    dnkdt[:] += (-kap*1j*ky*phik + C*(phik-nk))*sigk
    
    # Add the hyper viscosity terms on non-zonal modes
    dphikdt[:] = -nu*kpsq**2*phik*sigk
    dnkdt[:] = -D*kpsq**2*nk*sigk

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
    def __init__(self, solver, f_nonstiff, f_stiff, t0, y0, t1, fsave, fshow=None, fy=None, dtstep=0.1, dtshow=None, dtsave=None, dtfupdate=None, force_update=None, **kwargs):
        if(dtshow is None):
            dtshow=dtstep
        if(dtsave is None):
            dtsave=dtstep
        if isinstance(dtsave,float):
            dtsave=np.array([dtsave,])
        if isinstance(dtsave,list) or isinstance(dtsave,tuple):
            dtsave=np.array(dtsave)

        self.cp = cp  
            
        # Configure Julia to suppress CUDA library loading warnings
        jl.seval("""
        ENV["JULIA_CUDA_SOFT_RUNTIME_LOADING"] = "1"
        ENV["JULIA_CUDA_SILENT"] = "1"
        """)

        # Initialize Julia GPU environment
        jl.seval("using CUDA")
        jl.seval("using OrdinaryDiffEq") 
        jl.seval("using LinearAlgebra")
        jl.seval("using DiffEqCallbacks")
        jl.seval("using Sundials")
        jl.seval("using SparseArrays")

        # Pass the python RHS functions to Julia
        jl.py_rhs_nonstiff = lambda dy,y,p,t: f_nonstiff(self.jltocp(dy,y0),self.jltocp(y,y0),p,t) 
        jl.py_rhs_stiff = lambda dy,y,p,t: f_stiff(self.jltocp(dy,y0),self.jltocp(y,y0),p,t) 
        
        # Move initial data to GPU if it's not already there
        if not isinstance(y0, cp.ndarray):
            y0 = cp.asarray(y0)

        # Pass the CuPy array pointer to Julia and create a CuArray that shares the same memory
        jl.y0_ptr = y0.data.ptr
        jl.y0_shape = y0.shape
        jl.y0_size = y0.size
        jl.y0_dtype = str(y0.dtype)

        # Create a CuArray that shares memory with the CuPy array (generally complex128)
        jl.seval("""
            if y0_dtype == "complex128"
                y0_p = CuPtr{ComplexF64}(convert(UInt64, y0_ptr))
                py_y0_gpu = unsafe_wrap(CuArray, y0_p, (y0_size,))
            elseif y0_dtype == "float64"
                y0_p = CuPtr{Float64}(convert(UInt64, y0_ptr))
                py_y0_gpu = unsafe_wrap(CuArray, y0_p, (y0_size,))
            elseif y0_dtype == "complex64"
                y0_p = CuPtr{ComplexF32}(convert(UInt64, y0_ptr))
                py_y0_gpu = unsafe_wrap(CuArray, y0_p, (y0_size,))
            elseif y0_dtype == "float32"
                y0_p = CuPtr{Float32}(convert(UInt64, y0_ptr))
                py_y0_gpu = unsafe_wrap(CuArray, y0_p, (y0_size,))
            else
                error("Unsupported dtype: $y0_dtype")
            end
        """)

        jl.py_y0 = jl.py_y0_gpu
        jl.py_t0 = t0
        jl.py_t1 = t1

        jl.seval("""
        # Add a timer to track computation time
        ct = time()
        """)

        # Set up the Julia ODE problem
        jl.seval("""
        prob = SplitODEProblem(py_rhs_nonstiff, py_rhs_stiff, py_y0, (py_t0, py_t1))
        """)
        
        self.problem = jl.prob  # Store the Julia problem

        # Choose appropriate IMEX solver
        solver_name = "KenCarp4(autodiff=false)"  # Default
        if solver == 'jl.KenCarp3':
            solver_name = "KenCarp3(autodiff=false)"  # 3rd order IMEX method, 3 stages
        elif solver == 'jl.KenCarp4':
            solver_name = "KenCarp4(autodiff=false)"  # 4th order IMEX method, 5 stages, default solver
        elif solver == 'jl.KenCarp47':
            solver_name = "KenCarp47(autodiff=false)"  # 4th order IMEX method, 7 stages, more stable than KenCarp4
        elif solver == 'jl.KenCarp5':
            solver_name = "KenCarp5(autodiff=false)"  # 5th order IMEX method, 8 stages, higher accuracy
        elif solver == 'jl.KenCarp58':
            solver_name = "KenCarp58(autodiff=false)"  # 5th order IMEX method, 8 stages, more stable than KenCarp5

        # Set up solver parameters
        jl.rtol_val = kwargs.get('rtol', 1e-7)
        jl.atol_val = kwargs.get('atol', 1e-9)
        jl.dtmax_val = dtstep
        jl.solver_str = solver_name
        jl.kwargs_py = kwargs

        self.jl = jl

        # Pass Python callbacks to Julia
        if callable(fsave):
            # Create a list with the single function and proper conversion
            jl.py_fsave_list = [lambda t, y: fsave(t, self.jltocp(y, y0))]
        else:
            # Assuming fsave is already a list of functions with proper conversion
            jl.py_fsave_list = [lambda t, y, f=f: f(t, self.jltocp(y, y0)) for f in fsave]
            
        jl.py_fshow = lambda t, y: fshow(t,self.jltocp(y, y0))
        jl.dtsave_val = dtsave[0]
        jl.dtshow_val = dtshow
        
        # Create callback functions in Julia to call Python functions
        jl.seval("""
        # Create two callback functions: one for save, one for show
        function save_cb(integrator)
            # Get current state and time
            y = integrator.u
            t = integrator.t
            
            # Call all save functions
            for fsave in py_fsave_list
                fsave(t, y)
            end
            return false  # continue integration
        end
        
        function show_cb(integrator)
            if !isnothing(py_fshow)
                # Get current state and time
                y = integrator.u
                t = integrator.t
                
                elapsed = time() - ct
                print("t=$(round(t, digits=3)), $(round(elapsed, digits=3)) secs elapsed.", " ")
                # Call show function
                py_fshow(t, y)
            end
            return false  # continue integration
        end
        
        # Generate time points for callbacks
        start_time = py_t0
        end_time = py_t1
        
        # Create periodic callbacks
        save_callback = PeriodicCallback(save_cb, dtsave_val)
        show_callback = PeriodicCallback(show_cb, dtshow_val)
        
        # Add saving at the initial and final times
        save_at_endpoints = PresetTimeCallback([start_time, end_time], save_cb)
        show_at_endpoints = PresetTimeCallback([start_time, end_time], show_cb)
        
        # Combine callbacks
        callback_set = CallbackSet(save_callback, show_callback, save_at_endpoints, show_at_endpoints)
        """)

        # Store other parameters
        self.dtstep, self.dtshow, self.dtsave = dtstep, dtshow, dtsave
        self.t0, self.t1 = t0, t1
        if(not(fy is None) and not(force_update is None)):
            self.fy = fy
            self.force_update = force_update
            if(dtfupdate is None):
                dtfupdate = dtstep
            self.dtfupdate = dtfupdate
        if(callable(fsave)):
            self.fsave = [fsave,]
        else:
            self.fsave = fsave
        self.fshow = fshow
        self.jl = jl

    def run(self, verbose=True):
        """Run the integration using solve() directly with the callbacks"""
        jl = self.jl
        
        jl.seval("""
        # Create the solver with properly named kwargs
        solver = eval(Meta.parse(solver_str))
        
        # Convert Python kwargs to Julia kwargs
        kwargs = Dict{Symbol, Any}()
        for (k, v) in pairs(kwargs_py)
            kwargs[Symbol(k)] = v
        end

        # Directly solve the ODE problem
        sol = solve(prob, solver, 
                   dtmax=dtmax_val,
                   reltol=rtol_val,
                   abstol=atol_val,
                   save_everystep=false,
                   dense=false,
                   callback=callback_set;
                   kwargs...)
        """)
    
    # Add the jltocp method for GPU data transfer
    def jltocp(self, y, zk):
        """Convert Julia GPU array to CuPy array by memory sharing
        
        Arguments:
            y: Julia CuArray
            zk: NumPy array shape reference
        
        Returns:
            CuPy array with shared memory
        """
        p = self.jl.UInt64(self.jl.pointer(y))
        mem = self.cp.cuda.UnownedMemory(p, zk.nbytes, None)
        memptr = self.cp.cuda.MemoryPointer(mem, 0)
        return self.cp.ndarray(zk.shape, dtype=zk.dtype, memptr=memptr)

    def cleanup(self):
        # Clean up resources to free GPU memory
        if hasattr(self, 'r'):
            try:
                self.jl.seval("""
                # Clear integrator's solution to free memory
                if isdefined(Main, :integrator) && integrator !== nothing
                    integrator = nothing
                end
                
                # Force garbage collection in Julia
                GC.gc(true)
                
                # Explicitly clear GPU memory 
                if CUDA.functional()
                    CUDA.reclaim()
                end
                """)
            except Exception as e:
                print(f"Warning: Error during Julia GPU cleanup: {e}")
            
        # Clear Julia variables that might hold GPU arrays
        if hasattr(self, 'jl'):
            self.jl.seval("""
            # Clear any global variables holding GPU arrays
            if @isdefined py_y0_gpu
                py_y0_gpu = nothing
            end
            
            if @isdefined prob
                prob = nothing
            end
            
            if @isdefined integrator
                integrator = nothing
            end
            
            # Force garbage collection in Julia
            GC.gc(true)
            
            # Explicitly reclaim GPU memory
            if CUDA.functional()
                CUDA.reclaim()
            end
            """)
        
        # Force Python garbage collection
        gc.collect()


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
r=Gensolver(solver,rhs_nonstiff,rhs_stiff,t0,zk,t1,fsave=save_callback,fshow=fshow,
            dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,reltol=rtol,abstol=atol)

try:
    r.run()
finally:
    # Clean up GPU resources explicitly
    if hasattr(r, 'cleanup'):
        r.cleanup()
    
    # Clear any remaining CuPy arrays
    del zk
    if 'phik' in locals(): del phik
    if 'nk' in locals(): del nk
    
    # Force CuPy to release memory
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    
    # Force Python garbage collection
    gc.collect()
    
    # Close the output file
    fl.close()
    print("Output file closed and GPU memory released")