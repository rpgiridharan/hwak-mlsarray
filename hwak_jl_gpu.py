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
Nx,Ny=2*int(cp.floor(Npx/3)),2*int(cp.floor(Npy/3))
Lx,Ly=16*cp.pi,16*cp.pi
dkx,dky=2*cp.pi/Lx,2*cp.pi/Ly
sl=slicelist(Nx,Ny)
lkx,lky=init_kspace_grid(sl)
kx,ky=lkx*dkx,lky*dky
slbar=cp.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Ny/2)] #Slice to access only zonal modes
ky0=ky[:int(Ny/2)-1] #ky at just kx=0
# Construct real space with padded resolution because it's needed in the RHS when going to real space
xl,yl=cp.arange(0,Lx,Lx/Npx),cp.arange(0,Ly,Ly/Npy)
x,y=cp.meshgrid(cp.array(xl),cp.array(yl),indexing='ij')

# Physical parameters
kap=1.0
C=1.0
nu=4e-3*kymax(ky0,1.0,1.0)**2/kymax(ky0,kap,C)**2
D=4e-3*kymax(ky0,1.0,1.0)**2/kymax(ky0,kap,C)**2

output = 'out_jl_ROCK4_kap_' + f'{kap:.1f}'.replace('.', '_') + '_C_' + f'{C:.1f}'.replace('.', '_') + '.h5'

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
#     zk = cp.array(y, copy=False)
#     save_data(fl,'last',ext_flag=False,zk=zk,t=t)

def save_callback(t, y):
    # Ensure y is a proper numpy array
    zk = cp.array(y, copy=False)
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
    save_data(fl, 'fields', ext_flag=True, om=om, n=n, t=t)
    save_data(fl, 'fluxes', ext_flag=True, Gam=Gam, Pi=Pi, R=R, t=t)
    save_data(fl, 'fields/zonal/', ext_flag=True, vbar=vbar, ombar=ombar, nbar=nbar, t=t)

def fshow(t, y):
    zk = cp.array(y, copy=False)
    phik, nk = zk[:int(zk.size/2)], zk[int(zk.size/2):]
    kpsq = kx**2 + ky**2
    dyphi = irft2(1j*ky*phik)             
    n = irft2(nk)

    Gam = -cp.mean(n*dyphi)
    Ktot, Kbar = cp.sum(kpsq*abs(phik)**2), cp.sum(abs(kx[slbar] * phik[slbar])**2)
    
    # Print without elapsed time
    print(f"Gam={Gam:.3g}, Ktot={Ktot:.3g}, Kbar/Ktot={Kbar/Ktot*100:.1f}%")

def rhs(dy, y, p, t):
    """RHS function for Julia's ODE solvers (dy, y, p, t) format
    
    Arguments:
        dy: output array where derivatives should be stored
        y: current state vector (will be viewed as complex)
        p: parameters (unused in this case)
        t: current time
    """

    zk = cp.array(y,copy=False)
    dzkdt = cp.array(dy,copy=False)

    # Split zk into phik and nk
    phik, nk = zk[:int(zk.size/2)], zk[int(zk.size/2):]
    
    # Split dzkdt into dphikdt and dnkdt
    dphikdt, dnkdt = dzkdt[:int(zk.size/2)], dzkdt[int(zk.size/2):]

    kpsq = kx**2 + ky**2

    # Compute all the fields in real space that we need 
    dxphi = irft2(1j*kx*phik)
    dyphi = irft2(1j*ky*phik)
    n = irft2(nk)
    
    # Compute the non-linear terms
    dphikdt[:] = -1*(kx*ky*rft2(dxphi**2-dyphi**2) + (ky**2-kx**2)*rft2(dxphi*dyphi))/kpsq
    dnkdt[:] = 1j*kx*rft2(dyphi*n) - 1j*ky*rft2(dxphi*n)

    # Add the linear terms on non-zonal modes
    sigk = cp.sign(ky)
    dphikdt[:] += (-C*(phik-nk)/kpsq - nu*kpsq*phik)*sigk
    dnkdt[:] += (-kap*1j*ky*phik + C*(phik-nk) - D*kpsq*nk)*sigk

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
                if(cp.isscalar(m)):
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
    def __init__(self, solver, f, t0, y0, t1, fsave, fshow=None, fy=None, dtstep=0.1, dtshow=None, dtsave=None, dtfupdate=None, force_update=None, **kwargs):
        if(dtshow is None):
            dtshow=dtstep
        if(dtsave is None):
            dtsave=dtstep
        if isinstance(dtsave,float):
            dtsave=cp.array([dtsave,])
        if isinstance(dtsave,list) or isinstance(dtsave,tuple):
            dtsave=cp.array(dtsave)

        # Initialize Julia environment
        jl.seval("using OrdinaryDiffEq")
        jl.seval("using LinearAlgebra")
        jl.seval("using DiffEqCallbacks") 
        jl.seval("using CUDA")
        jl.seval("CUDA.allowscalar(false)")

        # Pass the python RHS function to Julia
        jl.py_rhs_func = f 
        
        # Pass initial values to Julia
        jl.py_y0 = y0
        jl.py_t0 = t0
        jl.py_t1 = t1

        # Define a direct wrapper in Julia that calls the Python function
        jl.seval("""
        function direct_rhs_wrapper(du, u, p, t)
            # Call the Python function directly
            # Both du and u stay on GPU throughout
            py_rhs_func(du, u, nothing, t)
            return nothing
        end
        """)

        # Set up the Julia ODE problem and solver
        jl.seval("""
        # Create the ODE problem using the direct wrapper
        prob = ODEProblem(direct_rhs_wrapper, py_y0, (py_t0, py_t1))
        """)
        
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

        # Set up solver parameters
        jl.rtol_val = kwargs.get('rtol', 1e-7)
        jl.atol_val = kwargs.get('atol', 1e-9)
        jl.dtmax_val = dtstep
        jl.solver_str = solver_name
        
        # Pass Python callbacks to Julia
        if callable(fsave):
            jl.py_fsave_list = [fsave]
        else:
            jl.py_fsave_list = fsave
            
        jl.py_fshow = fshow
        jl.dtsave_val = dtsave[0]
        jl.dtshow_val = dtshow
        
        # Create callback functions in Julia to call Python functions
        jl.seval("""
        # Create two callback functions: one for save, one for show
        function save_cb(integrator)
            # Get current state and time
            u = integrator.u
            t = integrator.t
            
            # Call all save functions
            for fsave in py_fsave_list
                fsave(t, u)
            end
            return false  # continue integration
        end
        
        function show_cb(integrator)
            if !isnothing(py_fshow)
                # Get current state and time
                u = integrator.u
                t = integrator.t
                
                print("t=$(t).", " ")
                # Call show function
                py_fshow(t, u)
            end
            return false  # continue integration
        end
        
        # Generate time points for callbacks
        start_time = py_t0
        end_time = py_t1
        
        # Generate save times (ensure we include t0 and t1)
        save_times = vcat([start_time], collect(ceil(start_time/dtsave_val)*dtsave_val:dtsave_val:end_time), [end_time])
        save_times = sort(unique(save_times))
        
        # Generate show times (ensure we include t0 and t1)
        show_times = vcat([start_time], collect(ceil(start_time/dtshow_val)*dtshow_val:dtshow_val:end_time), [end_time])
        show_times = sort(unique(show_times))
        
        # Create the callbacks
        save_callback = PresetTimeCallback(save_times, save_cb)
        show_callback = PresetTimeCallback(show_times, show_cb)
        
        # Combine callbacks
        callback_set = CallbackSet(save_callback, show_callback)
        
        # Create the solver with properly named kwargs
        solver = eval(Meta.parse(solver_str))
        
        # Use solve to create the solution and integrator with callbacks
        integrator = init(prob, solver, 
                    abstol=atol_val, 
                    reltol=rtol_val,
                    dtmax=dtmax_val,
                    save_everystep=false,
                    dense=false,
                    callback=callback_set)
        """)
        
        self.integrator = jl.integrator
        
        # Attach a Python-friendly interface
        self.r = JuliaIntegrator(self.integrator, jl)

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
        """Run the integration with the callbacks already configured in Julia"""
        t0, t1 = self.t0, self.t1
        r = self.r
        r.integrate(t1)
        
        # Note: We don't need to manually call fsave or fshow at the end
        # because the Julia callbacks will handle this at t1

class JuliaIntegrator:
    """Python wrapper for Julia integrator to mimic scipy interface"""
    
    def __init__(self, integrator, jl):
        self.integrator = integrator
        self.jl = jl
        
        # Get current time and solution array
        self.t = integrator.t
        self.y = integrator.u
        
    def integrate(self, final_time):
        """Integrate the solution from current time to final_time"""
        # Avoid integration if already at final time
        if abs(self.t - final_time) < 1e-10:
            return
        
        jl.final_time = final_time
        
        try:
            # Use the proper function to integrate to the final time directly
            self.jl.seval("""
            # Find next multiple of dtshow_val after start_time
            start_time = integrator.t
            first_report = ceil(start_time / dtshow_val) * dtshow_val
            
            # Create report times from first_report to final_time in steps of dtshow_val
            report_times = first_report:dtshow_val:final_time
            
            # Add integer time points if they're not already included
            integer_times = ceil(start_time):1.0:floor(final_time)
            all_report_times = sort(unique(vcat(collect(report_times), collect(integer_times))))
            
            # Set up step points - first add the final time to ensure we stop there
            all_times = sort(unique(vcat(all_report_times, [final_time])))
            
            # Step through the integration manually to the specified points
            for next_time in all_times
                if next_time > integrator.t
                    step!(integrator, next_time - integrator.t, true)
                end
            end
            """)
        except Exception as e:
            print(f"Exception during integration: {e}")
            raise
        
        # Get updated time and solution
        self.t = self.jl.integrator.t 
        self.y = self.jl.integrator.u 

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
r=Gensolver('julia.Dopri8',rhs,t0,zk,t1,fsave=save_callback,fshow=fshow,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,rtol=rtol,atol=atol)

try:
    print(f"Starting simulation: t0={t0:.2f}, t1={t1:.2f}")   
    r.run()
    print("Simulation completed successfully")
except Exception as e:
    print(f"Error during simulation: {str(e)}")
finally:
    # No need to cancel alarm
    fl.close()
    print("Output file closed")