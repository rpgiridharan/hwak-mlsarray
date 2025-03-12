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
import signal
import sys
import logging

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

def timeout_handler(signum, frame):
    raise TimeoutError("Simulation took too long")

irft2 = partial(original_irft2, Npx=Npx, Npy=Npy, sl=sl)
rft2 = partial(original_rft2, sl=sl)
irft = partial(original_irft, Npx=Npx, Nx=Nx)
rft = partial(original_rft, Nx=Nx)

# def save_last(t,y,fl):
#     zk=y.view(dtype=complex)
#     save_data(fl,'last',ext_flag=False,zk=zk.get(),t=t.get())

def save_callback(t, y):
    # Ensure y is a proper numpy array
    y_array = np.array(y, dtype=float)
    zk = y_array.reshape(-1, 2).view(dtype=complex).flatten()
    
    phik, nk = zk[:int(zk.size/2)], zk[int(zk.size/2):]
    om = irft2(-phik*(kx**2+ky**2))
    vx = irft2(-1j*ky*phik)
    vy = irft2(1j*kx*phik)
    n = irft2(nk)
    Gam = np.mean(vx*n, 1)
    Pi = np.mean(vx*om, 1)
    R = np.mean(vx*vy, 1)
    vbar = np.mean(vy, 1)
    ombar = np.mean(om, 1)
    nbar = np.mean(n, 1)
    save_data(fl, 'fields', ext_flag=True, om=om, n=n, t=t)
    save_data(fl, 'fluxes', ext_flag=True, Gam=Gam, Pi=Pi, R=R, t=t)
    save_data(fl, 'fields/zonal/', ext_flag=True, vbar=vbar, ombar=ombar, nbar=nbar, t=t)

def fshow(t, y):
    # Ensure y is a proper numpy array
    y_array = np.array(y, dtype=float)
    zk = y_array.reshape(-1, 2).view(dtype=complex).flatten()
    
    phik, nk = zk[:int(zk.size/2)], zk[int(zk.size/2):]
    kpsq = kx**2 + ky**2
    dyphi = irft2(1j*ky*phik)             
    n = irft2(nk)

    Gam = -np.mean(n*dyphi)
    Ktot, Kbar = np.sum(kpsq*abs(phik)**2), np.sum(abs(kx[slbar] * phik[slbar])**2)
    print(f"Gam={Gam:.3g}, Ktot={Ktot:.3g}, Kbar/Ktot={Kbar/Ktot*100:.1f}%")
                
    del phik, nk, kpsq, dyphi, n

def rhs(dy, y, p, t):
    """RHS function for Julia's ODE solvers (dy, y, p, t) format
    
    Arguments:
        dy: output array where derivatives should be stored
        y: current state vector (will be viewed as complex)
        p: parameters (unused in this case)
        t: current time
    """

    # Convert arrays to numpy arrays if they're not already
    # This handles both Python numpy arrays and Julia arrays
    y_array = np.array(y, dtype=float)
    
    # Get the fields and create the time derivative - use reshape instead of view
    zk = y_array.reshape(-1, 2).view(dtype=complex).flatten()  # Convert to complex view
    phik, nk = zk[:int(zk.size/2)], zk[int(zk.size/2):]
    kpsq = kx**2 + ky**2

    # Compute all the fields in real space that we need 
    dxphi = irft2(1j*kx*phik)
    dyphi = irft2(1j*ky*phik)
    n = irft2(nk)
    
    # Create temporary arrays for the complex derivatives
    dzkdt = np.zeros_like(zk)
    dphikdt, dnkdt = dzkdt[:int(zk.size/2)], dzkdt[int(zk.size/2):]

    #Compute the non-linear terms
    dphikdt[:] = -1*(kx*ky*rft2(dxphi**2-dyphi**2) + (ky**2-kx**2)*rft2(dxphi*dyphi))/kpsq
    dnkdt[:] = 1j*kx*rft2(dyphi*n) - 1j*ky*rft2(dxphi*n)

    #Add the linear terms on non-zonal modes
    sigk = np.sign(ky)
    dphikdt[:] += (-C*(phik-nk)/kpsq - nu*kpsq*phik)*sigk
    dnkdt[:] += (-kap*1j*ky*phik + C*(phik-nk) - D*kpsq*nk)*sigk

    # Convert complex to real and safely assign to the output array
    dy_real = dzkdt.view(dtype=float)
    
    # Use direct assignment for Julia arrays (avoid np.copyto)
    for i in range(len(dy)):
        dy[i] = float(dy_real[i])

    # dy[:] = dzkdt.view(dtype=float)

    # Cleanup
    del phik, nk, dphikdt, dnkdt, kpsq, dxphi, dyphi, n, dzkdt, zk, y_array

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

class Gensolver:    
    def __init__(self, solver, f, t0, y0, t1, fsave, fshow=None, fy=None, dtstep=0.1, dtshow=None, dtsave=None, dtfupdate=None, force_update=None, **kwargs):
        if(dtshow is None):
            dtshow=dtstep
        if(dtsave is None):
            dtsave=dtstep
        if isinstance(dtsave,float):
            dtsave=np.array([dtsave,])
        if isinstance(dtsave,list) or isinstance(dtsave,tuple):
            dtsave=np.array(dtsave)

        # Convert to standard numpy array - no GPU arrays
        y0_array = np.array(y0)

        # Initialize Julia environment
        jl.seval("using OrdinaryDiffEq")
        jl.seval("using LinearAlgebra")
        jl.seval("using DiffEqCallbacks") 

        # Pass the python RHS function to Julia
        jl.py_rhs_func = f 
        
        # Pass initial values to Julia
        jl.py_y0 = y0_array
        jl.py_t0 = float(t0)
        jl.py_t1 = float(t1)

        # Define a direct wrapper in Julia that calls the Python function
        jl.seval("""
        function direct_rhs_wrapper(du, u, p, t)
            # Call the Python function directly
            # Note: Julia will handle passing the values and converting them back
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
        jl.rtol_val = float(kwargs.get('rtol', 1e-3))
        jl.atol_val = float(kwargs.get('atol', 1e-6))
        jl.dtmax_val = float(dtstep)
        jl.solver_str = solver_name
        
        # Pass Python callbacks to Julia
        if callable(fsave):
            jl.py_fsave_list = [fsave]
        else:
            jl.py_fsave_list = fsave
            
        jl.py_fshow = fshow
        jl.dtsave_val = float(dtsave[0])
        jl.dtshow_val = float(dtshow)
        
        # Create callback functions in Julia to call Python functions
        jl.seval("""
        # Create two callback functions: one for save, one for show
        function save_cb(integrator)
            # Get current state and time
            u = integrator.u
            t = integrator.t
            
            # Convert to proper Python types
            t_py = convert(Float64, t)
            u_py = convert(Array{Float64}, Array(u))
            
            # Call all save functions
            for fsave in py_fsave_list
                fsave(t_py, u_py)
            end
            return false  # continue integration
        end
        
        function show_cb(integrator)
            if !isnothing(py_fshow)
                # Get current state and time
                u = integrator.u
                t = integrator.t
                
                # Convert to proper Python types
                t_py = convert(Float64, t)
                u_py = convert(Array{Float64}, Array(u))
                
                print("t=$(t_py).", " ")
                # Call show function
                py_fshow(t_py, u_py)
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
        sol = solve(prob, solver, 
                    abstol=atol_val, 
                    reltol=rtol_val,
                    dtmax=dtmax_val,
                    save_everystep=false,
                    dense=false,
                    callback=callback_set)
                    
        # Get the integrator
        integrator = sol.integrator
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
        
        print(f"Starting integration from t={t0:.2f} to t={t1:.2f}")
        
        # For large integrations, use checkpoints for progress reporting only
        if t1 - t0 > 10.0:
            checkpoint_times = np.arange(t0 + 5.0, t1, 5.0)
            for next_stop in checkpoint_times:
                # Just log progress, callbacks handle actual saving/showing
                print(f"Integration progress: {next_stop:.1f}/{t1:.1f}")
                r.integrate(next_stop)
        else:
            r.integrate(t1)
        
        # Note: We don't need to manually call fsave or fshow at the end
        # because the Julia callbacks will handle this at t1
        print(f"Integration completed at t={r.t:.2f}")

class JuliaIntegrator:
    """Python wrapper for Julia integrator to mimic scipy interface"""
    
    def __init__(self, integrator, jl):
        self.integrator = integrator
        self.jl = jl
        
        # Get current time as float
        self.t = float(self.jl.seval("convert(Float64, integrator.t)"))
        
        # Get solution array with proper type conversion
        self.y = np.array(self.jl.seval("convert(Array{Float64}, Array(integrator.u))"))
        
    def integrate(self, final_time):
        """Integrate the solution from current time to final_time"""
        # Avoid integration if already at final time
        if abs(self.t - final_time) < 1e-10:
            return
        
        # Pass the final time to Julia
        self.jl.final_time = float(final_time)
        
        # Pass dtshow to Julia for proper intervals
        self.jl.dtshow_val = float(self.jl.dtshow)
        
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
            
            # Create a callback function that prints at specific time points
            function time_report_cb(u, t, integrator)
                return false  # continue integration
            end
            
            # Setup callbacks - only use the discrete time callback
            if length(all_report_times) > 0
                cb = PresetTimeCallback(all_report_times, time_report_cb)
                
                # Solve to final time with reporting at specific times
                solve!(integrator, 
                    callback=cb, 
                    save_everystep=false, 
                    tstops=all_report_times)
            else
                # If no report times, just integrate directly
                solve!(integrator, save_everystep=false)
            end
            """)
        except Exception as e:
            logging.error(f"Exception during integration: {e}")
            raise
        
        # Update time with proper type
        self.t = float(self.jl.seval("convert(Float64, integrator.t)"))
        
        # Get updated solution with proper type
        self.y = np.array(self.jl.seval("convert(Array{Float64}, Array(integrator.u))"))
        # julia_array = self.jl.seval("convert(Array{Float64}, Array(integrator.u))")
        # self.y = np.array(julia_array, dtype=float)

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
    save_data(fl,'data',ext_flag=False,x=x,y=y,kap=kap,C=C,nu=nu,D=D)

save_data(fl,'params',ext_flag=False,C=C,kap=kap,nu=nu,D=D,Lx=Lx,Ly=Ly,Npx=Npx, Npy=Npy)
#initialize with .view(dtype=float): a+bi becomes [a,b]
r=Gensolver('julia.Dopri8',rhs,t0,zk.view(dtype=float),t1,fsave=save_callback,fshow=fshow,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,rtol=rtol,atol=atol)

try:
    print(f"Starting simulation: t0={t0:.2f}, t1={t1:.2f}")
    # Set a timeout based on expected simulation time
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(3600)  # 1 hour timeout for production runs
    
    r.run()
    print("Simulation completed successfully")
except TimeoutError as e:
    logging.error(f"Timeout: {e}")
except Exception as e:
    logging.error(f"Error during simulation: {str(e)}")
finally:
    signal.alarm(0)  # Cancel the alarm
    fl.close()
    print("Output file closed")