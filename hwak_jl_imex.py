#%% Import libraries

import numpy as np
import gc
import os
from modules.mlsarray_cpu import mlsarray,slicelist,init_kspace_grid
from modules.mlsarray_cpu import irft2 as original_irft2, rft2 as original_rft2, irft as original_irft, rft as original_rft
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

solver='jl.KenCarp3'
output = 'out_'+solver.replace('.','_')+'_kap_' + f'{kap:.1f}'.replace('.', '_') + '_C_' + f'{C:.1f}'.replace('.', '_') + '.h5'

# All times needs to be in float for the solver
dtstep,dtshow,dtsave=0.05,1.0,1.0
gamma_time=50/gammax(ky0, kap, C)
t0,t1=0.0,int(round(gamma_time/dtstep))*dtstep
rtol,atol=1e-10,1e-12 
wecontinue=False

w=10.0
phik=1e-3*np.exp(-lkx**2/2/w**2-lky**2/w**2)*np.exp(1j*2*np.pi*np.random.rand(lkx.size).reshape(lkx.shape))
nk=1e-3*np.exp(-lkx**2/w**2-lky**2/w**2)*np.exp(1j*2*np.pi*np.random.rand(lkx.size).reshape(lkx.shape))
zk=np.hstack((phik,nk))

del lkx,lky,xl,yl
gc.collect()

#%% Define functions and classes

irft2 = partial(original_irft2, Npx=Npx, Npy=Npy, sl=sl)
rft2 = partial(original_rft2, sl=sl)
irft = partial(original_irft, Npx=Npx, Nx=Nx)
rft = partial(original_rft, Nx=Nx)

# def save_last(t,y,fl):
#     zk = np.array(y, copy=False)
#     save_data(fl,'last',ext_flag=False,zk=zk,t=t)

def save_callback(t, y):
    # Ensure y is a proper numpy array
    zk = np.array(y, copy=False)
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
    zk = np.array(y, copy=False)
    phik, nk = zk[:int(zk.size/2)], zk[int(zk.size/2):]
    kpsq = kx**2 + ky**2
    dyphi = irft2(1j*ky*phik)             
    n = irft2(nk)

    Gam = -np.mean(n*dyphi)
    Ktot, Kbar = np.sum(kpsq*abs(phik)**2), np.sum(abs(kx[slbar] * phik[slbar])**2)
    
    # Print without elapsed time
    print(f"Gam={Gam:.3g}, Ktot={Ktot:.3g}, Kbar/Ktot={Kbar/Ktot*100:.1f}%")

def rhs_nonstiff(dy, y, p, t):
    """Non-stiff part of the RHS (advection, linear coupling)"""
    zk = np.array(y, copy=False)
    dzkdt = np.array(dy, copy=False)
    
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

    # Add the linear terms on non-zonal modes
    dphikdt[:] += (-C*(phik-nk)/kpsq)*sigk
    dnkdt[:] += (-kap*1j*ky*phik + C*(phik-nk))*sigk

def rhs_stiff(dy, y, p, t):
    """Stiff part of the RHS (viscosity terms)"""
    zk = np.array(y, copy=False)
    dzkdt = np.array(dy, copy=False)
    
    # Split zk into phik and nk
    phik, nk = zk[:int(zk.size/2)], zk[int(zk.size/2):]
    
    # Split dzkdt into dphikdt and dnkdt
    dphikdt, dnkdt = dzkdt[:int(zk.size/2)], dzkdt[int(zk.size/2):]
    
    kpsq = kx**2 + ky**2
    sigk = np.sign(ky) # zero for ky=0, 1 for ky>0
    
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

        # Initialize Julia environment
        jl.seval("using OrdinaryDiffEq")
        jl.seval("using LinearAlgebra")
        jl.seval("using DiffEqCallbacks") 

        # Pass the python RHS functions to Julia
        jl.py_f_nonstiff = f_nonstiff
        jl.py_f_stiff = f_stiff
        
        # Pass initial values to Julia
        jl.py_y0 = y0
        jl.py_t0 = t0
        jl.py_t1 = t1

        jl.seval("""
        # Add a timer to track computation time
        ct = time()
        """)

        # Define wrappers in Julia that call the Python functions
        jl.seval("""
        function nonstiff_wrapper(du, u, p, t)
            py_f_nonstiff(du, u, nothing, t)
            return nothing
        end
        
        function stiff_wrapper(du, u, p, t)
            py_f_stiff(du, u, nothing, t)
            return nothing
        end
        """)

        # Set up the Julia split ODE problem
        jl.seval("""
        # Create the split ODE problem
        prob = SplitODEProblem(nonstiff_wrapper, stiff_wrapper, py_y0, (py_t0, py_t1))
        """)
        
        self.problem = jl.prob  # Store the Julia problem

        # Choose appropriate IMEX solver
        # Disable automatic differentiation since they can't handle complex numbers
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
        # Note: The 'KenCarp' family implements Kennedy & Carpenter's IMEX methods
        # for stiff + non-stiff split problems, ideal for advection-diffusion systems

        # Set up solver parameters
        jl.rtol_val = kwargs.get('rtol', 1e-7)
        jl.atol_val = kwargs.get('atol', 1e-9)
        jl.dtmax_val = dtstep
        jl.solver_str = solver_name
        jl.kwargs_py = kwargs
        
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
                
                elapsed = time() - ct
                print("t=$(round(t, digits=3)), $(round(elapsed, digits=3)) secs elapsed.", " ")
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
        save_times = sort(unique(round.(save_times, digits=3)))
        
        # Generate show times (ensure we include t0 and t1)
        show_times = vcat([start_time], collect(ceil(start_time/dtshow_val)*dtshow_val:dtshow_val:end_time), [end_time])
        show_times = sort(unique(round.(show_times, digits=3))) 
        
        # Create the callbacks
        save_callback = PresetTimeCallback(save_times, save_cb)
        show_callback = PresetTimeCallback(show_times, show_cb)
        
        # Combine callbacks
        callback_set = CallbackSet(save_callback, show_callback)
        
        # Create the solver with properly named kwargs
        solver = eval(Meta.parse(solver_str))
        
        kwargs = Dict{Symbol, Any}()
        for (k, v) in pairs(kwargs_py)
            kwargs[Symbol(k)] = v
        end

        # Use solve to create the solution and integrator with callbacks
        integrator = init(prob, solver, 
                    dtmax=dtmax_val,
                    save_everystep=false,
                    dense=false,
                    callback=callback_set;
                    kwargs...)
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
            all_times = round.(all_times, digits=3)  
            
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
r=Gensolver(solver,rhs_nonstiff,rhs_stiff,t0,zk,t1,fsave=save_callback,fshow=fshow,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,reltol=rtol,abstol=atol)

try:  
    r.run()
except Exception as e:
    print(f"Error during simulation: {str(e)}")
finally:
    # No need to cancel alarm
    fl.close()
    print("Output file closed")