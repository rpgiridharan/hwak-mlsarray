using CUDA
using FFTW
using LinearAlgebra
using HDF5
using OrdinaryDiffEq
using DiffEqCallbacks
using Printf

# Import custom modules
include("modules/mlsarray_gpu.jl")
include("modules/gamma.jl")

# Define parameters
Npx, Npy = 256, 256
Nx, Ny = 2*floor(Int, Npx/3), 2*floor(Int, Npy/3)
Lx, Ly = 16π, 16π
dkx, dky = 2π/Lx, 2π/Ly
sl = SliceList(Nx, Ny)
lkx, lky = init_kspace_grid(sl)
kx, ky = lkx * dkx, lky * dky
slbar = (div(Ny,2)-1):div(Ny,2):(div(Ny,2)*div(Nx,2)-1) # Slice to access only zonal modes
ky0 = ky[1:div(Ny,2)-1] # ky at just kx=0

# Construct real space with padded resolution
xl, yl = range(0, Lx, length=Npx), range(0, Ly, length=Npy)
x = [i for i in xl, j in yl]
y = [j for i in xl, j in yl]

# Physical parameters
kap = 1.0
C = 1.0
# Convert ky0 to CPU for computation
ky0_cpu = Array(ky0)
nu = 1e-3 * kymax(ky0_cpu, 1.0, 1.0)^4 / kymax(ky0_cpu, kap, C)^4
D = 1e-3 * kymax(ky0_cpu, 1.0, 1.0)^4 / kymax(ky0_cpu, kap, C)^4 

solver = "CKLLSRK65_4M_4R"
# solver = "CKLLSRK43" # Uncomment to use a low-storage method
output = "out_$(replace(solver, "." => "_"))_gpu_kap_$(replace(@sprintf("%.1f", kap), "." => "_"))_C_$(replace(@sprintf("%.1f", C), "." => "_")).h5"

# All times need to be in float for the solver
dtstep, dtshow, dtsave = 0.1, 1.0, 1.0
gamma_time = 50/gammax(ky0_cpu, kap, C)
t0, t1 = 0.0, Int(round(gamma_time/dtstep))*dtstep
rtol, atol = 1e-6, 1e-8
wecontinue = false

w = 10.0
phik = 1e-3 * exp.(-lkx.^2 / 2 / w^2 .- lky.^2 / w^2) .* exp.(im * 2π * CUDA.rand(size(lkx)...))
nk = 1e-3 * exp.(-lkx.^2 / w^2 .- lky.^2 / w^2) .* exp.(im * 2π * CUDA.rand(size(lkx)...))
zk = [phik; nk]

# Define helper functions for FFTs
function irft2(uk)
    return mod_irft2(uk, Npx, Npy, sl)
end

function rft2(u)
    return mod_rft2(u, sl)
end

function irft(vk)
    return mod_irft(vk, Npx, Nx)
end

function rft(v)
    return mod_rft(v, Nx)
end

# Define callback for saving data
function save_callback(zk, t)
    phik, nk = zk[1:div(length(zk),2)], zk[div(length(zk),2)+1:end]
    om = irft2(-phik .* (kx.^2 .+ ky.^2))
    vx = irft2(-im .* ky .* phik)
    vy = irft2(im .* kx .* phik)
    n = irft2(nk)
    Gam = mean(vx .* n, dims=1)
    Pi = mean(vx .* om, dims=1)
    R = mean(vx .* vy, dims=1)
    vbar = mean(vy, dims=1)
    ombar = mean(om, dims=1)
    nbar = mean(n, dims=1)
    save_data(fl, "fields", true, om=Array(om), n=Array(n), t=t)
    save_data(fl, "fluxes", true, Gam=Array(Gam), Pi=Array(Pi), R=Array(R), t=t)
    save_data(fl, "fields/zonal/", true, vbar=Array(vbar), ombar=Array(ombar), nbar=Array(nbar), t=t)
end

function fshow(zk, t)
    phik, nk = zk[1:div(length(zk),2)], zk[div(length(zk),2)+1:end]
    kpsq = kx.^2 .+ ky.^2
    vx = irft2(-im .* ky .* phik)             
    n = irft2(nk)

    Gam = mean(vx .* n)
    Ktot, Kbar = sum(kpsq .* abs.(phik).^2), sum(abs.(kx[slbar] .* phik[slbar]).^2)
    
    # Print without elapsed time
    @printf("Gam=%.3g, Ktot=%.3g, Kbar/Ktot=%.1f%%", Gam, Ktot, Kbar/Ktot*100)
end

function rhs!(dzkdt, zk, p, t)
    """RHS function for Julia's ODE solvers
    
    Arguments:
        dzkdt: output array where derivatives should be stored (CuArray)
        zk: current state vector (CuArray)
        p: parameters (unused)
        t: current time
    """
    # Split zk into phik and nk
    n_half = div(length(zk), 2)
    phik, nk = view(zk, 1:n_half), view(zk, n_half+1:2*n_half)
    
    # Split dzkdt into dphikdt and dnkdt
    dphikdt, dnkdt = view(dzkdt, 1:n_half), view(dzkdt, n_half+1:2*n_half)

    kpsq = kx.^2 .+ ky.^2
    sigk = sign.(ky)  # zero for ky=0, 1 for ky>0

    # Compute all the fields in real space that we need 
    dxphi = irft2(im .* kx .* phik)
    dyphi = irft2(im .* ky .* phik)
    n = irft2(nk)
    
    # Compute the non-linear terms
    dphikdt .= -1 .* (kx .* ky .* rft2(dxphi.^2 .- dyphi.^2) .+ (ky.^2 .- kx.^2) .* rft2(dxphi .* dyphi)) ./ kpsq
    dnkdt .= im .* kx .* rft2(dyphi .* n) .- im .* ky .* rft2(dxphi .* n)

    # Add the linear terms on non-zonal modes
    dphikdt .+= (-C .* (phik .- nk) ./ kpsq) .* sigk
    dnkdt .+= (-kap .* im .* ky .* phik .+ C .* (phik .- nk)) .* sigk

    # Add the hyper viscosity terms on non-zonal modes
    dphikdt .+= -nu .* kpsq.^2 .* phik .* sigk
    dnkdt .+= -D .* kpsq.^2 .* nk .* sigk
end

function save_data(fl, grpname, ext_flag; kwargs...)
    if !haskey(fl, grpname)
        grp = create_group(fl, grpname)
    else
        grp = fl[grpname]
    end
    
    for (key, value) in kwargs
        # Convert CUDA arrays to CPU if needed
        if typeof(value) <: CuArray
            value = Array(value)
        end
        
        if !haskey(grp, string(key))
            if !ext_flag
                grp[string(key)] = value
            else
                if isscalar(value)
                    dset = create_dataset(grp, string(key), datatype(typeof(value)), dataspace((1,)), 
                                          chunk=(1,), max_dims=(typemax(Int),))
                    dset[1] = value
                else
                    dset = create_dataset(grp, string(key), datatype(eltype(value)), 
                                         dataspace((1, size(value)...)), 
                                         chunk=(1, size(value)...), 
                                         max_dims=(typemax(Int), size(value)...))
                    dset[1, :] = value
                end
            end
        elseif ext_flag
            dset = grp[string(key)]
            old_size = size(dset)
            resize!(dset, (old_size[1] + 1, old_size[2:end]...))
            dset[end, :] = value
        end
    end
    flush(fl)
end

function isscalar(x)
    return length(x) == 1 && !isa(x, Array)
end

# Main simulation execution
if wecontinue
    fl = h5open(output, "r+")
    om = read(fl["fields/om"])[end, :]
    n = read(fl["fields/n"])[end, :]
    omk = rft2(CuArray(om))
    nk = rft2(CuArray(n))
    phik = -omk ./ (kx.^2 .+ ky.^2)
    t0 = read(fl["fields/t"])[end]
    zk = [phik; nk]
else
    fl = h5open(output, "w")
    t = t0
    save_data(fl, "data", false; x=Array(x), y=Array(y), kap=kap, C=C, nu=nu, D=D)
end

save_data(fl, "params", false; C=C, kap=kap, nu=nu, D=D, Lx=Lx, Ly=Ly, Npx=Npx, Npy=Npy)

# Create ODE problem
prob = ODEProblem(rhs!, CuArray{ComplexF64}(zk), (t0, t1))

# Create callbacks for saving and displaying data
save_cb = PeriodicCallback((u, t, integrator) -> save_callback(u, t), dtsave)
show_cb = PeriodicCallback((u, t, integrator) -> fshow(u, t), dtshow)

# Create a callback collection
cb = CallbackSet(save_cb, show_cb)

# Select appropriate ODE solver based on solver string
solver_func = eval(Meta.parse(solver))()

# Start time measurement
start_time = time()

# Print initial state
println("Starting simulation at t = $t0")
fshow(zk, t0)

try
    # Solve the ODE problem
    sol = solve(prob, solver_func; 
                dt=dtstep, 
                adaptive=false,
                reltol=rtol, 
                abstol=atol,
                save_everystep=false,
                callback=cb)
    
    # Save final state
    save_callback(sol.u[end], sol.t[end])
    
    # Report completion
    elapsed = time() - start_time
    println("\nSimulation completed in $(round(elapsed, digits=2)) seconds")
    println("Final time: $(sol.t[end])")
    fshow(sol.u[end], sol.t[end])
catch e
    println("Error during simulation: $e")
finally
    # Clean up GPU resources
    CUDA.reclaim()
    GC.gc()
    
    # Close the HDF5 file
    close(fl)
    println("Output file closed and GPU memory released")
end