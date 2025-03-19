using CUDA
using FFTW

struct SliceList
    insl::Vector{Tuple{UnitRange{Int64}, UnitRange{Int64}}}
    shape::Tuple{Int64, Int64}
    shps::Vector{Vector{Int64}}
    Ns::Vector{Int64}
    outsl::Vector{UnitRange{Int64}}
end

function SliceList(Nx::Int, Ny::Int)
    shp = (Nx, Ny)
    
    # For Fourier transforms, the first dimension will be Nx÷2+1 after FFT
    # Matching the Python implementation
    insl = [
        (1:1, 2:div(Ny, 2)),
        (2:div(Nx, 2), 1:div(Ny, 2)),
        ((Nx - div(Nx, 2) + 2):Nx, 2:div(Ny, 2))
    ]
    
    # Calculate shapes of each slice
    shps = [[length(l[j]) for j in 1:length(l)] for l in insl]
    
    # Calculate total elements in each slice
    Ns = [prod(l) for l in shps]
    
    # Calculate output slice ranges
    outsl = [sum(Ns[1:l-1])+1:sum(Ns[1:l]) for l in 1:length(Ns)]
    
    return SliceList(insl, shp, shps, Ns, outsl)
end

# Custom array type that inherits from CuArray
struct MLSArray{T} <: AbstractArray{T, 2}
    data::CuArray{Complex{T}, 2}
    
    # Constructor for creating a new zeroed array
    function MLSArray{T}(Nx::Int, Ny::Int) where T <: AbstractFloat
        # For a real array of size (Nx, Ny), the complex FFT result has dimensions (Nx÷2+1, Ny)
        return new(CuArray{Complex{T}}(zeros(Complex{T}, Nx÷2+1, Ny)))
    end
    
    # Alternative constructor accepting an existing CuArray
    function MLSArray{T}(data::CuArray{Complex{T}, 2}) where T <: AbstractFloat
        return new(data)
    end
end

# Define the basic array interface for our custom type
Base.size(A::MLSArray) = size(A.data)
Base.getindex(A::MLSArray, I...) = getindex(A.data, I...)
Base.setindex!(A::MLSArray, v, I...) = setindex!(A.data, v, I...)

# Implement specialized getindex for SliceList - fixed type annotation
function Base.getindex(A::MLSArray{T}, sl::SliceList) where T
    return [vec(A.data[i, j]) for (i, j) in sl.insl]
end

# Implement specialized setindex! for SliceList
function Base.setindex!(A::MLSArray{T}, value::Vector, sl::SliceList) where T
    for (idx, (l, j, shp)) in enumerate(zip(sl.insl, sl.outsl, sl.shps))
        i, k = l
        # Extract correct portion of value based on index
        val_portion = value[idx]  # This is the key change - using idx instead of j
        # Reshape to match the expected dimensions
        reshaped = reshape(val_portion, (shp[1], shp[2]))
        A.data[i, k] = reshaped
    end
    return A
end

# FFT methods
function irfft2!(A::MLSArray{T}) where T
    real_view = reinterpret(reshape, T, A.data)
    # Use CUDA's implementation of irfft with proper dimensions
    A_size = size(A.data)
    # FFTW expects the first dimension to match (d >> 1 + 1) where d is output size
    real_view[:, 1:end-2] = real(CUDA.CUFFT.irfft(A.data, A_size[1] * 2 - 2, 1))
end

function rfft2!(A::MLSArray{T}) where T <: AbstractFloat
    real_view = reinterpret(reshape, T, A.data)
    A.data[:] = CUDA.CUFFT.rfft(real_view[:, 1:end-2], 1)
end

function ifftx!(A::MLSArray)
    A.data[:] = CUDA.CUFFT.ifft(A.data, 1)
end

function fftx!(A::MLSArray)
    A.data[:] = CUDA.CUFFT.fft(A.data, 1)
end

# Utility functions
function init_kspace_grid(sl::SliceList)
    Nx, Ny = sl.shape
    
    # Create wavenumber grids (equivalent to Python's np.r_)
    kxl = vcat(0:div(Nx, 2)-1, -div(Nx, 2):-1)
    kyl = 0:div(Ny, 2)
    
    # Create meshgrid
    kx = [kxl[i] for i in 1:length(kxl), j in 1:length(kyl)]
    ky = [kyl[j] for i in 1:length(kxl), j in 1:length(kyl)]
    
    # Extract and concatenate slices
    kx_sliced = vcat([vec(kx[i, j]) for (i, j) in sl.insl]...)
    ky_sliced = vcat([vec(ky[i, j]) for (i, j) in sl.insl]...)
    
    return CuArray(kx_sliced), CuArray(ky_sliced)
end

function mod_irft2(uk::AbstractArray, Npx::Int, Npy::Int, sl::SliceList)
    println("Debug: uk size: $(size(uk)), Npx: $Npx, Npy: $Npy")
    
    # Create a new MLSArray to hold the data
    u = MLSArray{Float64}(Npx, Npy)
    println("Debug: complex_data size: $(size(u.data))")
    
    # Distribute data to the correct slices
    # Need to convert uk to sections first
    uk_total = length(uk)
    
    # Calculate how to split uk based on slice lengths
    start_idx = 1
    uk_sections = []
    
    println("Debug: Number of elements in slices: $(sl.Ns)")
    
    for len in sl.Ns
        end_idx = start_idx + len - 1
        if end_idx > uk_total
            error("UK array length ($uk_total) inconsistent with expected slice sizes ($(sum(sl.Ns)))")
        end
        push!(uk_sections, uk[start_idx:end_idx])
        start_idx = end_idx + 1
    end
    
    # Now set each section to the appropriate slice
    try
        u[sl] = uk_sections
        # Run IRFFT
        irfft2!(u)
        # Return the real data
        return reinterpret(reshape, Float64, u.data)[:, 1:end-2]
    catch e
        println("Error in mod_irft2: $e")
        throw(e)
    end
end

function mod_rft2(u::AbstractArray{T,2}, sl::SliceList) where T
    # Create MLSArray wrapper for the data
    temp_array = MLSArray{Float64}(size(u, 1), size(u, 2))
    # Copy data into the array
    reinterpret(reshape, Float64, temp_array.data)[:, 1:end-2] = u
    # Run FFT
    rfft2!(temp_array)
    # Extract slices and concatenate
    return vcat(temp_array[sl]...)
end

function mod_irft(vk::AbstractVector{T}, Npx::Int, Nx::Int) where T
    v = CuArray{Complex{Float64}}(zeros(Complex{Float64}, div(Npx, 2) + 1))
    v[2:div(Nx, 2)] = vk[:]
    return CUDA.CUFFT.irfft(v, Npx)
end

function mod_rft(v::AbstractVector{T}, Nx::Int) where T
    return CUDA.CUFFT.rfft(v)[2:div(Nx, 2)]
end