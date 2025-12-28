module tools

using LinearAlgebra
using Nemo
using BlockDiagonals
using ThreadsX
using ProgressMeter


"""
    sweep(func::Function, param_ranges...::AbstractVector; parallel=false, progress=false, desc="Sweeping")

Sweep a function `func` over one or more parameter ranges. For multiple ranges, 
computes the Cartesian product and applies the function to all combinations.

# Parameters
- `func`: Function that takes parameter value(s) and returns a result
- `param_ranges`: One or more vectors of parameter values to sweep over
- `parallel`: Whether to use multi-threading (default: false)
- `progress`: Whether to show a progress bar (default: false)
- `desc`: Description for the progress bar (default: "Sweeping")

# Returns
Array of results with dimensions matching the input parameter ranges.
For a single range, returns a 1D array. For multiple ranges, returns a 
multi-dimensional array where dimension `i` corresponds to `param_ranges[i]`.

# Examples
```julia
# Single parameter sweep
results = tools.sweep([1e-3, 1e-2, 1e-1]) do μ
    tmsv.probability_success(μ, 0.9)
end
# results is a 3-element Vector

# Two parameter sweep (Cartesian product)
results = tools.sweep([1e-3, 1e-2, 1e-1], [0.8, 0.9, 1.0]) do μ, η
    tmsv.probability_success(μ, η)
end
# results is a 3×3 Matrix

# Three parameter sweep with parallelization and progress bar
results = tools.sweep([1e-3, 1e-2], [0.8, 0.9], [0.5, 0.6, 0.7]; parallel=true, progress=true) do μ, η, ζ
    complex_calculation(μ, η, ζ)
end
# results is a 2×2×3 Array
```
"""
function sweep(func::Function, param_ranges::AbstractVector...; parallel=false, progress=false, desc="Sweeping")
    # Create Cartesian product of all parameter ranges
    param_grid = collect(Iterators.product(param_ranges...))
    
    # Apply function to each combination, unpacking the tuple of parameters
    if progress
        if parallel
            results = @showprogress desc ThreadsX.map(params -> func(params...), param_grid)
        else
            results = @showprogress desc map(params -> func(params...), param_grid)
        end
    else
        if parallel
            results = ThreadsX.map(params -> func(params...), param_grid)
        else
            results = map(params -> func(params...), param_grid)
        end
    end
    
    return results
end

"""
Precompute Wick partitions (perfect pairings) of 1:n
Each partition is a Vector of (i, j) pairs (as Tuples)
"""
function _wick_partitions(n::Int)
    @assert iseven(n) "n must be even"
    
    result = Vector{Vector{Tuple{Int,Int}}}()  # will hold all partitions
    
    # Recursive helper
    function backtrack(remaining::Vector{Int}, current::Vector{Tuple{Int,Int}})
        if isempty(remaining)
            # Found a complete pairing
            push!(result, copy(current))
            return
        end
        
        # Always take the smallest remaining index to avoid duplicates
        i = remaining[1]
        
        # Try pairing i with each other remaining element
        for k in 2:length(remaining)
            j = remaining[k]
            
            # Build the next "remaining" list without i and j
            next_remaining = [remaining[2:k-1]; remaining[k+1:end]]
            
            push!(current, (i, j))
            backtrack(next_remaining, current)
            pop!(current)
        end
    end
    
    backtrack(collect(1:n), Tuple{Int,Int}[])
    return result
end
const wick_partitions = Dict(n => _wick_partitions(n) for n in (0, 2, 4, 6, 8)) # Precompute for n=0,2,4,6,8

function wick_out(coef::ComplexF64, moment_vector::Vector{Int}, Ainv::Matrix{ComplexF64})
    # Iterate over Wick partitions
    coeff_sum = zero(ComplexF64)
    for partition in wick_partitions[length(moment_vector)]
        sum_factor = one(ComplexF64)
        for (i,j) in partition
            sum_factor *= Ainv[moment_vector[i], moment_vector[j]]
        end
        coeff_sum += sum_factor
    end
    return coeff_sum * coef
end

# Turn a polynomial C into a reusable list of (coef::ComplexF64, idxs::Vector{Int})
function extract_W_terms(C::Nemo.Generic.MPoly{Nemo.ComplexFieldElem})
    n_vars = nvars(parent(C))
    terms = Vector{Tuple{ComplexF64, Vector{Int}}}()

    for (mon, coeff) in zip(monomials(C), coefficients(C))
        idxs = Int[]
        sizehint!(idxs, 8)
        @inbounds for i in 1:n_vars
            if exponent(mon, 1, i) == 1
                push!(idxs, i)
            end
        end
        push!(terms, (ComplexF64(coeff), idxs))
    end
    return terms
end

function W(C::Nemo.Generic.MPoly{Nemo.ComplexFieldElem}, Ainv::Matrix{ComplexF64})
    elm = zero(ComplexF64)
    n_vars = nvars(parent(C))
    for (mon, coeff) in zip(monomials(C), coefficients(C))
        elm += wick_out(ComplexF64(coeff), [i for i in 1:n_vars if exponent(mon, 1, i) == 1], Ainv)
    end
    return elm
end

# Fast evaluator that avoids Nemo work (uses precompiled terms of specific polynomials)
function W(terms::Vector{Tuple{ComplexF64, Vector{Int}}}, Ainv::Matrix{ComplexF64})
    elm = zero(ComplexF64)
    @inbounds for (coef, idxs) in terms
        elm += wick_out(coef, idxs, Ainv)
    end
    return elm
end

function permutation_matrix(permutations::Vector{Int})
    n = length(permutations)
    P = zeros(n, n)
    for i in 1:n
        P[i, permutations[i]] = 1
    end
    return P
end

"""
Reorder covariance matrix from qpqp to qqpp ordering
"""
function reorder(covariance_matrix::Union{Matrix{Float64}, BlockDiagonal{Float64, Matrix{Float64}}})
    sz = size(covariance_matrix)[1]
    perm_matrix = permutation_matrix([1:2:sz; 2:2:sz])
    return perm_matrix * covariance_matrix * perm_matrix'
end

"""
A slightly faster version of k_function_matrix that avoids creating intermediate BlockArrays.
This was generated by unrolling the above function using ChatGPT. Not a huge speedup, so
possibly not worth maintaining separately.
    Benchmark Results:
    k_function_matrix: 5.395 μs → 2.485 μs
    allocs: 56.9 KiB → 32.8 KiB
"""
function k_function_matrix(covariance_matrix::Matrix{Float64})
    Γ = covariance_matrix + (1/2)*I

    # Invert Γ via LU (same numerical result as inv(Γ), but lets us reuse LU storage)
    F = lu!(Γ)            # factors in-place
    Γinv = inv(F)         # 16×16 Float64

    sz = size(Γinv, 1) ÷ 2
    n = 2sz

    # Views of Γinv blocks (Float64)
    A  = @view Γinv[1:sz,    1:sz]
    C  = @view Γinv[1:sz,    sz+1:n]
    Cᵀ = @view Γinv[sz+1:n,  1:sz]
    B  = @view Γinv[sz+1:n,  sz+1:n]

    # Build BB (16×16 ComplexF64) without intermediates
    BB = Matrix{ComplexF64}(undef, n, n)

    @inbounds for j in 1:sz, i in 1:sz
        a  = A[i,j]
        b  = B[i,j]
        c  = C[i,j]
        ct = Cᵀ[i,j]

        # handy subexpressions
        csum = c + ct
        abd  = a - b

        # BB block entries (each multiplied by 1/2 overall)
        BB[i,     j     ] = 0.5*a  + (im/4)*csum  # (1/2)*(A  + (i/2)(C+Ct))
        BB[i,     j+sz  ] = 0.5*c  - (im/4)*abd   # (1/2)*(C  - (i/2)(A-B))
        BB[i+sz,  j     ] = 0.5*ct - (im/4)*abd   # (1/2)*(Ct - (i/2)(A-B))
        BB[i+sz,  j+sz  ] = 0.5*b  - (im/4)*csum  # (1/2)*(B  - (i/2)(C+Ct))
    end

    # Return block diagonal [BB, conj(BB)] as a plain 32×32 matrix
    K = zeros(ComplexF64, 2n, 2n)
    @inbounds for j in 1:n, i in 1:n
        v = BB[i,j]
        K[i,   j  ] = v
        K[i+n, j+n] = conj(v)
    end

    return K
end

export sweep, wick_out, W, extract_W_terms, permutation_matrix, reorder, k_function_matrix

end # module
