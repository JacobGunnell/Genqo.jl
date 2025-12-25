module tools

using LinearAlgebra
using Nemo
using BlockDiagonals, BlockArrays
using PythonCall


struct ZALMParams
    mean_photon::Float64
    schmidt_coeffs::Vector{Float64}
    detection_efficiency::Float64
    bsm_efficiency::Float64
    outcoupling_efficiency::Float64
    dark_counts::Float64
    visibility::Float64
end

ZALMParams(params_py::Py) = ZALMParams(
    pyconvert(Float64, params_py.mean_photon),
    pyconvert(Vector{Float64}, params_py.schmidt_coeffs),
    pyconvert(Float64, params_py.detection_efficiency),
    pyconvert(Float64, params_py.bsm_efficiency),
    pyconvert(Float64, params_py.outcoupling_efficiency),
    pyconvert(Float64, params_py.dark_counts),
    pyconvert(Float64, params_py.visibility)
)

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

function W(C::Nemo.Generic.MPoly{Nemo.ComplexFieldElem}, Ainv::Matrix{ComplexF64})
    elm = zero(Float64)
    n_vars = nvars(parent(C))
    for (mon, coeff) in zip(monomials(C), coefficients(C))
        elm += wick_out(ComplexF64(coeff), [i for i in 1:n_vars if exponent(mon, 1, i) == 1], Ainv)
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
Calculate the K function portion of the A matrix.
"""
function k_function_matrix(covariance_matrix::Matrix{Float64})
    Γ = covariance_matrix + (1/2)*I
    sz = size(Γ)[1] ÷ 2
    Γinv = BlockArray(inv(Γ), [sz,sz], [sz,sz])

    A = Γinv[Block(1,1)]
    C = Γinv[Block(1,2)]
    Cᵀ = Γinv[Block(2,1)]
    B = Γinv[Block(2,2)]

    BB = (1/2)*mortar(reshape([
        A+(im/2)*(C+Cᵀ), C-(im/2)*(A-B),
        Cᵀ-(im/2)*(A-B), B-(im/2)*(C+Cᵀ)
    ], 2, 2))

    return Matrix(BlockDiagonal([BB, conj(BB)]))
end

end # module
