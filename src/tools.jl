module tools

using LinearAlgebra
using Nemo
using BlockDiagonals
using PythonCall


Sweepable{T} = Union{T, AbstractRange{<:T}, AbstractVector{<:T}} where T

abstract type GenqoBase end

# Enable broadcasting over Genqo objects with sweep parameters
# TODO: give explicit sweep type like in Python wrapper so that Genqo objects can have mixed sweep/non-sweep (1+)-dimensional fields (e.g. schmidt coefficients)
# For now, we assume all fields are either single values or StepRangeLen sweeps of single values
function Base.length(gq::T) where T<:GenqoBase
    len = 1
    for field in fieldnames(T)
        len *= length(getproperty(gq, field))
    end
    return len
end

# TODO: allow iteration to preserve the shape of cartesian product over fields, as in tmsv.probability_success.(tmsv.TMSV(0.01:0.01:10, 0.7:0.1:0.9))
# Currently, we flatten to 1D iteration, so above returns a 3000-element Vector instead of a 1000x3 Matrix
function Base.iterate(gq::T, state=1) where T<:GenqoBase
    total_len = length(gq)
    
    if state > total_len
        return nothing
    end
    
    # Get field names and convert each field to a collection
    fields = fieldnames(T)
    field_collections = map(fields) do field
        val = getproperty(gq, field)
        val isa Union{AbstractRange, AbstractVector} ? val : [val]
    end
    
    # Calculate the multi-dimensional index for the Cartesian product
    field_lengths = map(length, field_collections)
    indices = zeros(Int, length(fields))
    
    # Convert linear state to multi-dimensional indices
    remainder = state - 1
    for i in length(fields):-1:1
        indices[i] = remainder % field_lengths[i] + 1
        remainder ÷= field_lengths[i]
    end
    
    # Extract values at the calculated indices
    field_values = map(enumerate(field_collections)) do (i, collection)
        collection[indices[i]]
    end
    
    return T(field_values...), state + 1
end

"""
Convert a Python object to a Julia type, using custom conversion for sweep types.
"""
function _pyconvert_sweepable(T::Type, py_obj::Py)
    name = pyconvert(String, py_obj.__class__.__name__)
    if name == "linsweep"
        return range(
            pyconvert(T, py_obj.start), 
            stop=pyconvert(T, py_obj.stop), 
            length=pyconvert(Int, py_obj.length)
        )
    elseif name == "logsweep"
        return logrange(
            pyconvert(T, py_obj.start), 
            pyconvert(T, py_obj.stop), 
            length=pyconvert(Int, py_obj.length)
        )
    else
        return pyconvert(T, py_obj)
    end
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

export Sweepable, GenqoBase, wick_out, W, extract_W_terms, permutation_matrix, reorder, k_function_matrix

end # module
