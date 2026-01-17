module tools

using LinearAlgebra
using Nemo
using BlockDiagonals


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

"""
    extract_W_terms(C::Nemo.Generic.MPoly{Nemo.ComplexFieldElem})

Precompile a Nemo polynomial into a reusable list of contraction terms for `W`.

The ZALM/SPDC code builds certain moment polynomials symbolically (in the `q/p` variables) and then
evaluates them many times numerically. This function converts the polynomial into a vector of
`(coef, idxs)` where:
- `coef` is the complex coefficient of a monomial, and
- `idxs` are the variable indices that appear with exponent 1 in that monomial.

This avoids repeated Nemo monomial parsing and exponent scanning during hot loops.

Assumption (matched to current usage): moment polynomials are multilinear in the variables used for Wick
evaluation (exponents are 0/1 for the variables of interest).

# Parameters
- C : Nemo multivariate polynomial over `ComplexField`

# Returns
`Vector{Tuple{ComplexF64, Vector{Int}}}` suitable for `W(terms, Ainv)`.
"""
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

"""
    W(C::Nemo.Generic.MPoly{Nemo.ComplexFieldElem}, Ainv::Matrix{ComplexF64})

Evaluate a (symbolic) moment polynomial by Wick contraction against `Ainv`.

This expands `C` into monomials and, for each monomial, collects the indices of variables present
(exponent 1), then calls `wick_out` to perform the sum over pairings. This is the general (slower) path
used when terms have not been precompiled.

# Parameters
- C    : Moment polynomial to evaluate (symbolic)
- Ainv : Inverse A-matrix providing the contraction kernel

# Returns
Complex value of the Gaussian moment implied by `C` and `Ainv`.
"""
function W(C::Nemo.Generic.MPoly{Nemo.ComplexFieldElem}, Ainv::Matrix{ComplexF64})
    elm = zero(ComplexF64)
    n_vars = nvars(parent(C))
    for (mon, coeff) in zip(monomials(C), coefficients(C))
        elm += wick_out(ComplexF64(coeff), [i for i in 1:n_vars if exponent(mon, 1, i) == 1], Ainv)
    end
    return elm
end

"""
    W(terms::Vector{Tuple{ComplexF64, Vector{Int}}}, Ainv::Matrix{ComplexF64})

Fast Wick evaluator for precompiled moment terms.

This is the hot-path used by SPDC/ZALM fidelity and probability calculations. It skips Nemo entirely:
each `(coef, idxs)` term represents one monomial, and `wick_out` performs the contraction.

# Parameters
- terms: Output of `extract_W_terms` for a specific moment polynomial
- Ainv : Inverse A-matrix providing the contraction kernel

# Returns
Complex value of the contracted moment.
"""
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
    k_function_matrix(covariance_matrix::Matrix{Float64})

Construct the complex-valued K-matrix used to form the Gaussian contraction matrix `A`.

Given a physical covariance matrix (in qqpp ordering), this function forms Γ = cov + (1/2)I, computes
Γ⁻¹, and then builds the complex block matrix (and its conjugate block) used in the ZALM/SPDC moment
formalism. Downstream code forms `A = K + G`, where `G` encodes loss / measurement modeling, and then
uses `A⁻¹` as the contraction kernel for Wick evaluation via `W`.

Implementation note: This version is an unrolled/optimized construction that avoids intermediate block
arrays and reuses an LU factorization for Γ⁻¹.

# Parameters
- covariance_matrix : Real covariance matrix in qqpp ordering

# Returns
A `ComplexF64` matrix `K` (block diagonal `[BB, conj(BB)]`) suitable for `A = K + loss_matrix`.
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

export wick_out, W, extract_W_terms, permutation_matrix, reorder, k_function_matrix

end # module
