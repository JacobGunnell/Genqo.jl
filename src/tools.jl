module tools

using LinearAlgebra



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
const wick_partitions = Dict(n => _wick_partitions(n) for n in (2, 4, 6, 8)) # Precompute for n=2,4,6,8

function wick_out(coef::ComplexF64, moment_vector::Vector{Int}, Anv::Matrix{ComplexF64})
    # Iterate over Wick partitions
    coeff_sum = zero(Float64)
    for partition in wick_partitions[length(moment_vector)]
        sum_factor = one(Float64)
        for (i,j) in partition
            sum_factor *= Anv[moment_vector[i], moment_vector[j]]
        end
        coeff_sum += sum_factor
    end
    return coeff_sum * coef
end


# function wick_coupling_indices(term_factors, bv_map::Dict)
#     coef = 1
#     mv_idx = Int[]

#     for f in term_factors
#         if f isa Number
#             coef *= ComplexF64(f)
#         else
#             idx = bv_map[f]
#             push!(mv_idx, idx)
#         end
#     end

#     return coef, mv_idx
# end


# function W(Cni, Amat, bv_map)
#     Anv = inv(Amat)
#     total = 0

#     for term in Cni
#         coef, mv = wick_coupling_indices(term, bv_map)
#         total += wick_out(coef, mv, Anv)
#     end

#     return total
# end
"""
    W(C, Amat) -> ComplexF64

Compute the Wick-contracted expectation/value associated with a polynomial `C`
using the inverse covariance/coupling matrix `Anv = inv(Amat)`.

This wrapper iterates over the terms of
`C` (monomials and their coefficients), converts each monomial into a
`moment_vector` (a list of generator indices, repeated by exponent), and then
sums the Wick pairings via `wick_out`.

Assumptions / requirements:
  - `C` lives in a polynomial ring where `monomials(C)` and `coefficients(C)`
    are defined (e.g., Nemo/AbstractAlgebra).
  - The ordering of the ring generators matches the indexing convention used
    to access `Anv` (i.e., generator #i corresponds to `Anv[i, :]` / `Anv[:, i]`).
  - Each monomial's total degree must be even (Wick pairing), and `wick_partitions`
    must be available for that degree (currently precomputed for n ∈ {2,4,6,8}).

Notes:
  - Terms with exponent > 1 are expanded by repeating the corresponding index,
    so `x^3*y` becomes `[x,x,x,y]` before Wick pairing.
  - For performance, assertions are gated behind a debug flag.

See also: `wick_out`, `wick_partitions`, `monomial_to_moment_vector`.
"""
DEBUG_WICK = false
function W(C, Amat)
    Anv = inv(Amat)
    total = zero(ComplexF64)

    # number of generators/variables in the ring
    nvars = length(gens(parent(C))) # gens(parent(C)) is a list of all variables in the ring

    for (mon, coeff) in zip(monomials(C), coefficients(C))
        mv = monomial_to_moment_vector(mon, nvars)

        # optional sanity checks (helps catch weird terms early)
        if (DEBUG_WICK)
            @assert iseven(length(mv)) "Wick needs an even number of factors; got $(length(mv))"
            @assert haskey(wick_partitions, length(mv)) "No precomputed wick_partitions for n=$(length(mv))"
        end
    
        total += wick_out(ComplexF64(coeff), [i for i in 1:nvars if exponent(mon, 1, i) == 1], Anv)
    end

    return total
end




function permutation_matrix(permutations::Vector{Int})
    n = length(permutations)
    P = zeros(n, n)
    for i in 1:n
        P[i, permutations[i]] = 1
    end
    return P
end

end # module
