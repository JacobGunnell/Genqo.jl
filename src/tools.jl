module tools

using DocStringExtensions

"""
$TYPEDSIGNATURES

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

function wick_out(coef, moment_vector, Anv)
    # Iterate over Wick partitions
    coeff_sum = 0.0
    for partition in wick_partitions[length(moment_vector)]
        sum_factor = 1.0
        for (i,j) in partition
            sum_factor *= Anv[moment_vector[i], moment_vector[j]]
        end
        coeff_sum += sum_factor
    end
    return coeff_sum * coef
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
