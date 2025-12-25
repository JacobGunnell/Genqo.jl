module tmsv

using BlockArrays

using ..tools: GenqoParams


"""
    covariance_matrix(μ::Float64)

Construct the covariance matrix for a TMSV state.

# Parameters
- μ : The mean photon number of the TMSV state

# Returns
The covariance matrix for the TMSV state, in the qpqp ordering
"""
function covariance_matrix(μ::Float64)
    A = [
        1+2μ 0;
        0 1+2μ;
    ]
    B = [
        2sqrt(μ*(μ+1)) 0;
        0 -2sqrt(μ*(μ+1));
    ]
    return (1/2)*mortar(reshape([
        A, B,
        B, A
    ], 2, 2))
end
covariance_matrix(params::GenqoParams) = covariance_matrix(params.mean_photon)

end # module
