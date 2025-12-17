module tmsv

using DocStringExtensions
using BlockArrays

"""
$TYPEDSIGNATURES

Construct the covariance matrix for a TMSV state
Arguments
    - mean_photon: The mean photon number of the TMSV state
    Output
    - The covariance matrix for the TMSV state, in the qpqp ordering
"""
function covariance_matrix(mean_photon::Float64)
    μ = mean_photon
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

end # module
