module spdc

using DocStringExtensions
using BlockDiagonals

import ..tmsv
import ..tools

"""
$TYPEDSIGNATURES

Calculate the covariance matrix of the SPDC source
"""
function covariance_matrix(mean_photon::Float64)
    covar = begin
        tmsv_covar = tmsv.covariance_matrix(mean_photon)
        BlockDiagonal([tmsv_covar, tmsv_covar])
    end

    perm_indices = [1,2,7,8,5,6,3,4]
    perm_matrix = tools.permutation_matrix(perm_indices)
    return perm_matrix * covar * perm_matrix'
end

end # module
