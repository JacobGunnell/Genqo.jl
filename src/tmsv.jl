module tmsv

using BlockArrays
using Nemo
using LinearAlgebra

import ..tools
using ..tools: GenqoParams


# Global canonical position and momentum variables
const mds = 2 # Number of modes for our system

_qai = ["qa$i" for i in 1:mds]
_pai = ["pa$i" for i in 1:mds]
_qbi = ["qb$i" for i in 1:mds]
_pbi = ["pb$i" for i in 1:mds]
all_qps = hcat(_qai, _pai, _qbi, _pbi)
CC = ComplexField()
i = onei(CC) # Imaginary unit in CC ring
R, generators = polynomial_ring(CC, all_qps)
(qai, pai, qbi, pbi) = (generators[:,i] for i in 1:4)

# Define the alpha and beta vectors
α = (qai + i .* pai) / sqrt(2)
β = (qbi - i .* pbi) / sqrt(2)

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
    return Matrix(
        (1/2)*mortar(reshape([
            A, B,
            B, A
        ], 2, 2))
    )
end
covariance_matrix(params::GenqoParams) = covariance_matrix(params.mean_photon)

"""
Calculates the portion of the A matrix that arrises due to incorporating loss
"""
function loss_matrix_pgen(ηᵈ::Float64)
    G = zeros(ComplexF64, 8, 8)

    for i in 1:2
        G[i, i+4] = ηᵈ - 1
        G[i, i+6] = -im*(ηᵈ - 1)
        G[i+2, i+4] = im*(ηᵈ - 1)
        G[i+2, i+6] = ηᵈ - 1
    end

    return (G + transpose(G) + I) / 2
end
loss_matrix_pgen(params::GenqoParams) = loss_matrix_pgen(params.detection_efficiency)

function moment_vector(n::Int)
    (α[1]*α[2])^n / factorial(n) * (β[1]*β[2])^n / factorial(n)
end

"""
    probability_success(μ::Float64, ηᵈ::Float64)

Calculate the probability of photon-photon state generation with the given parameters.

# Parameters
- μ : The mean photon number of the TMSV state
- ηᵈ : Detection efficiency

# Returns
Probability of successful photon-photon state generation
"""
function probability_success(μ::Float64, ηᵈ::Float64)
    covar_qpqp = covariance_matrix(μ)

    # Reorder qpqp → qqpp
    perm_matrix = tools.permutation_matrix([1:2:3; 2:2:4])
    covar_qqpp = perm_matrix * covar_qpqp * perm_matrix'

    A = tools.k_function_matrix(covar_qqpp) + loss_matrix_pgen(ηᵈ)
    Ainv = inv(A)
    Γ = covar_qqpp + (1/2)*I
    detΓ = det(Γ)

    N1 = ηᵈ^2
    D1 = sqrt(det(A))
    D2 = detΓ^(1/4)
    D3 = conj(detΓ)^(1/4)
    Coef = N1/(D1*D2*D3)

    C = moment_vector(1)

    return real(Coef * tools.W(C, Ainv))
end
probability_success(params::GenqoParams) = probability_success(params.mean_photon, params.detection_efficiency)

end # module
