module tmsv

using Nemo
using LinearAlgebra
using PythonCall

using ..Genqo: GenqoBase, Sweepable, _pyconvert_sweepable
using ..tools


# TODO: generalize sweep handling to support explicit arrays of sweep parameters and not just StepRangeLen
Base.@kwdef mutable struct TMSV <: GenqoBase
    mean_photon::Sweepable{AbstractFloat} = 1e-2
    detection_efficiency::Sweepable{AbstractFloat} = 1.0
end
Base.convert(::Type{TMSV}, tmsv_py::Py) = TMSV(
    _pyconvert_sweepable(Float64, tmsv_py.mean_photon), 
    _pyconvert_sweepable(Float64, tmsv_py.detection_efficiency),
)

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
    covariance_matrix(μ::Real)

Construct the covariance matrix for a TMSV state.

# Parameters
- μ : The mean photon number of the TMSV state

# Returns
The covariance matrix for the TMSV state, in the qpqp ordering
"""
covariance_matrix(μ::Real) = [
    0.5 + μ        0               sqrt(μ*(μ+1))  0;
    0              0.5 + μ         0              -sqrt(μ*(μ+1));
    sqrt(μ*(μ+1))  0               0.5 + μ        0;
    0              -sqrt(μ*(μ+1))  0              0.5 + μ;
]
covariance_matrix(tmsv::TMSV) = covariance_matrix(tmsv.mean_photon)

"""
Calculates the portion of the A matrix that arrises due to incorporating loss
"""
function loss_matrix_pgen(ηᵈ::Real)
    G = zeros(ComplexF64, 8, 8)

    for i in 1:2
        G[i, i+4] = ηᵈ - 1
        G[i, i+6] = -im*(ηᵈ - 1)
        G[i+2, i+4] = im*(ηᵈ - 1)
        G[i+2, i+6] = ηᵈ - 1
    end

    return (G + transpose(G) + I) / 2
end
loss_matrix_pgen(tmsv::TMSV) = loss_matrix_pgen(tmsv.detection_efficiency)

function moment_vector(n::Int)
    (α[1]*α[2])^n / factorial(n) * (β[1]*β[2])^n / factorial(n)
end

"""
    probability_success(μ::Real, ηᵈ::Real)

Calculate the probability of photon-photon state generation with the given parameters.

# Parameters
- μ : The mean photon number of the TMSV state
- ηᵈ : Detection efficiency

# Returns
Probability of successful photon-photon state generation
"""
function probability_success(μ::Real, ηᵈ::Real)
    # Compute covariance matrix and reorder qpqp → qqpp
    cov = reorder(covariance_matrix(μ))

    A = k_function_matrix(cov) + loss_matrix_pgen(ηᵈ)
    Ainv = inv(A)
    Γ = cov + (1/2)*I
    detΓ = det(Γ)

    N1 = ηᵈ^2
    D1 = sqrt(det(A))
    D2 = detΓ^(1/4)
    D3 = conj(detΓ)^(1/4)
    Coef = N1/(D1*D2*D3)

    C = moment_vector(1)

    return real(Coef * W(C, Ainv))
end
probability_success(tmsv::TMSV) = probability_success(tmsv.mean_photon, tmsv.detection_efficiency)

end # module
