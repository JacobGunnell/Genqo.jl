module sigsag

using LinearAlgebra
using BlockDiagonals
using Nemo
using PythonCall

using ..Genqo: GenqoBase, Sweepable, _pyconvert_sweepable
import ..spdc
using ..tools


Base.@kwdef mutable struct SIGSAG <: GenqoBase
    mean_photon::Sweepable{AbstractFloat} = 1e-2
    detection_efficiency::Sweepable{AbstractFloat} = 1.0
    bsm_efficiency::Sweepable{AbstractFloat} = 1.0
    outcoupling_efficiency::Sweepable{AbstractFloat} = 1.0
end
Base.convert(::Type{SIGSAG}, sigsag_py::Py) = SIGSAG(
    _pyconvert_sweepable(Float64, sigsag_py.mean_photon), 
    _pyconvert_sweepable(Float64, sigsag_py.detection_efficiency),
    _pyconvert_sweepable(Float64, sigsag_py.bsm_efficiency),
    _pyconvert_sweepable(Float64, sigsag_py.outcoupling_efficiency),
)

# Global canonical position and momentum variables
const mds = 6 # Number of modes for our system

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

# Symplective matrices that represent 50/50 beamsplitters between the bell state modes
_S35 = begin
    Id2 = Matrix{Float64}(I, 2, 2)
    St35 = [
        1/sqrt(2)   0  1/sqrt(2)  0;
        0           1  0          0;
        -1/sqrt(2)  0  1/sqrt(2)  0;
        0           0  0          1;
    ]
    Matrix(BlockDiagonal([Id2, St35, Id2, St35]))
end
_S46 = begin
    Id2 = Matrix{Float64}(I, 2, 2)
    St46 = [
        1  0           0  0;
        0  1/sqrt(2)   0  1/sqrt(2);
        0  0           1  0;
        0  -1/sqrt(2)  0  1/sqrt(2);
    ]
    Matrix(BlockDiagonal([Id2, St46, Id2, St46]))
end

"""
Calculate the covariance matrix of the single Sagnac source
"""
function covariance_matrix(μ::Real)
    # Expand SPDC covariance matrix to 6 modes by adding vacuum modes
    covar_qpqp = zeros(2*mds, 2*mds)
    covar_qpqp[1:8, 1:8] = spdc.covariance_matrix(μ)
    for i in 9:12
        covar_qpqp[i,i] = 1/2
    end
    
    # Reorder qpqp → qqpp and apply beamsplitters
    covar_qqpp = reorder(covar_qpqp) 
    return _S46 * _S35 * covar_qqpp * _S35' * _S46'
end
covariance_matrix(sigsag::SIGSAG) = covariance_matrix(sigsag.mean_photon)

"""
Calculate the loss portion of the A matrix, specifically when calculating the fidelity.
"""
function loss_bsm_matrix_fid(ηᵗ::Real, ηᵈ::Real)
    G = zeros(ComplexF64, 4*mds, 4*mds)
    η = [ηᵈ, ηᵈ, ηᵗ, ηᵗ, ηᵗ, ηᵗ]

    for i in 1:mds
        G[i,     i+2*mds] = (η[i] - 1)
        G[i,     i+3*mds] = -im*(η[i] - 1)
        G[i+mds, i+2*mds] = im*(η[i] - 1)
        G[i+mds, i+3*mds] = (η[i] - 1)
    end

    return (G + transpose(G) + I) / 2
end
loss_bsm_matrix_fid(sigsag::SIGSAG) = loss_bsm_matrix_fid(sigsag.outcoupling_efficiency, sigsag.detection_efficiency)

"""
Calculate the loss portion of the A matrix, specifically when calculating probability of success.
"""
function loss_bsm_matrix_pgen(ηᵗ::Real, ηᵈ::Real)
    G = zeros(ComplexF64, 4*mds, 4*mds)
    η = [ηᵈ, ηᵈ, ηᵗ, ηᵗ, ηᵗ, ηᵗ]

    for i in 1:mds
        if i in (3,4,5,6)
            G[i,     i+2*mds] = -1
            G[i,     i+3*mds] = im
            G[i+mds, i+2*mds] = -im
            G[i+mds, i+3*mds] = -1
        else
            G[i,     i+2*mds] = (η[i] - 1)
            G[i,     i+3*mds] = -im*(η[i] - 1)
            G[i+mds, i+2*mds] = im*(η[i] - 1)
            G[i+mds, i+3*mds] = (η[i] - 1)
        end
    end

    return (G + transpose(G) + I) / 2
end
loss_bsm_matrix_pgen(sigsag::SIGSAG) = loss_bsm_matrix_pgen(sigsag.outcoupling_efficiency, sigsag.detection_efficiency)

function moment_vector(n1::Vector{Int}, n2::Vector{Int}, ηᵗ::Real, ηᵈ::Real)
    Ca12 = ηᵈ * (α[1]*α[2])
    Cb12 = ηᵈ * (β[1]*β[2])
    prod = one(R)
    for i in 3:mds
        prod *= (α[i]*sqrt(ηᵗ))^n1[i-2]/factorial(n1[i-2]) * (β[i]*sqrt(ηᵗ))^n2[i-2]/factorial(n2[i-2])
    end
    return Ca12 * Cb12 * prod
end


# TODO: review docstrings with Gabe for accuracy
"""
    probability_success(μ::Real, ηᵗ::Real, ηᵈ::Real, ηᵇ::Real, dark_counts::Real)

Calculate the probability of photon-photon state generation with the given parameters.

# Parameters
- μ : Mean photon number
- ηᵗ : Outcoupling efficiency
- ηᵈ : Detection efficiency
- ηᵇ : Bell state measurement efficiency
- dark_counts : Probability of click with no photon present

# Returns
Probability of successful photon-photon state generation
"""
function probability_success(μ::Real, ηᵗ::Real, ηᵈ::Real)
    cov = covariance_matrix(μ)
    A = k_function_matrix(cov) + loss_bsm_matrix_pgen(ηᵗ, ηᵈ)
    Ainv = inv(A)
    Γ = cov + (1/2)*I
    detΓ = det(Γ)

    D1 = sqrt(det(A))
    D2 = detΓ^(1/4)
    D3 = conj(detΓ)^(1/4)
    Coef = 1/(D1*D2*D3)

    C = ηᵈ^2 * (α[1]*α[2]) * (β[1]*β[2]) # moment_vector([0,0,0,0], [0,0,0,0], ηᵗ, ηᵈ)

    return real(Coef * W(C, Ainv))
end
probability_success(sigsag::SIGSAG) = probability_success(sigsag.mean_photon, sigsag.outcoupling_efficiency, sigsag.detection_efficiency)

# TODO: complete docstring
"""
    fidelity(μ::Real, ηᵗ::Real, ηᵈ::Real)

...

# Parameters
- μ  : Mean photon number (pair production strength)
- ηᵗ : Outcoupling / transmission efficiency
- ηᵈ : Detection efficiency

# Returns
Real-valued Bell-state fidelity of the SPDC source for the given parameters.
"""
function fidelity(μ::Real, ηᵗ::Real, ηᵈ::Real)
    cov = covariance_matrix(μ)
    A = k_function_matrix(cov) + loss_bsm_matrix_fid(ηᵗ, ηᵈ)
    Ainv = inv(A)
    Γ = cov + (1/2)*I
    detΓ = det(Γ)

    # Wick terms (cached)
    Fsum =
        W(moment_vector([1,0,0,1], [1,0,0,1], ηᵗ, ηᵈ), Ainv) +
        W(moment_vector([0,1,1,0], [0,1,1,0], ηᵗ, ηᵈ), Ainv) +
        W(moment_vector([1,0,0,1], [0,1,1,0], ηᵗ, ηᵈ), Ainv) +
        W(moment_vector([0,1,1,0], [1,0,0,1], ηᵗ, ηᵈ), Ainv)

    D1 = sqrt(det(A))
    D2 = detΓ^(1/4)
    D3 = conj(detΓ)^(1/4)

    pgen = probability_success(μ, ηᵗ, ηᵈ)

    coef = 1 / (2 * D1 * D2 * D3 * pgen)

    value = coef * Fsum
    if abs(imag(value)) > 1e-10
        @warn "fidelity has nontrivial imaginary part" imag=imag(value) value=value
    end
    return real(value)
end
fidelity(sigsag::SIGSAG) = fidelity(sigsag.mean_photon, sigsag.outcoupling_efficiency, sigsag.detection_efficiency)

end # module