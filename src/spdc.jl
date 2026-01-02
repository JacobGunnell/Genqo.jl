module spdc

using BlockDiagonals
using Nemo
using LinearAlgebra
using PythonCall

using ..Genqo: GenqoBase, Sweepable, _pyconvert_sweepable
import ..tmsv
using ..tools


Base.@kwdef mutable struct SPDC <: GenqoBase
    mean_photon::Sweepable{AbstractFloat} = 1e-2
    detection_efficiency::Sweepable{AbstractFloat} = 1.0
    bsm_efficiency::Sweepable{AbstractFloat} = 1.0
    outcoupling_efficiency::Sweepable{AbstractFloat} = 1.0
end
Base.convert(::Type{SPDC}, spdc_py::Py) = SPDC(
    tools._pyconvert_sweepable(Float64, spdc_py.mean_photon),
    #pyconvert(Vector{Float64}, spdc_py.schmidt_coeffs),
    tools._pyconvert_sweepable(Float64, spdc_py.detection_efficiency),
    tools._pyconvert_sweepable(Float64, spdc_py.bsm_efficiency),
    tools._pyconvert_sweepable(Float64, spdc_py.outcoupling_efficiency),
    #pyconvert(Float64, spdc_py.dark_counts),
    #pyconvert(Float64, spdc_py.visibility),
)

# Global canonical position and momentum variables
const mds = 4 # Number of modes for our system

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

_perm_matrix_12785634 = permutation_matrix([1,2,7,8,5,6,3,4])

"""
Calculate the covariance matrix of the SPDC source
"""
function covariance_matrix(μ::Float64)
    tmsv_covar = tmsv.covariance_matrix(μ)
    covar = Matrix(BlockDiagonal([tmsv_covar, tmsv_covar]))

    return _perm_matrix_12785634 * covar * _perm_matrix_12785634'
end
covariance_matrix(spdc::SPDC) = covariance_matrix(spdc.mean_photon)

"""
Calculate the loss portion of the A matrix, specifically when calculating the fidelity.
"""
function loss_bsm_matrix_fid(ηᵗ::Float64, ηᵈ::Float64)
    G = zeros(ComplexF64, 16, 16)
    η = ηᵗ*ηᵈ

    for i in 1:4
        G[i, i+8] = (η - 1)
        G[i, i+12] = -im*(η - 1)
        G[i+4, i+8] = im*(η - 1)
        G[i+4, i+12] = (η - 1)
    end

    return (G + transpose(G) + I) / 2
end
loss_bsm_matrix_fid(spdc::SPDC) = loss_bsm_matrix_fid(spdc.outcoupling_efficiency, spdc.detection_efficiency)

"""
Calculating the portion of the A matrix that arises due to incorporating loss, specifically for the trace of the BSM matrix
"""
loss_bsm_matrix_trace::Matrix{ComplexF64} = begin
    G = zeros(ComplexF64, 16, 16)

    for i in 1:4
        G[i, i+8] = -1
        G[i, i+12] = im
        G[i+4, i+8] = -im
        G[i+4, i+12] = -1
    end

    (G + transpose(G) + I) / 2
end

"""
    dmijZ(dmi::Int, dmj::Int, Ainv::Matrix{ComplexF64}, nvec::Vector{Int}, ηᵗ::Float64, ηᵈ::Float64, ηᵇ::Float64)

Calculate a single element of the unnormalized density matrix.

# Parameters
- dmi : Row number for the corresponding density matrix element
- dmj : Column number for the corresponding density matrix element
- Ainv : Numerical inverse of the A matrix
- nvec : The vector of nᵢ's for the system, where nᵢ is the number of photons in mode i
- ηᵗ : Transmission efficiency
- ηᵈ : Detection efficiency
- ηᵇ : Bell state measurement efficiency

# Returns
Density matrix element for the ZALM source
"""
function dmijZ(dmi::Int, dmj::Int, Ainv::Matrix{ComplexF64}, nvec::Vector{Int}, ηᵗ::Float64, ηᵈ::Float64)
    η = [ηᵗ*ηᵈ, ηᵗ*ηᵈ, ηᵗ*ηᵈ, ηᵗ*ηᵈ]

    # Calculate Ca based on dmi value
    if dmi == 1
        Ca₁ = ((α[1]*sqrt(η[1]) - α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(η[1]) + α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[3]*sqrt(η[3]) - α[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[3])
        Ca₄ = ((α[3]*sqrt(η[3]) + α[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[4])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    elseif dmi == 2
        Ca₁ = ((α[1]*sqrt(η[1]) - α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(η[1]) + α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[3]*sqrt(η[3]) + α[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[3])
        Ca₄ = ((α[3]*sqrt(η[3]) - α[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[4])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    elseif dmi == 3
        Ca₁ = ((α[1]*sqrt(η[1]) + α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(η[1]) - α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[3]*sqrt(η[3]) - α[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[3])
        Ca₄ = ((α[3]*sqrt(η[3]) + α[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[4])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    elseif dmi == 4
        Ca₁ = ((α[1]*sqrt(η[1]) + α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(η[1]) - α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[3]*sqrt(η[3]) + α[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[3])
        Ca₄ = ((α[3]*sqrt(η[3]) - α[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[4])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    else
        Ca = 1
    end

    # Calculate Cb based on dmj value
    if dmj == 1
        Cb₁ = ((β[1]*sqrt(η[1]) - β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(η[1]) + β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[3]*sqrt(η[3]) - β[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[3])
        Cb₄ = ((β[3]*sqrt(η[3]) + β[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[4])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    elseif dmj == 2
        Cb₁ = ((β[1]*sqrt(η[1]) - β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(η[1]) + β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[3]*sqrt(η[3]) + β[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[3])
        Cb₄ = ((β[3]*sqrt(η[3]) - β[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[4])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    elseif dmj == 3
        Cb₁ = ((β[1]*sqrt(η[1]) + β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(η[1]) - β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[3]*sqrt(η[3]) - β[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[3])
        Cb₄ = ((β[3]*sqrt(η[3]) + β[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[4])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    elseif dmj == 4
        Cb₁ = ((β[1]*sqrt(η[1]) + β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(η[1]) - β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[3]*sqrt(η[3]) + β[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[3])
        Cb₄ = ((β[3]*sqrt(η[3]) - β[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[4])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    else
        Cb = 1
    end

    C = Ca*Cb

    # Sum over wick partitions
    return W(C, Ainv)
end

"""
    spin_density_matrix(μ::Float64, ηᵗ::Float64, ηᵈ::Float64, ηᵇ::Float64, nvec::Vector{Int})

Calculate the density operator of the single-mode ZALM source on the spin-spin state.

# Parameters
- μ : Mean photon number
- ηᵗ : Outcoupling efficiency
- ηᵈ : Detection efficiency
- ηᵇ : Bell state measurement efficiency
- nvec : The vector of nᵢ's for the system, where nᵢ is the number of photons in mode i

# Returns
Numerical complete spin density matrix
"""
function spin_density_matrix(μ::Float64, ηᵗ::Float64, ηᵈ::Float64, nvec::Vector{Int})
    lmat = 4
    mat = Matrix{ComplexF64}(undef, lmat, lmat)
    cov = covariance_matrix(μ)
    A = k_function_matrix(cov) + loss_bsm_matrix_fid(ηᵗ, ηᵈ)
    Ainv = inv(A)
    Γ = cov + (1/2)*I
    detΓ = det(Γ)

    D1 = sqrt(det(A))
    D2 = detΓ^(1/4)
    D3 = conj(detΓ)^(1/4)
    Coef = 1/(4*D1*D2*D3)

    for i in 1:lmat
        for j in 1:lmat
            mat[i,j] = dmijZ(i, j, Ainv, nvec, ηᵗ, ηᵈ)
        end
    end

    return Coef * mat
end
spin_density_matrix(spdc::SPDC, nvec::Vector{Int}) = spin_density_matrix(spdc.mean_photon, spdc.outcoupling_efficiency, spdc.detection_efficiency, nvec)

"""
    probability_success(μ::Float64, ηᵇ::Float64)

Calculate the probability of photon-photon state generation with the given parameters.

# Parameters
- μ : The mean photon number
- ηᵇ : Bell state measurement efficiency

# Returns
Probability of successful photon-photon state generation
"""
function probability_success(μ::Float64, ηᵇ::Float64)
    cov = covariance_matrix(μ)
    A = k_function_matrix(cov) + loss_bsm_matrix_trace
    #Ainv = inv(A)
    Γ = cov + (1/2)*I
    detΓ = det(Γ)

    N1 = ηᵇ^2
    D1 = sqrt(det(A))
    D2 = detΓ^(1/4)
    D3 = conj(detΓ)^(1/4)
    Coef = N1/(D1*D2*D3)

    return real(Coef)
end
probability_success(spdc::SPDC) = probability_success(spdc.mean_photon, spdc.bsm_efficiency)

end # module
