# TODO: find a way to cache function results

module zalm

using BlockDiagonals
using Nemo
using LinearAlgebra

import ..spdc
import ..tools
using ..tools: ZALMParams


# Global quadrature variables
mds = 8 # Number of modes for our system

_qai = ["qa$i" for i in 1:mds]
_pai = ["pa$i" for i in 1:mds]
_qbi = ["qb$i" for i in 1:mds]
_pbi = ["pb$i" for i in 1:mds]
all_qps = hcat(_qai, _pai, _qbi, _pbi)
CC = ComplexField()
i = onei(CC) # Imaginary unit in CC ring
R, generators = polynomial_ring(CC, all_qps)
(qai, pai, qbi, pbi) = (generators[:,i] for i in 1:mds)

# Define the alpha and beta vectors
α = (qai + i .* pai) / sqrt(2)
β = (qbi - i .* pbi) / sqrt(2)

"""
Calculate the covariance matrix of the single-mode ZALM source
"""
function covariance_matrix(μ::Float64)
    # Initial ZALM covariance matrix in qpqp ordering
    covar_qpqp = begin
        spdc_covar = spdc.covariance_matrix(μ)
        BlockDiagonal([spdc_covar, spdc_covar])
    end

    # Reorder qpqp → qqpp
    perm_indices = [1:2:15; 2:2:16]
    perm_matrix = tools.permutation_matrix(perm_indices)
    covar_qqpp = perm_matrix * covar_qpqp * perm_matrix'
    
    # Apply the symplective matrices that represent 50/50 beamsplitters between the bell state modes
    S35 = begin
        Id2 = Matrix{Float64}(I, 2, 2)
        St35 = [
            1/sqrt(2) 0 1/sqrt(2) 0;
            0 1 0 0;
            -1/sqrt(2) 0 1/sqrt(2) 0;
            0 0 0 1;
        ]
        BlockDiagonal([Id2, St35, Id2, Id2, St35, Id2])
    end
    S46 = begin
        Id2 = Matrix{Float64}(I, 2, 2)
        St46 = [
            1 0 0 0;
            0 1/sqrt(2) 0 1/sqrt(2);
            0 0 1 0;
            0 -1/sqrt(2) 0 1/sqrt(2);
        ]
        BlockDiagonal([Id2, St46, Id2, Id2, St46, Id2])
    end

    return S46 * S35 * covar_qqpp * S35' * S46'
end
covariance_matrix(params::ZALMParams) = covariance_matrix(params.mean_photon)

"""
Calculate the loss portion of the A matrix, specifically when calculating the fidelity.
"""
function loss_bsm_matrix_fid(ηᵗ::Float64, ηᵈ::Float64, ηᵇ::Float64)
    G = zeros(ComplexF64, 32, 32)
    η = [ηᵗ*ηᵈ, ηᵗ*ηᵈ, ηᵇ, ηᵇ, ηᵇ, ηᵇ, ηᵗ*ηᵈ, ηᵗ*ηᵈ]

    for i in 1:8
        G[i, i+16] = (η[i] - 1)
        G[i, i+24] = -im*(η[i] - 1)
        G[i+16, i+8] = im*(η[i] - 1)
        G[i+24, i+8] = (η[i] - 1)
    end

    return (G + transpose(G) + I) / 2
end
loss_bsm_matrix_fid(params::ZALMParams) = loss_bsm_matrix_fid(params.outcoupling_efficiency, params.detection_efficiency, params.bsm_efficiency)

"""
Calculate the loss portion of the A matrix, specifically when calculating probability of success
"""
function loss_bsm_matrix_pgen(ηᵗ::Float64, ηᵈ::Float64, ηᵇ::Float64)
    G = zeros(ComplexF64, 32, 32)
    η = [ηᵗ*ηᵈ, ηᵗ*ηᵈ, ηᵇ, ηᵇ, ηᵇ, ηᵇ, ηᵗ*ηᵈ, ηᵗ*ηᵈ]

    for i in 1:8
        if i in (1,2,7,8)
            G[i, i+16] = -1
            G[i, i+24] = im
            G[i+16, i+8] = -im
            G[i+24, i+8] = -1
        else
            G[i, i+16] = (η[i] - 1)
            G[i, i+24] = -im*(η[i] - 1)
            G[i+16, i+8] = im*(η[i] - 1)
            G[i+24, i+8] = (η[i] - 1)
        end
    end

    return (G + transpose(G) + I) / 2
end
loss_bsm_matrix_pgen(params::ZALMParams) = loss_bsm_matrix_pgen(params.outcoupling_efficiency, params.detection_efficiency, params.bsm_efficiency)

"""
    dmijZ(dmi::Int, dmj::Int, Ainv::Matrix{ComplexF64}, nvec::Vector{Int}, ηᵗ::Float64, ηᵈ::Float64, ηᵇ::Float64)

Calculate a single element of the unnormalized density matrix.

# Parameters
- dmi   : Row number for the corresponding density matrix element
- dmj   : Column number for the corresponding density matrix element
- Ainv  : Numerical inverse of the A matrix
- nvec  : The vector of nᵢ's for the system, where nᵢ is the number of photons in mode i
- ηᵗ    : Transmission efficiency
- ηᵈ    : Detection efficiency
- ηᵇ    : Bell state measurement efficiency

# Returns
Density matrix element for the ZALM source
"""
function dmijZ(dmi::Int, dmj::Int, Ainv::Matrix{ComplexF64}, nvec::Vector{Int}, ηᵗ::Float64, ηᵈ::Float64, ηᵇ::Float64)
    η = [ηᵗ*ηᵈ, ηᵗ*ηᵈ, ηᵇ, ηᵇ, ηᵇ, ηᵇ, ηᵗ*ηᵈ, ηᵗ*ηᵈ]

    # Calculate Ca based on dmi value
    if dmi == 1
        Ca₁ = ((α[1]*sqrt(η[1]) - α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(η[1]) + α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[7]*sqrt(η[7]) - α[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[7])
        Ca₄ = ((α[7]*sqrt(η[7]) + α[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[8])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    elseif dmi == 2
        Ca₁ = ((α[1]*sqrt(η[1]) - α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(η[1]) + α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[7]*sqrt(η[7]) + α[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[7])
        Ca₄ = ((α[7]*sqrt(η[7]) - α[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[8])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    elseif dmi == 3
        Ca₁ = ((α[1]*sqrt(η[1]) + α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(η[1]) - α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[7]*sqrt(η[7]) - α[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[7])
        Ca₄ = ((α[7]*sqrt(η[7]) + α[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[8])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    elseif dmi == 4
        Ca₁ = ((α[1]*sqrt(η[1]) + α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(η[1]) - α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[7]*sqrt(η[7]) + α[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[7])
        Ca₄ = ((α[7]*sqrt(η[7]) - α[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[8])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    else
        Ca = 1
    end

    # Calculate Cb based on dmj value
    if dmj == 1
        Cb₁ = ((β[1]*sqrt(η[1]) - β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(η[1]) + β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[7]*sqrt(η[7]) - β[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[7])
        Cb₄ = ((β[7]*sqrt(η[7]) + β[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[8])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    elseif dmj == 2
        Cb₁ = ((β[1]*sqrt(η[1]) - β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(η[1]) + β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[7]*sqrt(η[7]) + β[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[7])
        Cb₄ = ((β[7]*sqrt(η[7]) - β[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[8])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    elseif dmj == 3
        Cb₁ = ((β[1]*sqrt(η[1]) + β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(η[1]) - β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[7]*sqrt(η[7]) - β[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[7])
        Cb₄ = ((β[7]*sqrt(η[7]) + β[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[8])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    elseif dmj == 4
        Cb₁ = ((β[1]*sqrt(η[1]) + β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(η[1]) - β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[7]*sqrt(η[7]) + β[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[7])
        Cb₄ = ((β[7]*sqrt(η[7]) - β[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[8])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    else
        Cb = 1
    end

    # Calculate Cd terms
    Cd₃ = (α[3]*β[3]*η[3])^(nvec[3])/factorial(nvec[3])
    Cd₄ = (α[4]*β[4]*η[4])^(nvec[4])/factorial(nvec[4])
    Cd₅ = (α[5]*β[5]*η[5])^(nvec[5])/factorial(nvec[5])
    Cd₆ = (α[6]*β[6]*η[6])^(nvec[6])/factorial(nvec[6])
    C = Cd₃*Cd₄*Cd₅*Cd₆*Ca*Cb

    # Sum over wick partitions
    return tools.W(C, Ainv)
end

"""
    spin_density_matrix(μ::Float64, ηᵗ::Float64, ηᵈ::Float64, ηᵇ::Float64, nvec::Vector{Int})

Calculate the density operator of the single-mode ZALM source on the spin-spin state.

# Parameters
- μ    : Mean photon number
- ηᵗ   : Outcoupling efficiency
- ηᵈ   : Detection efficiency
- ηᵇ   : Bell state measurement efficiency
- nvec : The vector of nᵢ's for the system, where nᵢ is the number of photons in mode i

# Returns
Numerical complete spin density matrix
"""
function spin_density_matrix(μ::Float64, ηᵗ::Float64, ηᵈ::Float64, ηᵇ::Float64, nvec::Vector{Int})
    lmat = 4
    mat = Matrix{ComplexF64}(undef, lmat, lmat)
    cov = covariance_matrix(μ)
    A = tools.k_function_matrix(cov) + loss_bsm_matrix_fid(ηᵗ, ηᵈ, ηᵇ)
    Ainv = inv(A)
    Γ = cov + (1/2)*I
    detΓ = det(Γ)

    D1 = sqrt(det(A))
    D2 = detΓ^(1/4)
    D3 = conj(detΓ)^(1/4)
    Coef = 1/(D1*D2*D3)

    for i in 1:lmat
        for j in 1:lmat
            mat[i,j] = dmijZ(i, j, Ainv, nvec, ηᵗ, ηᵈ, ηᵇ)
        end
    end

    return Coef * mat
end
spin_density_matrix(params::ZALMParams, nvec::Vector{Int}) = spin_density_matrix(params.mean_photon, params.outcoupling_efficiency, params.detection_efficiency, params.bsm_efficiency, nvec)

const moment_vector::Dict{Int, Nemo.Generic.MPoly{Nemo.ComplexFieldElem}} = begin
    Ca₁ = α[1]*α[3]*α[4]*α[8]
    Ca₂ = α[2]*α[3]*α[4]*α[7]
    Cb₁ = β[1]*β[3]*β[4]*β[8]
    Cb₂ = β[2]*β[3]*β[4]*β[7]

    # For calculating the normalization constant
    Ca₃ = α[1]*α[3]*α[4]*α[7]
    Ca₄ = α[2]*α[3]*α[4]*α[8]
    Cb₃ = β[1]*β[3]*β[4]*β[7]
    Cb₄ = β[2]*β[3]*β[4]*β[8]

    Dict(
        0 => α[3]*α[4]*β[3]*β[4],
        1 => Ca₁*Cb₁,
        2 => Ca₁*Cb₂,
        3 => Ca₂*Cb₁,
        4 => Ca₂*Cb₂,
        5 => Ca₃*Cb₃,
        6 => Ca₃*Cb₄,
        7 => Ca₄*Cb₃,
        8 => Ca₄*Cb₄,
        9 => α[3]*β[3],
        10 => α[4]*β[4],
        11 => α[3]*α[4]*β[3]*β[4],
        12 => α[1]*α[1]*β[1]*β[1],
        14 => one(R)
    )
end

function probability_success(μ::Float64, ηᵗ::Float64, ηᵈ::Float64, ηᵇ::Float64, dark_counts::Float64)
    cov = covariance_matrix(μ)
    A = tools.k_function_matrix(cov) + loss_bsm_matrix_pgen(ηᵗ, ηᵈ, ηᵇ)
    Ainv = inv(A)
    Γ = cov + (1/2)*I
    detΓ = det(Γ)

    D1 = sqrt(det(A))
    D2 = detΓ^(1/4)
    D3 = conj(detΓ)^(1/4)
    Coef = 1/(D1*D2*D3)

    # TODO: should this indexing be changed to 1-based? Or is there some mathematical meaning to the 0 index?
    C1 = moment_vector[0]
    C2 = moment_vector[9]
    C3 = moment_vector[10]
    C4 = moment_vector[14]

    return real(Coef * (
        ηᵇ^2 * (1-dark_counts)^4 * tools.W(C1, Ainv) +
        ηᵇ * dark_counts * (1-dark_counts)^3 * tools.W(C2, Ainv) +
        ηᵇ * dark_counts * (1-dark_counts)^3 * tools.W(C3, Ainv) +
        dark_counts^2 * (1-dark_counts)^2 * tools.W(C4, Ainv)
    ))
end
probability_success(params::ZALMParams) = probability_success(params.mean_photon, params.outcoupling_efficiency, params.detection_efficiency, params.bsm_efficiency, params.dark_counts)

function fidelity(μ::Float64, ηᵗ::Float64, ηᵈ::Float64, ηᵇ::Float64)
 # Calculate the fidelity with respect to the Bell state for the photon-photon single-mode ZALM source

    cov = covariance_matrix(μ)
    
    # Define the matrix element
    # Python: Cn1 = ZALM.moment_vector([1], 1)
    Cn0 = moment_vector[0]
    Cn1 = moment_vector[1]
    Cn2 = moment_vector[2]
    Cn3 = moment_vector[3]
    Cn4 = moment_vector[4]

    # The loss matrix will be unique for calculating the fidelity    
    L1 = loss_bsm_matrix_fid(ηᵗ, ηᵈ, ηᵇ)
    K = tools.k_function_matrix(cov)

    nA1 = K + L1
    Anv1 = inv(nA1)

    # ---- Compute W terms ----
    F1 = tools.W(Cn1, Anv1)
    F2 = tools.W(Cn2, Anv1)
    F3 = tools.W(Cn3, Anv1)
    F4 = tools.W(Cn4, Anv1)

    # Now calculate the trace of the state, which is equivalent to the probability of generation
    L2 = loss_bsm_matrix_pgen(ηᵗ, ηᵈ, ηᵇ)

    nA2 = K + L2

    N1 = (ηᵈ*ηᵗ)^2
    N2 = sqrt(det(nA2))

    # ---- Determinant normalizations ----
    #   If on of the determinants is complex, sqrt and ^(0.25) use the principal complex root.
    #   That matches NumPy broadly, but can change phase if det moves around.
    D1 = sqrt(det(nA1))

    coef = N1 * N2 / (2*D1)
    Trc = tools.W(Cn0, nA2)

    return coef * (F1 + F2 + F3 + F4) / Trc
end
fidelity(params::ZALMParams) = fidelity(params.mean_photon, params.outcoupling_efficiency, params.detection_efficiency, params.bsm_efficiency)

end # module
