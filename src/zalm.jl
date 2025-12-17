# TODO: find a way to cache function results

module zalm

using DocStringExtensions
using BlockDiagonals
using Nemo
using LinearAlgebra

import ..spdc
import ..tools

struct ZALM
    mean_photon::Float64
    schmidt_coeffs::Vector{Float64}
    detection_efficiency::Float64
    bsm_efficiency::Float64
    outcoupling_efficiency::Float64
    dark_counts::Float64
    visibility::Float64
end

"""
$TYPEDSIGNATURES

Calculate the covariance matrix of the single-mode ZALM source
"""
function covariance_matrix(mean_photon::Float64)
    # Initial ZALM covariance matrix in qpqp ordering
    covar_qpqp = begin
        spdc_covar = spdc.covariance_matrix(mean_photon)
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
covariance_matrix(zalm::ZALM) = covariance_matrix(zalm.mean_photon)

function dmijZ(dmi, dmj, nAinv, nvec, ηᵗ, ηᵈ, ηᵇ)
    mds = 8 # Number of modes for our system

    _qai = ["qa$i" for i in 1:mds]
    _pai = ["pa$i" for i in 1:mds]
    _qbi = ["qb$i" for i in 1:mds]
    _pbi = ["pb$i" for i in 1:mds]
    all_qps = hcat(_qai, _pai, _qbi, _pbi)
    CC = ComplexField()
    I = onei(CC) # Imaginary unit in CC ring
    R, generators = polynomial_ring(CC, all_qps)
    (qai, pai, qbi, pbi) = (generators[:,i] for i in 1:mds)
    
    # Define the alpha and beta vectors
    α = []
    β = []
    for j in 1:mds
        push!(α, (qai[j] + I * pai[j]) / sqrt(2))
        push!(β, (qbi[j] - I * pbi[j]) / sqrt(2))
    end

    ηᵛ = [ηᵗ*ηᵈ ηᵗ*ηᵈ ηᵇ ηᵇ ηᵇ ηᵇ ηᵗ*ηᵈ ηᵗ*ηᵈ]

    # Calculate Ca based on dmi value
    if dmi == 1
        Ca₁ = ((α[1]*sqrt(ηᵛ[1]) - α[2]*sqrt(ηᵛ[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(ηᵛ[1]) + α[2]*sqrt(ηᵛ[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[7]*sqrt(ηᵛ[7]) - α[8]*sqrt(ηᵛ[8])) * (1/sqrt(2)))^(nvec[7])
        Ca₄ = ((α[7]*sqrt(ηᵛ[7]) + α[8]*sqrt(ηᵛ[8])) * (1/sqrt(2)))^(nvec[8])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    elseif dmi == 2
        Ca₁ = ((α[1]*sqrt(ηᵛ[1]) - α[2]*sqrt(ηᵛ[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(ηᵛ[1]) + α[2]*sqrt(ηᵛ[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[7]*sqrt(ηᵛ[7]) + α[8]*sqrt(ηᵛ[8])) * (1/sqrt(2)))^(nvec[7])
        Ca₄ = ((α[7]*sqrt(ηᵛ[7]) - α[8]*sqrt(ηᵛ[8])) * (1/sqrt(2)))^(nvec[8])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    elseif dmi == 3
        Ca₁ = ((α[1]*sqrt(ηᵛ[1]) + α[2]*sqrt(ηᵛ[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(ηᵛ[1]) - α[2]*sqrt(ηᵛ[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[7]*sqrt(ηᵛ[7]) - α[8]*sqrt(ηᵛ[8])) * (1/sqrt(2)))^(nvec[7])
        Ca₄ = ((α[7]*sqrt(ηᵛ[7]) + α[8]*sqrt(ηᵛ[8])) * (1/sqrt(2)))^(nvec[8])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    elseif dmi == 4
        Ca₁ = ((α[1]*sqrt(ηᵛ[1]) + α[2]*sqrt(ηᵛ[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(ηᵛ[1]) - α[2]*sqrt(ηᵛ[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[7]*sqrt(ηᵛ[7]) + α[8]*sqrt(ηᵛ[8])) * (1/sqrt(2)))^(nvec[7])
        Ca₄ = ((α[7]*sqrt(ηᵛ[7]) - α[8]*sqrt(ηᵛ[8])) * (1/sqrt(2)))^(nvec[8])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    else
        Ca = 1
    end

    # Calculate Cb based on dmj value
    if dmj == 1
        Cb₁ = ((β[1]*sqrt(ηᵛ[1]) - β[2]*sqrt(ηᵛ[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(ηᵛ[1]) + β[2]*sqrt(ηᵛ[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[7]*sqrt(ηᵛ[7]) - β[8]*sqrt(ηᵛ[8])) * (1/sqrt(2)))^(nvec[7])
        Cb₄ = ((β[7]*sqrt(ηᵛ[7]) + β[8]*sqrt(ηᵛ[8])) * (1/sqrt(2)))^(nvec[8])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    elseif dmj == 2
        Cb₁ = ((β[1]*sqrt(ηᵛ[1]) - β[2]*sqrt(ηᵛ[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(ηᵛ[1]) + β[2]*sqrt(ηᵛ[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[7]*sqrt(ηᵛ[7]) + β[8]*sqrt(ηᵛ[8])) * (1/sqrt(2)))^(nvec[7])
        Cb₄ = ((β[7]*sqrt(ηᵛ[7]) - β[8]*sqrt(ηᵛ[8])) * (1/sqrt(2)))^(nvec[8])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    elseif dmj == 3
        Cb₁ = ((β[1]*sqrt(ηᵛ[1]) + β[2]*sqrt(ηᵛ[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(ηᵛ[1]) - β[2]*sqrt(ηᵛ[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[7]*sqrt(ηᵛ[7]) - β[8]*sqrt(ηᵛ[8])) * (1/sqrt(2)))^(nvec[7])
        Cb₄ = ((β[7]*sqrt(ηᵛ[7]) + β[8]*sqrt(ηᵛ[8])) * (1/sqrt(2)))^(nvec[8])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    elseif dmj == 4
        Cb₁ = ((β[1]*sqrt(ηᵛ[1]) + β[2]*sqrt(ηᵛ[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(ηᵛ[1]) - β[2]*sqrt(ηᵛ[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[7]*sqrt(ηᵛ[7]) + β[8]*sqrt(ηᵛ[8])) * (1/sqrt(2)))^(nvec[7])
        Cb₄ = ((β[7]*sqrt(ηᵛ[7]) - β[8]*sqrt(ηᵛ[8])) * (1/sqrt(2)))^(nvec[8])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    else
        Cb = 1
    end

    # Calculate Cd terms
    Cd₃ = (α[3]*β[3]*ηᵛ[3])^(nvec[3])/factorial(nvec[3])
    Cd₄ = (α[4]*β[4]*ηᵛ[4])^(nvec[4])/factorial(nvec[4])
    Cd₅ = (α[5]*β[5]*ηᵛ[5])^(nvec[5])/factorial(nvec[5])
    Cd₆ = (α[6]*β[6]*ηᵛ[6])^(nvec[6])/factorial(nvec[6])
    C = Cd₃*Cd₄*Cd₅*Cd₆*Ca*Cb

    # Sum over wick partitions
    elm = 0.0
    for (mon, coeff) in zip(monomials(C), coefficients(C))
        elm += wick_out(coeff, [i for i in 1:length(generators) if exponent(mon, 1, i) == 1], nAinv)
    end

    return elm
end

# Calculate density operator
function density_operator(nvec)
    lmat = 4
    mat = Matrix{ComplexF64}(undef, lmat, lmat)
    nA = k_function_matrix() + loss_bsm_matrix()
    nAnv = inv(nA)

    D1 = sqrt(det(nA))
    D2 = det(Gam)^0.25
    D3 = det(conj(Gam))^0.25
    Coef = 1/(D1 * D2 * D3)

    for i in 1:lmat
        for j in 1:lmat
            mat[i,j] = dmijZ(i, j, nAnv, nvec, outcoupling_efficiency, detection_efficiency, bsm_efficiency)
        end
    end

    return Coef * mat
end

end # module
