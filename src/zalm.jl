# TODO: find a way to cache function results

module zalm

using DocStringExtensions
using BlockDiagonals
using BlockArrays
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

"""
$TYPEDSIGNATURES

Calculate the K function portion of the A matrix for the single-mode ZALM source.
"""
function k_function_matrix(covariance_matrix::Matrix{Float64})
    Γ = covariance_matrix + (1/2)*I
    sz = size(Γ)[1] ÷ 2
    Γinv = BlockArray(inv(Γ), [sz,sz], [sz,sz])

    A = Γinv[Block(1,1)]
    C = Γinv[Block(1,2)]
    Cᵀ = Γinv[Block(2,1)]
    B = Γinv[Block(2,2)]

    BB = (1/2)*mortar(reshape([
        A+(im/2)*(C+Cᵀ), C-(im/2)*(A-B),
        Cᵀ-(im/2)*(A-B), B-(im/2)*(C+Cᵀ)
    ], 2, 2))

    return BlockDiagonal([BB, conj(BB)])
end
k_function_matrix(zalm::ZALM) = k_function_matrix(covariance_matrix(zalm))

"""
$TYPEDSIGNATURES

Calculate the loss portion of the A matrix, specifically when calculating the fidelity.
"""
function loss_bsm_matrix_fid(ηᵗ::Float64, ηᵈ::Float64, ηᵇ::Float64)
    # TODO: find out from Gabe why we have a different loss_bsm_matrix function for fidelity, generation probability, etc.
    G = zeros(ComplexF64, 32, 32)
    η = [ηᵗ*ηᵈ, ηᵗ*ηᵈ, ηᵇ, ηᵇ, ηᵇ, ηᵇ, ηᵗ*ηᵈ, ηᵗ*ηᵈ]

    for i in 1:8
        G[i, i+16] = (η[i] - 1)
        G[i, i+24] = -im*(η[i] - 1)
        G[i+16, i+8] = im*(η[i] - 1)
        G[i+24, i+8] = (η[i] - 1)
    end

    return (G + G' + I) / 2
end
loss_bsm_matrix_fid(zalm::ZALM) = loss_bsm_matrix_fid(zalm.outcoupling_efficiency, zalm.detection_efficiency, zalm.bsm_efficiency)

"""
$TYPEDSIGNATURES

Calculate a single element of the unnormalized density matrix.
Arguments:
- dmi: The row number for the cooresponding density matrix element
- dmj: The column number for the cooresponding density matrix element
- nAinv: The numerical inverse of the A matrix
- nvec: The vector of n_i's for the system, where n_i is the number of photons in mode i
- ηᵗ: The transmission efficiency
- ηᵈ: The detection efficiency
- ηᵇ: The Bell state measurement efficiency
Output:
- The density matrix element for the ZALM source
"""
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
    elm = 0.0
    for (mon, coeff) in zip(monomials(C), coefficients(C))
        elm += tools.wick_out(coeff, [i for i in 1:length(generators) if exponent(mon, 1, i) == 1], nAinv)
    end

    return elm
end

"""
$TYPEDSIGNATURES

Calculate the density operator of the single-mode ZALM source on the spin-spin state.
Arguments
- ηᵗ: Outcoupling efficiency
- ηᵈ: Detection efficiency
- ηᵇ: Bell state measurement efficiency
- nvec: The vector of n_i's for the system, where n_i is the number of photons in mode i
Output
- The numerical complete spin density matrix
"""
function density_operator(mean_photon::Float64, ηᵗ::Float64, ηᵈ::Float64, ηᵇ::Float64, nvec::Vector{Int})
    lmat = 4
    mat = Matrix{ComplexF64}(undef, lmat, lmat)
    cov = covariance_matrix(mean_photon)
    nA = k_function_matrix(cov) + loss_bsm_matrix_fid(ηᵗ, ηᵈ, ηᵇ)
    nAinv = inv(nA)
    Γ = cov + (1/2)*I

    D1 = sqrt(det(nA))
    D2 = det(Γ)^(1/4)
    D3 = det(conj(Γ))^(1/4)
    Coef = 1/(D1*D2*D3)

    for i in 1:lmat
        for j in 1:lmat
            mat[i,j] = dmijZ(i, j, nAinv, nvec, ηᵗ, ηᵈ, ηᵇ)
        end
    end

    return Coef * mat
end
density_operator(zalm::ZALM, nvec::Vector{Int}) = density_operator(zalm.mean_photon, zalm.outcoupling_efficiency, zalm.detection_efficiency, zalm.bsm_efficiency, nvec)

end # module
