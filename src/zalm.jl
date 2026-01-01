module zalm

using BlockDiagonals
using Nemo
using LinearAlgebra
using PythonCall

import ..spdc
using ..tools


Base.@kwdef mutable struct ZALM <: GenqoBase
    mean_photon::Sweepable{AbstractFloat} = 1e-2
    #schmidt_coeffs::Vector{Float64}
    detection_efficiency::Sweepable{AbstractFloat} = 1.0
    bsm_efficiency::Sweepable{AbstractFloat} = 1.0
    outcoupling_efficiency::Sweepable{AbstractFloat} = 1.0
    dark_counts::Sweepable{AbstractFloat} = 0.0
    #visibility::Sweepable{AbstractFloat} = 1.0
end
Base.convert(::Type{ZALM}, zalm_py::Py) = ZALM(
    tools._pyconvert_sweepable(Float64, zalm_py.mean_photon),
    #pyconvert(Vector{Float64}, zalm_py.schmidt_coeffs),
    tools._pyconvert_sweepable(Float64, zalm_py.detection_efficiency),
    tools._pyconvert_sweepable(Float64, zalm_py.bsm_efficiency),
    tools._pyconvert_sweepable(Float64, zalm_py.outcoupling_efficiency),
    tools._pyconvert_sweepable(Float64, zalm_py.dark_counts),
    #tools._pyconvert_sweepable(Float64, zalm_py.visibility),
)

# Global canonical position and momentum variables
const mds = 8 # Number of modes for our system

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
    Matrix(BlockDiagonal([Id2, St35, Id2, Id2, St35, Id2]))
end
_S46 = begin
    Id2 = Matrix{Float64}(I, 2, 2)
    St46 = [
        1  0           0  0;
        0  1/sqrt(2)   0  1/sqrt(2);
        0  0           1  0;
        0  -1/sqrt(2)  0  1/sqrt(2);
    ]
    Matrix(BlockDiagonal([Id2, St46, Id2, Id2, St46, Id2]))
end

"""
Calculate the covariance matrix of the single-mode ZALM source
"""
function covariance_matrix(μ::Float64)
    # Initial ZALM covariance matrix in qpqp ordering
    spdc_covar = spdc.covariance_matrix(μ)
    covar_qpqp = Matrix(BlockDiagonal([spdc_covar, spdc_covar]))

    # Reorder qpqp → qqpp and apply beamsplitters
    covar_qqpp = reorder(covar_qpqp) 
    return _S46 * _S35 * covar_qqpp * _S35' * _S46'
end
covariance_matrix(zalm::ZALM) = covariance_matrix(zalm.mean_photon)

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
loss_bsm_matrix_fid(zalm::ZALM) = loss_bsm_matrix_fid(zalm.outcoupling_efficiency, zalm.detection_efficiency, zalm.bsm_efficiency)

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
loss_bsm_matrix_pgen(zalm::ZALM) = loss_bsm_matrix_pgen(zalm.outcoupling_efficiency, zalm.detection_efficiency, zalm.bsm_efficiency)

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

    # Sum over wick partitions (compile polynomial into fast terms)
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
function spin_density_matrix(μ::Float64, ηᵗ::Float64, ηᵈ::Float64, ηᵇ::Float64, nvec::Vector{Int})
    lmat = 4
    mat = Matrix{ComplexF64}(undef, lmat, lmat)
    cov = covariance_matrix(μ)
    A = k_function_matrix(cov) + loss_bsm_matrix_fid(ηᵗ, ηᵈ, ηᵇ)
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
spin_density_matrix(zalm::ZALM, nvec::Vector{Int}) = spin_density_matrix(zalm.mean_photon, zalm.outcoupling_efficiency, zalm.detection_efficiency, zalm.bsm_efficiency, nvec)

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
# Precompiled terms for fast Wick evaluation for specific polynomials (avoids Nemo monomial parsing per call)
# current precompiled polynomials: all in moment_vector
const moment_terms::Dict{Int, Vector{Tuple{ComplexF64, Vector{Int}}}} = Dict(
    k => extract_W_terms(v) for (k, v) in moment_vector
)

"""
    probability_success(μ::Float64, ηᵗ::Float64, ηᵈ::Float64, ηᵇ::Float64, dark_counts::Float64)

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
function probability_success(μ::Float64, ηᵗ::Float64, ηᵈ::Float64, ηᵇ::Float64, dark_counts::Float64)
    cov = covariance_matrix(μ)
    A = k_function_matrix(cov) + loss_bsm_matrix_pgen(ηᵗ, ηᵈ, ηᵇ)
    Ainv = inv(A)
    Γ = cov + (1/2)*I
    detΓ = det(Γ)

    D1 = sqrt(det(A))
    D2 = detΓ^(1/4)
    D3 = conj(detΓ)^(1/4)
    Coef = 1/(D1*D2*D3)

    # TODO: should this indexing be changed to 1-based? Or is there some mathematical meaning to the 0 index?
    C1 = moment_terms[0]
    C2 = moment_terms[9]
    C3 = moment_terms[10]
    C4 = moment_terms[14]

    return real(Coef * (
        ηᵇ^2 * (1-dark_counts)^4 * W(C1, Ainv) +
        ηᵇ * dark_counts * (1-dark_counts)^3 * W(C2, Ainv) +
        ηᵇ * dark_counts * (1-dark_counts)^3 * W(C3, Ainv) +
        dark_counts^2 * (1-dark_counts)^2 * W(C4, Ainv)
    ))
end
probability_success(zalm::ZALM) = probability_success(zalm.mean_photon, zalm.outcoupling_efficiency, zalm.detection_efficiency, zalm.bsm_efficiency, zalm.dark_counts)

function fidelity(μ::Float64, ηᵗ::Float64, ηᵈ::Float64, ηᵇ::Float64)
 # Calculate the fidelity with respect to the Bell state for the photon-photon single-mode ZALM source

    cov = covariance_matrix(μ)
    K = k_function_matrix(cov)

    # The loss matrix will be unique for calculating the fidelity    
    # --- A1 (fidelity loss) ---
    A1 = K + loss_bsm_matrix_fid(ηᵗ, ηᵈ, ηᵇ)

    factoredA1 = lu(A1) # factorizes for reuse
    Ainv1 = inv(factoredA1)
    D1 = sqrt(det(factoredA1)) # reuses factorization

    # ---- Compute W terms (cached) ----
    Fsum =
        W(moment_terms[1], Ainv1) +
        W(moment_terms[2], Ainv1) +
        W(moment_terms[3], Ainv1) +
        W(moment_terms[4], Ainv1)

    # --- A2 (trace / generation normalization loss) ---
    A2 = K + loss_bsm_matrix_pgen(ηᵗ, ηᵈ, ηᵇ)
    factoredA2 = lu(A2) # factorizes for reuse
    Ainv2 = inv(factoredA2)
    N2 = sqrt(det(factoredA2)) # reuses factorization
    
    Trc = W(moment_terms[0], Ainv2)  # <-- cached + defined

    N1 = (ηᵈ * ηᵗ) ^ 2
    coef = N1 * N2 / (2 * D1)

    value = coef * Fsum / Trc 

    if abs(imag(value)) > 1e-10
        @warn "fidelity has nontrivial imaginary part" imag=imag(value) value=value
    end
    return real(value)#<-- is the restriction to real correct?
end
fidelity(zalm::ZALM) = fidelity(zalm.mean_photon, zalm.outcoupling_efficiency, zalm.detection_efficiency, zalm.bsm_efficiency)

end # module
