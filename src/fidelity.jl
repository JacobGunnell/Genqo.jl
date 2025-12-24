using LinearAlgebra 
using Genqo
using Genqo.tools: wick_out, W
using Genqo.zalm: ZALM, loss_bsm_matrix, k_function_matrix, moment_vector # moment_vector pending



# TODO: moment_vector, basisv, calculate_loss_bsm_matrix_pgen

function fidelity(zalm::ZALM)
 # Calculate the fidelity with respect to the Bell state for the photon-photon single-mode ZALM source
    
    # Define the matrix element
    # Python: Cn1 = ZALM.moment_vector([1], 1)
    Cn0 = moment_vector([1], 0)
    Cn1 = moment_vector([1], 1)
    Cn2 = moment_vector([1], 2)
    Cn3 = moment_vector([1], 3)
    Cn4 = moment_vector([1], 4)

    # The loss matrix will be unique for calculating the fidelity    
    L1 = loss_bsm_matrix_fid(zalm)
    K1 = k_function_matrix(zalm)

    nA1 = K1 + L1
    Anv1 = inv(nA1)

    # ---- Compute W terms ----
    F1 = W(Cn1, Anv1)
    F2 = W(Cn2, Anv1)
    F3 = W(Cn3, Anv1)
    F4 = W(Cn4, Anv1)

    # Now calculate the trace of the state, which is equivalent to the probability of generation
    L2 = loss_bsm_matrix_pgen(zalm)
    K2 = k_function_matrix(zalm)

    nA2 = K2 + L2

    N1 = ((zalm.detection_efficiency^2) * (zalm.outcoupling_efficiency^2))^2
    N2 = sqrt(det(nA2))

    # ---- Determinant normalizations ----
    #   If on of the determinants is complex, sqrt and ^(0.25) use the principal complex root.
    #   That matches NumPy broadly, but can change phase if det moves around.
    D1 = sqrt(det(nA1))

    coef = N1 * N2 / (2*D1)
    Trc = W(Cn0, nA2)

    return coef * (F1 + F2 + F3 + F4) / Trc
end