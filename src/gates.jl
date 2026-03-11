module gates

using LinearAlgebra

export Gate, num_qubits, Linear2QubitGate, getTransformMatrix, expand, apply!, BeamSplitter


abstract type Gate end
num_qubits(::Gate) = -1 # Any number of qubits

struct Linear2QubitGate <: Gate
    S::Matrix{Float64}
end
num_qubits(::Linear2QubitGate) = 2

function getTransformMatrix(gate::Linear2QubitGate, i::Int, j::Int, mds::Int)::Matrix{Float64}
    S = Matrix{Float64}(I, 2*mds, 2*mds)
    idx = [i, j, i+mds, j+mds]
    @views S[idx,idx] .= gate.S

    """
    Effectively sets rows/columns corresponding to qi, pi, qj, pj using row/column 1, 2, 3, 4 from `gate`. qqpp ordering.
    This does the equivalent of:

    S[i,i] = gate.S[1,1]
    S[i,j] = gate.S[1,2]
    S[j,i] = gate.S[2,1]
    S[j,j] = gate.S[2,2]

    S[i+mds,i] = gate.S[3,1]
    S[i+mds,j] = gate.S[3,2]
    S[j+mds,i] = gate.S[4,1]
    S[j+mds,j] = gate.S[4,2]

    ...

    """

    return S
end

function expand(gate::Linear2QubitGate, indices::Vector{Int}, mds::Int)::Linear2QubitGate
    Linear2QubitGate(getTransformMatrix(gate, indices[1], indices[2], mds))
end

function apply!(gate::Linear2QubitGate, V::Matrix{Float64})
    V .= gate.S * V * gate.S'
end

# All gates in qqpp representation

BeamSplitter(t::Real=0.5)::Linear2QubitGate = Linear2QubitGate(
    [
        sqrt(t)     sqrt(1-t)  0           0         ;
        -sqrt(1-t)  sqrt(t)    0           0         ;
        0           0          sqrt(t)     sqrt(1-t) ;
        0           0          -sqrt(1-t)  sqrt(t)   ;
    ]
)

# TODO: add φ parameter
Squeeze2Mode(r::Real, φ::Real=0)::Linear2QubitGate = Linear2QubitGate(
    1/sqrt(2) * 
    [
        cosh(r)  sinh(r)  0         0        ;
        sinh(r)  cosh(r)  0         0        ;
        0        0        cosh(r)   -sinh(r) ;
        0        0        -sinh(r)  cosh(r)  ;
    ]
)

ModeSwap()::Linear2QubitGate = Linear2QubitGate(
    [
        0   1   0   0 ;
        1   0   0   0 ;
        0   0   0   1 ;
        0   0   1   0 ;
    ]
)

end # module
   