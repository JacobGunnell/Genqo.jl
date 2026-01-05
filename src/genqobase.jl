using PythonCall


Sweepable{T} = Union{T, AbstractVector{<:T}} where T

abstract type GenqoBase end

# Enable broadcasting over Genqo objects with sweep parameters
# TODO: give explicit sweep type like in Python wrapper so that Genqo objects can have mixed sweep/non-sweep (1+)-dimensional fields (e.g. schmidt coefficients)
# For now, we assume all fields are either single values or sweeps of single values
Base.size(gq::GenqoBase) = Tuple(length(getfield(gq, f)) for f in fieldnames(typeof(gq)))
Base.length(gq::GenqoBase) = prod(size(gq))

struct SweepStyle <: Base.Broadcast.BroadcastStyle end
Base.Broadcast.BroadcastStyle(::Type{<:GenqoBase}) = SweepStyle()
Base.Broadcast.broadcastable(x::GenqoBase) = x

# N-dimensional broadcast shape
function Base.axes(gq::T) where T<:GenqoBase 
    Tuple(Base.OneTo(length(getfield(gq, f))) for f in fieldnames(T) if length(getfield(gq, f)) > 1)
end

# Element access during broadcast
@inline function Base.getindex(gq::T, I::CartesianIndex) where T<:GenqoBase
    swept_idx = 1
    params = map(fieldnames(T)) do f
        field_val = getfield(gq, f)
        if length(field_val) > 1
            idx = I[swept_idx]
            swept_idx += 1
            field_val[idx]
        else
            field_val[1]
        end
    end
    return T(params...)
end

@inline function Base.getindex(gq::T, i::Int) where T<:GenqoBase
    # Map linear index to CartesianIndex based on swept dimensions only
    swept_axes = axes(gq)
    cart_idx = CartesianIndices(swept_axes)[i]
    return gq[cart_idx]
end

function Base.similar(bc::Broadcast.Broadcasted{SweepStyle}, ::Type{ElType}) where ElType
    ax = axes(bc)
    # Return a vector for single-parameter sweeps, array for multi-parameter
    length(ax) == 1 ? similar(Vector{ElType}, ax[1]) : similar(Array{ElType}, ax)
end

"""
Convert a Python object to a Julia type, using custom conversion for sweep types.
"""
function _pyconvert_sweepable(T::Type, py_obj::Py)
    name = pyconvert(String, py_obj.__class__.__name__)
    if name == "linsweep"
        return range(
            pyconvert(T, py_obj.start), 
            stop=pyconvert(T, py_obj.stop), 
            length=pyconvert(Int, py_obj.length)
        )
    elseif name == "logsweep"
        return logrange(
            pyconvert(T, py_obj.start), 
            pyconvert(T, py_obj.stop), 
            length=pyconvert(Int, py_obj.length)
        )
    elseif name == "ptsweep"
        return pyconvert(Vector{T}, py_obj.points)
    else
        return pyconvert(T, py_obj)
    end
end
