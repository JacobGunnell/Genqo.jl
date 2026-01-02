using PythonCall


Sweepable{T} = Union{T, AbstractRange{<:T}, AbstractVector{<:T}} where T

abstract type GenqoBase end

# Enable broadcasting over Genqo objects with sweep parameters
# TODO: give explicit sweep type like in Python wrapper so that Genqo objects can have mixed sweep/non-sweep (1+)-dimensional fields (e.g. schmidt coefficients)
# For now, we assume all fields are either single values or StepRangeLen sweeps of single values
function Base.length(gq::T) where T<:GenqoBase
    len = 1
    for field in fieldnames(T)
        len *= length(getproperty(gq, field))
    end
    return len
end

# TODO: allow iteration to preserve the shape of cartesian product over fields, as in tmsv.probability_success.(tmsv.TMSV(0.01:0.01:10, 0.7:0.1:0.9))
# Currently, we flatten to 1D iteration, so above returns a 3000-element Vector instead of a 1000x3 Matrix
function Base.iterate(gq::T, state=1) where T<:GenqoBase
    total_len = length(gq)
    
    if state > total_len
        return nothing
    end
    
    # Get field names and convert each field to a collection
    fields = fieldnames(T)
    field_collections = map(fields) do field
        val = getproperty(gq, field)
        val isa Union{AbstractRange, AbstractVector} ? val : [val]
    end
    
    # Calculate the multi-dimensional index for the Cartesian product
    field_lengths = map(length, field_collections)
    indices = zeros(Int, length(fields))
    
    # Convert linear state to multi-dimensional indices
    remainder = state - 1
    for i in length(fields):-1:1
        indices[i] = remainder % field_lengths[i] + 1
        remainder ÷= field_lengths[i]
    end
    
    # Extract values at the calculated indices
    field_values = map(enumerate(field_collections)) do (i, collection)
        collection[indices[i]]
    end
    
    return T(field_values...), state + 1
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
