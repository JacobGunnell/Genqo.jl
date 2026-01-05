from juliacall import Main as jl

from abc import ABC
from attrs import fields
from functools import wraps

import numpy as np


# Class telling genqo to sweep a parameter
class sweep(ABC):
    pass

class ptsweep(sweep):
    """Class representing a sweep over a set of discrete points."""
    def __init__(self, points: list[float] | np.ndarray) -> None:
        if isinstance(points, np.ndarray):
            self.points = points
        else:
            self.points = np.asarray(points)

    def __le__(self, other: float) -> bool:
        return np.all(point <= other for point in self.points)
    
    def __ge__(self, other: float) -> bool:
        return np.all(point >= other for point in self.points)

    def __len__(self) -> int:
        """Return the number of points in the sweep."""
        return len(self.points)
    
    def __array__(self) -> np.ndarray:
        """Convert sweep to numpy array for use with matplotlib and numpy operations."""
        return np.asarray(self.points)
    
    # Arithmetic operations
    def __add__(self, other: float):
        return ptsweep(self.points + other)
    
    def __radd__(self, other: float):
        return ptsweep(other + self.points)
    
    def __sub__(self, other: float):
        return ptsweep(self.points - other)
    
    def __rsub__(self, other: float):
        return ptsweep(other - self.points)
    
    def __mul__(self, other: float):
        return ptsweep(self.points * other)
    
    def __rmul__(self, other: float):
        return ptsweep(other * self.points)
    
    def __truediv__(self, other: float):
        return ptsweep(self.points / other)
    
    def __pow__(self, other: float):
        return ptsweep(self.points ** other)
    
    def __rpow__(self, other: float):
        return ptsweep(other ** self.points)
    
    def __neg__(self):
        return ptsweep(-self.points)

class linsweep(sweep):
    """Class representing a linear sweep from start to stop with a given length."""

    def __init__(self, start: float, stop: float, length: int) -> None:
        self.start = start
        self.stop = stop
        self.length = length

    def __le__(self, other: float) -> bool:
        return self.start <= other and self.stop <= other
    
    def __ge__(self, other: float) -> bool:
        return self.start >= other and self.stop >= other

    def __len__(self) -> int:
        """Return the length of the sweep."""
        return self.length
    
    def __array__(self) -> np.ndarray:
        """Convert sweep to numpy array for use with matplotlib and numpy operations."""
        return np.linspace(self.start, self.stop, self.length)
    
    # Arithmetic operations
    def __add__(self, other: float):
        return linsweep(self.start + other, self.stop + other, self.length)
    
    def __radd__(self, other: float):
        return linsweep(other + self.start, other + self.stop, self.length)
    
    def __sub__(self, other: float):
        return linsweep(self.start - other, self.stop - other, self.length)
    
    def __rsub__(self, other: float):
        return linsweep(other - self.start, other - self.stop, self.length)
    
    def __mul__(self, other: float):
        return linsweep(self.start * other, self.stop * other, self.length)
    
    def __rmul__(self, other: float):
        return linsweep(other * self.start, other * self.stop, self.length)
    
    def __truediv__(self, other: float):
        return linsweep(self.start / other, self.stop / other, self.length)
    
    def __pow__(self, other: float):
        return ptsweep(np.array(self) ** other)
    
    def __rpow__(self, other: float):
        return ptsweep(other ** np.array(self))
    
    def __neg__(self):
        return linsweep(-self.start, -self.stop, self.length)
    
class logsweep(sweep):
    """Class representing a logarithmic sweep from start to stop with a given length."""
    def __init__(self, start: float, stop: float, length: int) -> None:
        self.start = start
        self.stop = stop
        self.length = length

    def __le__(self, other: float) -> bool:
        return self.start <= other and self.stop <= other
    
    def __ge__(self, other: float) -> bool:
        return self.start >= other and self.stop >= other

    def __len__(self) -> int:
        """Return the length of the sweep."""
        return self.length
    
    def __array__(self) -> np.ndarray:
        """Convert sweep to numpy array for use with matplotlib and numpy operations."""
        return np.logspace(np.log10(self.start), np.log10(self.stop), self.length)
    
    # Arithmetic operations
    def __add__(self, other: float):
        return ptsweep(np.array(self) + other)
    
    def __radd__(self, other: float):
        return ptsweep(other + np.array(self))
    
    def __sub__(self, other: float):
        return ptsweep(np.array(self) - other)
    
    def __rsub__(self, other: float):
        return ptsweep(other - np.array(self))
    
    def __mul__(self, other: float):
        return logsweep(self.start * other, self.stop * other, self.length)
    
    def __rmul__(self, other: float):
        return logsweep(other * self.start, other * self.stop, self.length)
    
    def __truediv__(self, other: float):
        return logsweep(self.start / other, self.stop / other, self.length)
    
    def __pow__(self, other: float):
        return logsweep(self.start ** other, self.stop ** other, self.length)
    
    def __rpow__(self, other: float):
        return ptsweep(other ** np.array(self))
    
    def __neg__(self):
        return logsweep(-self.start, -self.stop, self.length)

# Decorator to support sweeping by setting a particular parameter to a sweep object: zalm.mean_photon = gq.logsweep(1e-4, 1e-2, 100)
def _sweepable(func: callable) -> callable:
    cls = func.__qualname__.split(".")[0]
    module = cls.lower()
    jl_module = getattr(jl, module)
    jl_cls_type = getattr(jl_module, cls)
        
    @wraps(func)
    def wrapper(self, *args, _func_name: str = func.__name__):
        # If no sweeping is required, call the function once directly
        if not any(isinstance(getattr(self, param.name), sweep) for param in fields(self.__class__)):
            return func(self, *args)

        # If sweeping is required, perform fast broadcast sweep in Julia
        else:
            # TODO: support function args other than self
            converted_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    if arg.dtype == np.float64:
                        converted_args.append(jl.convert(jl.Array[jl.Float64], arg))
                    elif arg.dtype == np.float32:
                        converted_args.append(jl.convert(jl.Array[jl.Float32], arg))
                    elif arg.dtype == np.int64:
                        converted_args.append(jl.convert(jl.Array[jl.Int64], arg))
                    elif arg.dtype == np.int32:
                        converted_args.append(jl.convert(jl.Array[jl.Int32], arg))
                    else:
                        converted_args.append(arg)
                elif isinstance(arg, list):
                    if all(isinstance(x, int) for x in arg):
                        converted_args.append(jl.convert(jl.Array[jl.Int], arg))
                    elif all(isinstance(x, float) for x in arg):
                        converted_args.append(jl.convert(jl.Array[jl.Float], arg))
                    else:
                        converted_args.append(arg)
                else:
                    converted_args.append(arg)

            # TODO: test to see if there's a problem here with the returned array elements still being Julia objects
            return np.asarray(
                jl.broadcast(
                    getattr(jl_module, _func_name),
                    jl.convert(jl_cls_type, self),
                    *converted_args,
                )
            )

    return wrapper
