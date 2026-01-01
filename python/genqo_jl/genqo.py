"""Python wrapper for Genqo.jl"""

from juliacall import Main as jl
from juliacall import Pkg as jlPkg

from abc import ABC
from attrs import define, field, fields
from attrs.validators import le, ge
from functools import wraps

import numpy as np

jlPkg.activate(".")
jl.seval("using Genqo")


# Class telling genqo to sweep a parameter
class sweep(ABC):
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

class linsweep(sweep):
    """Class representing a linear sweep from start to stop with a given length."""    
    def __array__(self) -> np.ndarray:
        """Convert sweep to numpy array for use with matplotlib and numpy operations."""
        return np.linspace(self.start, self.stop, self.length)
    
class logsweep(sweep):
    """Class representing a logarithmic sweep from start to stop with a given length."""  
    def __array__(self) -> np.ndarray:
        """Convert sweep to numpy array for use with matplotlib and numpy operations."""
        return np.logspace(np.log10(self.start), np.log10(self.stop), self.length)

# Decorator to support sweeping by setting a particular parameter to a sweep object: zalm.mean_photon = gq.logsweep(1e-4, 1e-2, 100)
def _sweepable(func: callable) -> callable:
    cls = func.__qualname__.split(".")[0]
    module = cls.lower()
    jl_module = getattr(jl, module)
    jl_cls_type = getattr(jl_module, cls)
        
    @wraps(func)
    def wrapper(self, _func_name: str = func.__name__):
        # If no sweeping is required, call the function once directly
        if not any(isinstance(getattr(self, param.name), sweep) for param in fields(self.__class__)):
            return func(self)

        # If sweeping is required, perform fast broadcast sweep in Julia
        else:
            # TODO: support function args other than self
            # converted_args = []
            # for arg in args:
            #     if (jl_type := _jl_types.get(type(arg))) is not None:
            #         converted_args.append(jl.convert(jl_type, arg))
            #     else:
            #         converted_args.append(arg)

            # TODO: test to see if there's a problem here with the returned array elements still being Julia objects
            return np.asarray(
                jl.broadcast(
                    getattr(jl_module, _func_name),
                    jl.convert(jl_cls_type, self),
                    # *converted_args,
                )
            )

    return wrapper
    

class GenqoBase:
    @classmethod
    def from_dict(cls, params: dict):
        """
        Create a GenqoParams object from a dictionary of parameters.
        Args:
            params: dictionary of parameters.

        Returns:
            GenqoParams object.

        >>> params = {"mean_photon": 1e-3, "detection_efficiency": 0.9}
        >>> zalm = ZALM.from_dict(params)
        """
        return cls(**params)
    
    def set(self, **kwargs):
        """
        Set the parameters of the GenqoParams object.
        Args:
            **kwargs: keyword arguments to set the parameters.
        
        >>> zalm = ZALM()
        >>> zalm.set(mean_photon=1e-3)
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
    

@define
class TMSV(GenqoBase):
    mean_photon: float | sweep = field(default=1e-2, validator=ge(0.0))
    detection_efficiency: float | sweep = field(default=1.0, validator=[ge(0.0), le(1.0)])

    @_sweepable
    def covariance_matrix(self) -> np.ndarray:
        return np.asarray(
            jl.tmsv.covariance_matrix(
                jl.convert(jl.tmsv.TMSV, self)
            )
        )
    
    @_sweepable
    def loss_matrix_pgen(self) -> np.ndarray:
        return np.asarray(
            jl.tmsv.loss_matrix_pgen(
                jl.convert(jl.tmsv.TMSV, self)
            )
        )
    
    @_sweepable
    def probability_success(self) -> float:
        return jl.tmsv.probability_success(
            jl.convert(jl.tmsv.TMSV, self)
        )
    

@define
class SPDC(GenqoBase):
    mean_photon: float = field(default=1e-2, validator=ge(0.0))
    detection_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])
    bsm_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])
    outcoupling_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])

    @_sweepable
    def covariance_matrix(self) -> np.ndarray:
        return np.asarray(
            jl.spdc.covariance_matrix(
                jl.convert(jl.spdc.SPDC, self)
            )
        )
    
    @_sweepable
    def loss_bsm_matrix_fid(self) -> np.ndarray:
        return np.asarray(
            jl.spdc.loss_bsm_matrix_fid(
                jl.convert(jl.spdc.SPDC, self)
            )
        )
    
    def spin_density_matrix(self, nvec: np.ndarray) -> np.ndarray:
        return np.asarray(
            jl.spdc.spin_density_matrix(
                jl.convert(jl.spdc.SPDC, self),
                jl.convert(jl.Vector[jl.Int], nvec)
            )
        )
    
    @_sweepable
    def probability_success(self) -> float:
        return jl.spdc.probability_success(
            jl.convert(jl.spdc.SPDC, self)
        )
    

@define
class ZALM(GenqoBase):
    mean_photon: float = field(default=1e-2, validator=ge(0.0))
    #schmidt_coeffs: list[float] = field(default_factory=lambda: [1.0])
    detection_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])
    bsm_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])
    outcoupling_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])
    dark_counts: float = field(default=0.0, validator=ge(0.0))
    #visibility: float = 1.0
    
    @_sweepable
    def covariance_matrix(self) -> np.ndarray:
        return np.asarray(
            jl.zalm.covariance_matrix(
                jl.convert(jl.zalm.ZALM, self)
            )
        )
    
    @_sweepable
    def loss_bsm_matrix_fid(self) -> np.ndarray:
        return np.asarray(
            jl.zalm.loss_bsm_matrix_fid(
                jl.convert(jl.zalm.ZALM, self)
            )
        )

    @_sweepable
    def loss_bsm_matrix_pgen(self) -> np.ndarray:
        return np.asarray(
            jl.zalm.loss_bsm_matrix_pgen(
                jl.convert(jl.zalm.ZALM, self)
            )
        )

    def spin_density_matrix(self, nvec: np.ndarray) -> np.ndarray:
        return np.asarray(
            jl.zalm.spin_density_matrix(
                jl.convert(jl.zalm.ZALM, self),
                jl.convert(jl.Vector[jl.Int], nvec)
            )
        )
    
    @_sweepable
    def probability_success(self) -> float:
        return jl.zalm.probability_success(
            jl.convert(jl.zalm.ZALM, self)
        )
    
    @_sweepable
    def fidelity(self) -> float:
        return jl.zalm.fidelity(
            jl.convert(jl.zalm.ZALM, self)
        )
    
    
def k_function_matrix(covariance_matrix: np.ndarray) -> np.ndarray:
    return np.asarray(
        jl.tools.k_function_matrix(
            jl.convert(jl.Matrix[jl.Float64], covariance_matrix)
        )
    )
