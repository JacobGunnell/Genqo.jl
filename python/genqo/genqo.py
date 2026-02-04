"""Python wrapper for Genqo.jl"""

from juliacall import Main as jl

from attrs import define, field
from attrs.validators import le, ge
from typing import get_type_hints
from functools import wraps

import numpy as np

from .sweep import sweep, _sweepable


def _convert_args(func: callable) -> callable:
    if get_type_hints(func).get("return") == np.ndarray:
        post_call = np.asarray
    else:
        post_call = lambda x: x

    @wraps(func)
    def wrapper(self, *args, _post_call: callable = post_call):
        # Python to Julia argument conversion
        converted_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                if arg.dtype == np.complex128:
                    converted_args.append(jl.convert(jl.Array[jl.ComplexF64], arg))
                elif arg.dtype == np.complex64:
                    converted_args.append(jl.convert(jl.Array[jl.ComplexF32], arg))
                elif arg.dtype == np.float64:
                    converted_args.append(jl.convert(jl.Array[jl.Float64], arg))
                elif arg.dtype == np.float32:
                    converted_args.append(jl.convert(jl.Array[jl.Float32], arg))
                elif arg.dtype == np.int_:
                    converted_args.append(jl.convert(jl.Array[jl.Int], arg))
                else:
                    raise TypeError(f"Unsupported numpy array dtype {arg.dtype} for argument conversion to Julia.")
            else:
                raise TypeError(f"Unsupported argument type {type(arg)} for conversion to Julia.")
            
        return _post_call(
            func(self, *converted_args)
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

    @_convert_args
    @_sweepable
    def covariance_matrix(self) -> np.ndarray:
        return jl.tmsv.covariance_matrix(self)
        
    @_convert_args
    @_sweepable
    def loss_matrix_pgen(self) -> np.ndarray:
        return jl.tmsv.loss_matrix_pgen(self)
    
    @_convert_args
    @_sweepable
    def probability_success(self) -> float:
        return jl.tmsv.probability_success(self)
    

@define
class SPDC(GenqoBase):
    mean_photon: float = field(default=1e-2, validator=ge(0.0))
    detection_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])
    bsm_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])
    outcoupling_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])

    @_convert_args
    @_sweepable
    def covariance_matrix(self) -> np.ndarray:
        return jl.spdc.covariance_matrix(self)
    
    @_convert_args
    @_sweepable
    def loss_bsm_matrix_fid(self) -> np.ndarray:
        return jl.spdc.loss_bsm_matrix_fid(self)
    
    @_convert_args
    @_sweepable
    def spin_density_matrix(self, nvec: np.ndarray) -> np.ndarray:
        return jl.spdc.spin_density_matrix(self, nvec)
    
    @_convert_args
    @_sweepable
    def fidelity(self) -> float:
        return jl.spdc.fidelity(self)
    

@define
class ZALM(GenqoBase):
    mean_photon: float = field(default=1e-2, validator=ge(0.0))
    #schmidt_coeffs: list[float] = field(default_factory=lambda: [1.0])
    detection_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])
    bsm_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])
    outcoupling_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])
    dark_counts: float = field(default=0.0, validator=ge(0.0))
    #visibility: float = 1.0
    
    @_convert_args
    @_sweepable
    def covariance_matrix(self) -> np.ndarray:
        return jl.zalm.covariance_matrix(self)
    
    @_convert_args
    @_sweepable
    def loss_bsm_matrix_fid(self) -> np.ndarray:
        return jl.zalm.loss_bsm_matrix_fid(self)

    @_convert_args
    @_sweepable
    def loss_bsm_matrix_pgen(self) -> np.ndarray:
        return jl.zalm.loss_bsm_matrix_pgen(self)

    @_convert_args
    @_sweepable
    def spin_density_matrix(self, nvec: np.ndarray) -> np.ndarray:
        return jl.zalm.spin_density_matrix(self, nvec)
    
    @_convert_args
    @_sweepable
    def probability_success(self) -> float:
        return jl.zalm.probability_success(self)
    
    @_convert_args
    @_sweepable
    def fidelity(self) -> float:
        return jl.zalm.fidelity(self)
    
    
def k_function_matrix(covariance_matrix: np.ndarray) -> np.ndarray:
    return np.asarray(
        jl.tools.k_function_matrix(
            jl.convert(jl.Matrix[jl.Float64], covariance_matrix)
        )
    )


@define
class SIGSAG(GenqoBase):
    mean_photon: float = field(default=1e-2, validator=ge(0.0))
    detection_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])
    bsm_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])
    outcoupling_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])
    
    @_convert_args
    @_sweepable
    def covariance_matrix(self) -> np.ndarray:
        return jl.sigsag.covariance_matrix(self)
    
    @_convert_args
    @_sweepable
    def loss_bsm_matrix_fid(self) -> np.ndarray:
        return jl.sigsag.loss_bsm_matrix_fid(self)
    
    @_convert_args
    @_sweepable
    def probability_success(self) -> float:
        return jl.sigsag.probability_success(self)
    
    @_convert_args
    @_sweepable
    def fidelity(self) -> float:
        return jl.sigsag.fidelity(self)
