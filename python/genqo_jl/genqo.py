"""Python wrapper for Genqo.jl"""

from juliacall import Main as jl

from attrs import define, field
from attrs.validators import le, ge

import numpy as np

from .sweep import sweep, _sweepable
    

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
                jl.convert(jl.tmsv.TMSV, self) # TODO: clean up arg type conversions with decorator @_convert_args(self=jl.tmsv.TMSV)
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
    def fidelity(self) -> float:
        return jl.spdc.fidelity(
            jl.convert(jl.spdc.SPDC, self)
        )
        raise NotImplementedError
    

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
