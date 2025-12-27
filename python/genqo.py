"""Python wrapper for Genqo.jl"""

from juliacall import Main as jl
from juliacall import Pkg as jlPkg

from attrs import define, field
from attrs.validators import le, ge

import numpy as np

jlPkg.activate(".")
jl.seval("using Genqo")


class GenqoParams:
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
class TMSV(GenqoParams):
    mean_photon: float = field(default=1e-2, validator=ge(0.0))
    detection_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])

    def covariance_matrix(self):
        return np.asarray(
            jl.tmsv.covariance_matrix(
                jl.tmsv.TMSV(self)
            )
        )
    
    def loss_matrix_pgen(self):
        return np.asarray(
            jl.tmsv.loss_matrix_pgen(
                jl.tmsv.TMSV(self)
            )
        )
    
    def probability_success(self):
        return jl.tmsv.probability_success(
            jl.tmsv.TMSV(self)
        )
    

@define
class SPDC(GenqoParams):
    mean_photon: float = field(default=1e-2, validator=ge(0.0))
    detection_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])
    bsm_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])
    outcoupling_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])

    def covariance_matrix(self):
        return np.asarray(
            jl.spdc.covariance_matrix(
                jl.spdc.SPDC(self)
            )
        )
    
    def loss_bsm_matrix_fid(self):
        return np.asarray(
            jl.spdc.loss_bsm_matrix_fid(
                jl.spdc.SPDC(self)
            )
        )
    
    def spin_density_matrix(self, nvec):
        return np.asarray(
            jl.spdc.spin_density_matrix(
                jl.spdc.SPDC(self),
                jl.convert(jl.Vector[jl.Int], nvec)
            )
        )
    
    def probability_success(self):
        return jl.spdc.probability_success(
            jl.spdc.SPDC(self)
        )
    

@define
class ZALM(GenqoParams):
    mean_photon: float = field(default=1e-2, validator=ge(0.0))
    #schmidt_coeffs: list[float] = field(default_factory=lambda: [1.0])
    detection_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])
    bsm_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])
    outcoupling_efficiency: float = field(default=1.0, validator=[ge(0.0), le(1.0)])
    dark_counts: float = field(default=0.0, validator=ge(0.0))
    #visibility: float = 1.0
    
    def covariance_matrix(self):
        return np.asarray(
            jl.zalm.covariance_matrix(
                jl.zalm.ZALM(self)
            )
        )
    
    def loss_bsm_matrix_fid(self):
        return np.asarray(
            jl.zalm.loss_bsm_matrix_fid(
                jl.zalm.ZALM(self)
            )
        )

    def loss_bsm_matrix_pgen(self):
        return np.asarray(
            jl.zalm.loss_bsm_matrix_pgen(
                jl.zalm.ZALM(self)
            )
        )

    def spin_density_matrix(self, nvec):
        return np.asarray(
            jl.zalm.spin_density_matrix(
                jl.zalm.ZALM(self),
                jl.convert(jl.Vector[jl.Int], nvec)
            )
        )
    
    def probability_success(self):
        return jl.zalm.probability_success(
            jl.zalm.ZALM(self)
        )
    
    def fidelity(self):
        return jl.zalm.fidelity(
            jl.zalm.ZALM(self)
        )
    
    
def _k_function_matrix(covariance_matrix: np.ndarray) -> np.ndarray:
    return np.asarray(
        jl.tools.k_function_matrix(
            jl.convert(jl.Matrix[jl.Float64], covariance_matrix)
        )
    )
