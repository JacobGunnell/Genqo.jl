"""Python wrapper for Genqo.jl"""

from juliacall import Main as jl
from juliacall import Pkg as jlPkg

from dataclasses import dataclass, field

import numpy as np

jlPkg.activate(".")
jl.seval("using Genqo")


@dataclass
class GenqoParams:
    mean_photon: float = 1e-2
    schmidt_coeffs: list[float] = field(default_factory=lambda: [1.0])
    detection_efficiency: float = 1.0
    bsm_efficiency: float = 1.0
    outcoupling_efficiency: float = 1.0
    dark_counts: int = 0
    visibility: float = 1.0

    @classmethod
    def from_dict(cls, params: dict):
        """
        Create a GenqoParams object from a dictionary of parameters.
        Args:
            params: dictionary of parameters.

        Returns:
            GenqoParams object.

        >>> params = {"mean_photon": 1e-3, "schmidt_coeffs": [1.0]}
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

class TMSV(GenqoParams):
    def covariance_matrix(self):
        return np.asarray(
            jl.tmsv.covariance_matrix(
                jl.GenqoParams(self)
            )
        )
    
    def loss_matrix_pgen(self):
        return np.asarray(
            jl.tmsv.loss_matrix_pgen(
                jl.GenqoParams(self)
            )
        )
    
    def probability_success(self):
        return jl.tmsv.probability_success(
            jl.GenqoParams(self)
        )

class SPDC(GenqoParams):
    def covariance_matrix(self):
        return np.asarray(
            jl.spdc.covariance_matrix(
                jl.GenqoParams(self)
            )
        )
    
    def loss_bsm_matrix_fid(self):
        return np.asarray(
            jl.spdc.loss_bsm_matrix_fid(
                jl.GenqoParams(self)
            )
        )
    
    def spin_density_matrix(self, nvec):
        return np.asarray(
            jl.spdc.spin_density_matrix(
                jl.GenqoParams(self),
                jl.convert(jl.Vector[jl.Int], nvec)
            )
        )
    
    def probability_success(self):
        return jl.spdc.probability_success(
            jl.GenqoParams(self)
        )

class ZALM(GenqoParams):
    def covariance_matrix(self):
        return np.asarray(
            jl.zalm.covariance_matrix(
                jl.GenqoParams(self)
            )
        )
    
    def loss_bsm_matrix_fid(self):
        return np.asarray(
            jl.zalm.loss_bsm_matrix_fid(
                jl.GenqoParams(self)
            )
        )

    def loss_bsm_matrix_pgen(self):
        return np.asarray(
            jl.zalm.loss_bsm_matrix_pgen(
                jl.GenqoParams(self)
            )
        )

    def spin_density_matrix(self, nvec):
        return np.asarray(
            jl.zalm.spin_density_matrix(
                jl.GenqoParams(self),
                jl.convert(jl.Vector[jl.Int], nvec)
            )
        )
    
    def probability_success(self):
        return jl.zalm.probability_success(
            jl.GenqoParams(self)
        )
    
    def fidelity(self):
        return jl.zalm.fidelity(
            jl.GenqoParams(self)
        )
    
def _k_function_matrix(covariance_matrix: np.ndarray) -> np.ndarray:
    return np.asarray(
        jl.tools.k_function_matrix(
            jl.convert(jl.Matrix[jl.Float64], covariance_matrix)
        )
    )
