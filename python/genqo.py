"""Python wrapper for Genqo.jl"""

from juliacall import Main as jl
from juliacall import Pkg as jlPkg

from dataclasses import dataclass, field

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
        # Get valid field names for this dataclass
        valid_fields = set(self.__dataclass_fields__.keys())
        
        for key in kwargs.keys():
            if key not in valid_fields:
                raise AttributeError(
                    f"{self.__class__.__name__} has no parameter '{key}'. "
                    f"Valid parameters are: {', '.join(sorted(valid_fields))}"
                )
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
    

@dataclass
class TMSV(GenqoParams):
    mean_photon: float = 1e-2
    detection_efficiency: float = 1.0

    def __post_init__(self):
        if self.mean_photon <= 0:
            raise ValueError("mean_photon must be positive.")
        if not (0 < self.detection_efficiency <= 1):
            raise ValueError("detection_efficiency must be in (0, 1].")

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
    

@dataclass
class SPDC(GenqoParams):
    mean_photon: float = 1e-2
    detection_efficiency: float = 1.0
    bsm_efficiency: float = 1.0
    outcoupling_efficiency: float = 1.0

    def __post_init__(self):
        if self.mean_photon <= 0:
            raise ValueError("mean_photon must be positive.")
        for eff_name in ["detection_efficiency", "bsm_efficiency", "outcoupling_efficiency"]:
            eff_value = getattr(self, eff_name)
            if not (0 < eff_value <= 1):
                raise ValueError(f"{eff_name} must be in (0, 1].")

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
    

@dataclass
class ZALM(GenqoParams):
    mean_photon: float = 1e-2
    #schmidt_coeffs: list[float] = field(default_factory=lambda: [1.0])
    detection_efficiency: float = 1.0
    bsm_efficiency: float = 1.0
    outcoupling_efficiency: float = 1.0
    dark_counts: float = 0.0
    #visibility: float = 1.0

    def __post_init__(self):
        if self.mean_photon <= 0:
            raise ValueError("mean_photon must be positive.")
        for eff_name in ["detection_efficiency", "bsm_efficiency", "outcoupling_efficiency", "dark_counts"]:
            eff_value = getattr(self, eff_name)
            if not (0 < eff_value <= 1):
                raise ValueError(f"{eff_name} must be in (0, 1].")
        #if not (0 < self.visibility <= 1):
        #    raise ValueError("visibility must be in (0, 1].")
    
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
