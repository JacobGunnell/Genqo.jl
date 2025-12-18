from juliacall import Main as jl
from juliacall import Pkg as jlPkg

from dataclasses import dataclass, field

import numpy as np

jlPkg.activate(".")
jl.seval("using Genqo")


@dataclass
class ZALM:
    mean_photon: float = 1e-2
    schmidt_coeffs: list[float] = field(default_factory=lambda: [1.0])
    detection_efficiency: float = 1.0
    bsm_efficiency: float = 1.0
    outcoupling_efficiency: float = 1.0
    dark_counts: int = 0
    visibility: float = 1.0

    @classmethod
    def from_dict(cls, params: dict):
        return cls(**params)
    
    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def covariance_matrix(self):
        return np.asarray(
            jl.zalm.covariance_matrix(
                jl.zalm.ZALM(self)
            )
        )

    def k_function_matrix(self):
        return np.asarray(
            jl.zalm.k_function_matrix(
                jl.zalm.ZALM(self)
            )
        )

    def density_operator(self, nvec):
        return np.asarray(
            jl.zalm.density_operator(
                jl.zalm.ZALM(self),
                jl.convert(jl.Vector[jl.Int], nvec)
            )
        )
