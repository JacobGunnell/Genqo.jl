"""Python wrapper for Genqo.jl"""

from juliacall import Main as jl

from attrs import define, field

import numpy as np


def _to_jl_array(arr: np.ndarray):
    """Convert a numpy array to a Julia array with matching element type."""
    if arr.dtype == np.complex128:
        return jl.convert(jl.Array[jl.ComplexF64], arr)
    elif arr.dtype == np.complex64:
        return jl.convert(jl.Array[jl.ComplexF32], arr)
    elif arr.dtype == np.float64:
        return jl.convert(jl.Array[jl.Float64], arr)
    elif arr.dtype == np.float32:
        return jl.convert(jl.Array[jl.Float32], arr)
    elif np.issubdtype(arr.dtype, np.integer):
        return jl.convert(jl.Array[jl.Int], arr)
    else:
        raise TypeError(f"Unsupported numpy array dtype {arr.dtype}")


def _jl_call(jl_func, *args, ref_args=()):
    """Call a Julia function, automatically broadcasting if any arg is a numpy array.

    Shape the arrays to control broadcasting, e.g. a (100,) array broadcast
    against a (50,1) array produces a (50,100) result -- standard Julia/numpy
    broadcasting rules.

    Parameters
    ----------
    jl_func : Julia function reference
        The Julia function to call (e.g. ``jl.zalm.fidelity``).
    *args : float or np.ndarray
        Arguments passed positionally to ``jl_func``. Scalars are passed
        through; ndarrays are converted to Julia arrays and broadcast.
    ref_args : tuple of np.ndarray or scalar
        Extra arguments (e.g. ``nvec``) that should **not** be broadcast
        element-wise.  During a broadcast call they are wrapped in
        ``Ref()`` so Julia treats them as single values.

    Returns
    -------
    float, np.ndarray, or Julia value
        Scalar result when no broadcasting is needed, otherwise a numpy
        array of broadcast results.
    """
    needs_broadcast = any(isinstance(a, np.ndarray) and a.ndim > 0 for a in args)

    # Convert ref_args to Julia types
    converted_ref = []
    for a in ref_args:
        if isinstance(a, np.ndarray):
            jl_a = _to_jl_array(a)
            converted_ref.append(jl.Ref(jl_a) if needs_broadcast else jl_a)
        else:
            converted_ref.append(a)

    if not needs_broadcast:
        result = jl_func(*args, *converted_ref)
        return result

    # Convert array args to Julia arrays for broadcast
    jl_args = []
    for a in args:
        if isinstance(a, np.ndarray):
            jl_args.append(_to_jl_array(a))
        else:
            jl_args.append(a)

    return np.asarray(jl.broadcast(jl_func, *jl_args, *converted_ref))

def _ge(bound):
    """attrs validator: value >= bound, works with scalars and ndarrays."""
    def validator(instance, attribute, value):
        if isinstance(value, np.ndarray):
            if not np.all(value >= bound):
                raise ValueError(f"All elements of {attribute.name} must be >= {bound}")
        elif value < bound:
            raise ValueError(f"{attribute.name} must be >= {bound}")
    return validator

def _le(bound):
    """attrs validator: value <= bound, works with scalars and ndarrays."""
    def validator(instance, attribute, value):
        if isinstance(value, np.ndarray):
            if not np.all(value <= bound):
                raise ValueError(f"All elements of {attribute.name} must be <= {bound}")
        elif value > bound:
            raise ValueError(f"{attribute.name} must be <= {bound}")
    return validator


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
    mean_photon: float | np.ndarray = field(default=1e-2, validator=_ge(0.0))
    detection_efficiency: float | np.ndarray = field(default=1.0, validator=[_ge(0.0), _le(1.0)])

    def covariance_matrix(self) -> np.ndarray:
        return np.asarray(_jl_call(jl.tmsv.covariance_matrix, self.mean_photon))
        
    def loss_matrix_pgen(self) -> np.ndarray:
        return np.asarray(_jl_call(jl.tmsv.loss_matrix_pgen, self.detection_efficiency))
    
    def probability_success(self):
        return _jl_call(jl.tmsv.probability_success, self.mean_photon, self.detection_efficiency)
    

@define
class SPDC(GenqoBase):
    mean_photon: float | np.ndarray = field(default=1e-2, validator=_ge(0.0))
    detection_efficiency: float | np.ndarray = field(default=1.0, validator=[_ge(0.0), _le(1.0)])
    bsm_efficiency: float | np.ndarray = field(default=1.0, validator=[_ge(0.0), _le(1.0)])
    outcoupling_efficiency: float | np.ndarray = field(default=1.0, validator=[_ge(0.0), _le(1.0)])

    def covariance_matrix(self) -> np.ndarray:
        return np.asarray(_jl_call(jl.spdc.covariance_matrix, self.mean_photon))
    
    def loss_bsm_matrix_fid(self) -> np.ndarray:
        return np.asarray(_jl_call(
            jl.spdc.loss_bsm_matrix_fid, self.outcoupling_efficiency, self.detection_efficiency
        ))
    
    def spin_density_matrix(self, nvec: np.ndarray) -> np.ndarray:
        return np.asarray(_jl_call(
            jl.spdc.spin_density_matrix,
            self.mean_photon, self.outcoupling_efficiency, self.detection_efficiency,
            ref_args=(nvec,)
        ))
    
    def fidelity(self):
        return _jl_call(
            jl.spdc.fidelity, self.mean_photon, self.outcoupling_efficiency, self.detection_efficiency
        )
    

@define
class ZALM(GenqoBase):
    mean_photon: float | np.ndarray = field(default=1e-2, validator=_ge(0.0))
    #schmidt_coeffs: list[float] = field(default_factory=lambda: [1.0])
    detection_efficiency: float | np.ndarray = field(default=1.0, validator=[_ge(0.0), _le(1.0)])
    bsm_efficiency: float | np.ndarray = field(default=1.0, validator=[_ge(0.0), _le(1.0)])
    outcoupling_efficiency: float | np.ndarray = field(default=1.0, validator=[_ge(0.0), _le(1.0)])
    dark_counts: float | np.ndarray = field(default=0.0, validator=_ge(0.0))
    #visibility: float = 1.0
    
    def covariance_matrix(self) -> np.ndarray:
        return np.asarray(_jl_call(jl.zalm.covariance_matrix, self.mean_photon))
    
    def loss_bsm_matrix_fid(self) -> np.ndarray:
        return np.asarray(_jl_call(
            jl.zalm.loss_bsm_matrix_fid,
            self.outcoupling_efficiency, self.detection_efficiency, self.bsm_efficiency
        ))

    def loss_bsm_matrix_pgen(self) -> np.ndarray:
        return np.asarray(_jl_call(
            jl.zalm.loss_bsm_matrix_pgen,
            self.outcoupling_efficiency, self.detection_efficiency, self.bsm_efficiency
        ))

    def spin_density_matrix(self, nvec: np.ndarray) -> np.ndarray:
        return np.asarray(_jl_call(
            jl.zalm.spin_density_matrix,
            self.mean_photon, self.outcoupling_efficiency,
            self.detection_efficiency, self.bsm_efficiency,
            ref_args=(nvec,)
        ))
    
    def probability_success(self):
        return _jl_call(
            jl.zalm.probability_success,
            self.mean_photon, self.outcoupling_efficiency,
            self.detection_efficiency, self.bsm_efficiency, self.dark_counts
        )
    
    def fidelity(self):
        return _jl_call(
            jl.zalm.fidelity,
            self.mean_photon, self.outcoupling_efficiency,
            self.detection_efficiency, self.bsm_efficiency
        )
    
    
def k_function_matrix(covariance_matrix: np.ndarray) -> np.ndarray:
    return np.asarray(
        jl.tools.k_function_matrix(
            jl.convert(jl.Matrix[jl.Float64], covariance_matrix)
        )
    )


@define
class SIGSAG(GenqoBase):
    mean_photon: float | np.ndarray = field(default=1e-2, validator=_ge(0.0))
    detection_efficiency: float | np.ndarray = field(default=1.0, validator=[_ge(0.0), _le(1.0)])
    bsm_efficiency: float | np.ndarray = field(default=1.0, validator=[_ge(0.0), _le(1.0)])
    outcoupling_efficiency: float | np.ndarray = field(default=1.0, validator=[_ge(0.0), _le(1.0)])
    
    def covariance_matrix(self) -> np.ndarray:
        return np.asarray(_jl_call(jl.sigsag.covariance_matrix, self.mean_photon))
    
    def loss_bsm_matrix_fid(self) -> np.ndarray:
        return np.asarray(_jl_call(
            jl.sigsag.loss_bsm_matrix_fid, self.outcoupling_efficiency, self.detection_efficiency
        ))
    
    def probability_success(self):
        return _jl_call(
            jl.sigsag.probability_success,
            self.mean_photon, self.outcoupling_efficiency, self.detection_efficiency
        )
    
    def fidelity(self):
        return _jl_call(
            jl.sigsag.fidelity,
            self.mean_photon, self.outcoupling_efficiency, self.detection_efficiency
        )
