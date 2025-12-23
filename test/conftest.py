"""Pytest configuration for genqo tests."""

import pytest
import numpy as np

import genqo as gqpy
import python.genqo as gqjl


@pytest.fixture
def test_cases() -> list[dict]:
    """Return a list of test cases (dictionary of parameters)."""
    bsm_efficiencies = [1.0, 0.5, 0.75]
    outcoupling_efficiencies = [1.0, 0.75, 0.5]
    detection_efficiencies = [1.0, 0.6, 0.34]
    mean_photons = [1e-4, 1e-3, 1e-2]

    test_cases = []
    for bsm_efficiency, outcoupling_efficiency, detection_efficiency, mean_photon in zip(bsm_efficiencies, outcoupling_efficiencies, detection_efficiencies, mean_photons):
        test_cases.append({
            "bsm_efficiency": bsm_efficiency,
            "outcoupling_efficiency": outcoupling_efficiency,
            "detection_efficiency": detection_efficiency,
            "mean_photon": mean_photon,
        })
    return test_cases

@pytest.fixture
def test_case_rand() -> dict:
    """Return a random test case (dictionary of parameters)."""
    params = {
        "bsm_efficiency": np.random.uniform(0.5, 1.0),
        "outcoupling_efficiency": np.random.uniform(0.5, 1.0),
        "detection_efficiency": np.random.uniform(0.5, 1.0),
        "mean_photon": 10**np.random.uniform(-5, 1),
    }
    return params

@pytest.fixture
def tmsv_py(test_case_rand: dict) -> gqpy.TMSV:
    tmsv = gqpy.TMSV()
    tmsv.params.update(test_case_rand)
    return tmsv

@pytest.fixture
def tmsv_jl(test_case_rand: dict) -> gqjl.TMSV:
    return gqjl.TMSV().set(**test_case_rand)

@pytest.fixture
def spdc_py(test_case_rand: dict) -> gqpy.SPDC:
    spdc = gqpy.SPDC()
    spdc.params.update(test_case_rand)
    return spdc

@pytest.fixture
def spdc_jl(test_case_rand: dict) -> gqjl.SPDC:
    return gqjl.SPDC().set(**test_case_rand)

@pytest.fixture
def zalm_py(test_case_rand: dict) -> gqpy.ZALM:
    zalm = gqpy.ZALM()
    zalm.params.update(test_case_rand)
    return zalm

@pytest.fixture
def zalm_jl(test_case_rand: dict) -> gqjl.ZALM:
    return gqjl.ZALM().set(**test_case_rand)
