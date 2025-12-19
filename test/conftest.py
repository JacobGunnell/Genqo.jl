"""Pytest configuration for genqo tests."""

import pytest


# Add any shared fixtures here
@pytest.fixture
def zalm_py():
    """Return a basic ZALM instance from the Python library for testing."""
    import genqo as gqpy
    return gqpy.ZALM()

@pytest.fixture
def zalm_jl():
    """Return a basic ZALM instance from the Julia library for testing."""
    import python.genqo as gqjl
    return gqjl.ZALM()

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
