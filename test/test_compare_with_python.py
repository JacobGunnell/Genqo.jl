import genqo as gqpy
import python.genqo as gqjl

import numpy as np


tol = 1e-8

error_with_params = lambda params: f"Python-Julia comparison yielded results that do not agree for parameters:\n{'\n'.join([f'{k}={v}' for k, v in params.items()])}"

# TMSV tests


# SPDC tests

def test_spdc__covariance_matrix(spdc_py: gqpy.SPDC, spdc_jl: gqjl.SPDC, test_cases: list[dict]) -> None:
    for params in test_cases:
        spdc_py.params.update(params)
        covariance_matrix_py = spdc_py.spdc_covar(params["mean_photon"])

        spdc_jl.set(**params)
        covariance_matrix_jl = spdc_jl.covariance_matrix()

        assert np.allclose(covariance_matrix_py, covariance_matrix_jl, atol=tol), error_with_params(params)

def test_spdc__loss_bsm_matrix_fid(spdc_py: gqpy.SPDC, spdc_jl: gqjl.SPDC, test_cases: list[dict]) -> None:
    for params in test_cases:
        spdc_py.params.update(params)
        spdc_py.calculate_loss_matrix_fid()
        loss_bsm_matrix_py = spdc_py.results["loss_bsm_matrix"]

        spdc_jl.set(**params)
        loss_bsm_matrix_jl = spdc_jl.loss_bsm_matrix_fid()

        assert np.allclose(loss_bsm_matrix_py, loss_bsm_matrix_jl, atol=tol), error_with_params(params)

def test_spdc__spin_density_matrix(spdc_py: gqpy.SPDC, spdc_jl: gqjl.SPDC, test_cases: list[dict]) -> None:
    nvec = [1, 0, 1, 1, 0, 0, 1, 0]
    for params in test_cases:
        spdc_py.params.update(params)
        spdc_py.run()
        spdc_py.calculate_density_operator(nvec)
        spin_density_matrix_py = spdc_py.results["output_state"]

        spdc_jl.set(**params)
        spin_density_matrix_jl = spdc_jl.spin_density_matrix(nvec)
        assert np.allclose(spin_density_matrix_py, spin_density_matrix_jl, atol=tol), error_with_params(params)


# ZALM tests

def test_zalm__covariance_matrix(zalm_py: gqpy.ZALM, zalm_jl: gqjl.ZALM, test_cases: list[dict]) -> None:
    for params in test_cases:
        zalm_py.params.update(params)
        zalm_py.calculate_covariance_matrix()
        covariance_matrix_py = zalm_py.results["covariance_matrix"]

        zalm_jl.set(**params)
        covariance_matrix_jl = zalm_jl.covariance_matrix()

        assert np.allclose(covariance_matrix_py, covariance_matrix_jl, atol=tol), error_with_params(params)

def test_zalm__loss_bsm_matrix_fid(zalm_py: gqpy.ZALM, zalm_jl: gqjl.ZALM, test_cases: list[dict]) -> None:
    for params in test_cases:
        zalm_py.params.update(params)
        zalm_py.calculate_loss_bsm_matrix_fid()
        loss_bsm_matrix_py = zalm_py.results["loss_bsm_matrix"]

        zalm_jl.set(**params)
        loss_bsm_matrix_jl = zalm_jl.loss_bsm_matrix_fid()

        assert np.allclose(loss_bsm_matrix_py, loss_bsm_matrix_jl, atol=tol), error_with_params(params)

def test_zalm__loss_bsm_matrix_pgen(zalm_py: gqpy.ZALM, zalm_jl: gqjl.ZALM, test_cases: list[dict]) -> None:
    for params in test_cases:
        zalm_py.params.update(params)
        zalm_py.calculate_loss_bsm_matrix_pgen()
        loss_bsm_matrix_py = zalm_py.results["loss_bsm_matrix"]

        zalm_jl.set(**params)
        loss_bsm_matrix_jl = zalm_jl.loss_bsm_matrix_pgen()

        assert np.allclose(loss_bsm_matrix_py, loss_bsm_matrix_jl, atol=tol), error_with_params(params)

def test_zalm__spin_density_matrix(zalm_py: gqpy.ZALM, zalm_jl: gqjl.ZALM, test_cases: list[dict]) -> None:
    nvec = [1, 0, 1, 1, 0, 0, 1, 0]
    for params in test_cases:
        zalm_py.params.update(params)
        zalm_py.run()
        zalm_py.calculate_density_operator(nvec)
        spin_density_matrix_py = zalm_py.results["output_state"]

        zalm_jl.set(**params)
        spin_density_matrix_jl = zalm_jl.spin_density_matrix(nvec)
        assert np.allclose(spin_density_matrix_py, spin_density_matrix_jl, atol=tol), error_with_params(params)

def test_zalm__probability_success(zalm_py: gqpy.ZALM, zalm_jl: gqjl.ZALM, test_cases: list[dict]) -> None:
    for params in test_cases:
        zalm_py.params.update(params)
        zalm_py.run()
        zalm_py.calculate_probability_success()
        prob_success_py = zalm_py.results["probability_success"]

        zalm_jl.set(**params)
        prob_success_jl = zalm_jl.probability_success()

        assert np.isclose(prob_success_py, prob_success_jl, atol=tol), error_with_params(params)

def test_zalm__fidelity(zalm_py: gqpy.ZALM, zalm_jl: gqjl.ZALM, test_cases: list[dict]) -> None:
    for params in test_cases:
        zalm_py.params.update(params)
        zalm_py.run()
        zalm_py.calculate_fidelity()
        fidelity_py = zalm_py.results["fidelity"]

        zalm_jl.set(**params)
        fidelity_jl = zalm_jl.fidelity()

        assert np.isclose(fidelity_py, fidelity_jl, atol=tol), error_with_params(params)


# Other tests
def test_tools__k_function_matrix(zalm_py: gqpy.ZALM, zalm_jl: gqjl.ZALM, test_cases: list[dict]) -> None:
    for params in test_cases:
        zalm_py.params.update(params)
        zalm_py.run()
        k_function_matrix_py = zalm_py.results["k_function_matrix"]

        zalm_jl.set(**params)
        k_function_matrix_jl = gqjl._k_function_matrix(zalm_jl.covariance_matrix())

        assert np.allclose(k_function_matrix_py, k_function_matrix_jl, atol=tol), error_with_params(params)
