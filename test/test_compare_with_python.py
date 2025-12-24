import genqo as gqpy
import python.genqo as gqjl

import numpy as np


tol = 1e-8

error_with_params = lambda params: f"Python-Julia comparison yielded results that do not agree for parameters:\n{'\n'.join([f'{k}={v}' for k, v in params.items()])}"

def test_covariance_matrix(zalm_py: gqpy.ZALM, zalm_jl: gqjl.ZALM, test_cases: list[dict]) -> None:
    for params in test_cases:
        zalm_py.params.update(params)
        zalm_py.calculate_covariance_matrix()
        covariance_matrix_py = zalm_py.results["covariance_matrix"]

        zalm_jl.set(**params)
        covariance_matrix_jl = zalm_jl.covariance_matrix()

        assert np.allclose(covariance_matrix_py, covariance_matrix_jl, atol=tol), error_with_params(params)

def test_k_function_matrix(zalm_py: gqpy.ZALM, zalm_jl: gqjl.ZALM, test_cases: list[dict]) -> None:
    for params in test_cases:
        zalm_py.params.update(params)
        zalm_py.run()
        k_function_matrix_py = zalm_py.results["k_function_matrix"]

        zalm_jl.set(**params)
        k_function_matrix_jl = zalm_jl.k_function_matrix()

        assert np.allclose(k_function_matrix_py, k_function_matrix_jl, atol=tol), error_with_params(params)

def test_loss_bsm_matrix_fid(zalm_py: gqpy.ZALM, zalm_jl: gqjl.ZALM, test_cases: list[dict]) -> None:
    for params in test_cases:
        zalm_py.params.update(params)
        zalm_py.calculate_loss_bsm_matrix_fid()
        loss_bsm_matrix_py = zalm_py.results["loss_bsm_matrix"]

        zalm_jl.set(**params)
        loss_bsm_matrix_jl = zalm_jl.loss_bsm_matrix_fid()

        assert np.allclose(loss_bsm_matrix_py, loss_bsm_matrix_jl, atol=tol), error_with_params(params)

def test_loss_bsm_matrix_pgen(zalm_py: gqpy.ZALM, zalm_jl: gqjl.ZALM, test_cases: list[dict]) -> None:
    for params in test_cases:
        zalm_py.params.update(params)
        zalm_py.calculate_loss_bsm_matrix_pgen()
        loss_bsm_matrix_py = zalm_py.results["loss_bsm_matrix"]

        zalm_jl.set(**params)
        loss_bsm_matrix_jl = zalm_jl.loss_bsm_matrix_pgen()

        assert np.allclose(loss_bsm_matrix_py, loss_bsm_matrix_jl, atol=tol), error_with_params(params)

def test_spin_density_matrix(zalm_py: gqpy.ZALM, zalm_jl: gqjl.ZALM, test_cases: list[dict]) -> None:
    nvec = [1, 0, 1, 1, 0, 0, 1, 0]
    for params in test_cases:
        zalm_py.params.update(params)
        zalm_py.run()
        zalm_py.calculate_density_operator(nvec)
        spin_density_matrix_py = zalm_py.results["output_state"]

        zalm_jl.set(**params)
        spin_density_matrix_jl = zalm_jl.spin_density_matrix(nvec)
        assert np.allclose(spin_density_matrix_py, spin_density_matrix_jl, atol=tol), error_with_params(params)

def test_probability_success(zalm_py: gqpy.ZALM, zalm_jl: gqjl.ZALM, test_cases: list[dict]) -> None:
    for params in test_cases:
        zalm_py.params.update(params)
        zalm_py.run()
        zalm_py.calculate_probability_success()
        prob_success_py = zalm_py.results["probability_success"]

        zalm_jl.set(**params)
        prob_success_jl = zalm_jl.probability_success()

        assert np.isclose(prob_success_py, prob_success_jl, atol=tol), error_with_params(params)
