"""Benchmarks for Python genqo."""

import numpy as np
import genqo as gqpy


def test_probability_success(zalm_py: gqpy.ZALM, test_case_rand: dict, benchmark):
    zalm_py.params.update(test_case_rand)
    zalm_py.run()
    benchmark(zalm_py.calculate_probability_success)

def test_covariance_matrix(zalm_py: gqpy.ZALM, test_case_rand: dict, benchmark):
    zalm_py.params.update(test_case_rand)
    benchmark(zalm_py.calculate_covariance_matrix)

def test_k_function_matrix(zalm_py: gqpy.ZALM,  test_case_rand: dict, benchmark):
    zalm_py.params.update(test_case_rand)
    zalm_py.calculate_covariance_matrix()
    benchmark(zalm_py.calculate_k_function_matrix)

def test_loss_bsm_matrix_fid(zalm_py: gqpy.ZALM, test_case_rand: dict, benchmark):
    zalm_py.params.update(test_case_rand)
    benchmark(zalm_py.calculate_loss_bsm_matrix_fid)

def test_spin_density_matrix(zalm_py: gqpy.ZALM, test_case_rand: dict, benchmark):
    zalm_py.params.update(test_case_rand)
    zalm_py.run()
    benchmark(zalm_py.calculate_density_operator, np.array([1,0,1,1,0,0,1,0]))

def test_fidelity(zalm_py: gqpy.ZALM, test_case_rand: dict, benchmark):
    zalm_py.params.update(test_case_rand)
    zalm_py.run()
    benchmark(zalm_py.calculate_fidelity)
