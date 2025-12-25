"""Benchmarks for Python genqo."""

import numpy as np
import genqo as gqpy


# TMSV benchmarks

def test_tmsv__probability_success(tmsv_py: gqpy.TMSV, benchmark):
    benchmark(lambda: tmsv_py.run() and tmsv_py.calculate_probability_success())

def test_tmsv__covariance_matrix(tmsv_py: gqpy.TMSV, benchmark):
    benchmark(tmsv_py.calculate_covariance_matrix)

def test_tmsv__loss_matrix(tmsv_py: gqpy.TMSV, benchmark):
    benchmark(tmsv_py.calculate_loss_matrix)


# SPDC benchmarks

def test_spdc__probability_success(spdc_py: gqpy.SPDC, benchmark):
    benchmark(lambda: spdc_py.run() and spdc_py.calculate_probability_success())

def test_spdc__fidelity(spdc_py: gqpy.SPDC, benchmark):
    benchmark(lambda: spdc_py.run() and spdc_py.calculate_fidelity())

def test_spdc__spin_density_matrix(spdc_py: gqpy.SPDC, benchmark):
    benchmark(spdc_py.calculate_density_operator, np.array([1,0,1,1,0,0,1,0]))

def test_spdc__covariance_matrix(spdc_py: gqpy.SPDC, benchmark):
    benchmark(spdc_py.calculate_covariance_matrix)

def test_spdc__loss_bsm_matrix_fid(spdc_py: gqpy.SPDC, benchmark):
    benchmark(spdc_py.calculate_loss_matrix_fid)

def test_spdc__loss_bsm_matrix_trace(spdc_py: gqpy.SPDC, benchmark):
    benchmark(spdc_py.calculate_loss_bsm_matrix_trace)


# ZALM benchmarks

def test_zalm__probability_success(zalm_py: gqpy.ZALM, benchmark):
    benchmark(lambda: zalm_py.run() and zalm_py.calculate_probability_success())

def test_zalm__fidelity(zalm_py: gqpy.ZALM, benchmark):
    benchmark(lambda: zalm_py.run() and zalm_py.calculate_fidelity())

def test_zalm__spin_density_matrix(zalm_py: gqpy.ZALM, benchmark):
    benchmark(zalm_py.calculate_density_operator, np.array([1,0,1,1,0,0,1,0]))

def test_zalm__covariance_matrix(zalm_py: gqpy.ZALM, benchmark):
    benchmark(zalm_py.calculate_covariance_matrix)

def test_zalm__loss_bsm_matrix_fid(zalm_py: gqpy.ZALM, benchmark):
    benchmark(zalm_py.calculate_loss_bsm_matrix_fid)

# Other benchmarks

def test_tools__k_function_matrix(zalm_py: gqpy.ZALM, benchmark):
    zalm_py.calculate_covariance_matrix()
    benchmark(zalm_py.calculate_k_function_matrix)
