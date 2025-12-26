using Genqo
using BenchmarkTools


uniform(min_val, max_val) = min_val + (max_val-min_val)*rand(Float64)
log_uniform(min_exp, max_exp) = 10^uniform(min_exp, max_exp)

rand_params()::GenqoParams = GenqoParams(
    log_uniform(-5, 1),
    [one(Float64)],
    uniform(0.5, 1.0),
    uniform(0.5, 1.0),
    uniform(0.5, 1.0),
    zero(Float64),
    one(Float64)
)

nvec = [1,0,1,1,0,0,1,0]

suite = BenchmarkGroup()

# TMSV benchmarks
suite["tmsv.covariance_matrix"]      = @benchmarkable tmsv.covariance_matrix(p)           setup=(p=rand_params())
suite["tmsv.loss_matrix_pgen"]       = @benchmarkable tmsv.loss_matrix_pgen(p)            setup=(p=rand_params())
suite["tmsv.probability_success"]    = @benchmarkable tmsv.probability_success(p)         setup=(p=rand_params())

# SPDC benchmarks
suite["spdc.covariance_matrix"]      = @benchmarkable spdc.covariance_matrix(p)           setup=(p=rand_params())
suite["spdc.loss_bsm_matrix_fid"]    = @benchmarkable spdc.loss_bsm_matrix_fid(p)         setup=(p=rand_params())
suite["spdc.spin_density_matrix"]    = @benchmarkable spdc.spin_density_matrix(p, $nvec)  setup=(p=rand_params())
suite["spdc.probability_success"]    = @benchmarkable spdc.probability_success(p)         setup=(p=rand_params())

# ZALM benchmarks
suite["zalm.covariance_matrix"]      = @benchmarkable zalm.covariance_matrix(p)           setup=(p=rand_params())
suite["zalm.loss_bsm_matrix_fid"]    = @benchmarkable zalm.loss_bsm_matrix_fid(p)         setup=(p=rand_params())
suite["zalm.spin_density_matrix"]    = @benchmarkable zalm.spin_density_matrix(p, $nvec)  setup=(p=rand_params())
suite["zalm.probability_success"]    = @benchmarkable zalm.probability_success(p)         setup=(p=rand_params())
suite["zalm.fidelity"]               = @benchmarkable zalm.fidelity(p)                    setup=(p=rand_params())

# Other benchmarks
suite["tools.k_function_matrix"]     = @benchmarkable tools.k_function_matrix(cov)        setup=(cov=zalm.covariance_matrix(rand_params()))

results = run(suite)
for (func, trial) in results
    println("$func:")
    display(trial)
    println()
end
BenchmarkTools.save(".benchmarks/jl-bench.json", results)
