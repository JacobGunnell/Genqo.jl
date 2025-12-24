using Genqo
using BenchmarkTools


uniform(min_val, max_val) = min_val + (max_val-min_val)*rand(Float64)
log_uniform(min_exp, max_exp) = 10^uniform(min_exp, max_exp)

rand_zalm()::zalm.ZALM = zalm.ZALM(
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

# SPDC benchmarks

# ZALM benchmarks
suite["zalm.covariance_matrix"]    = @benchmarkable zalm.covariance_matrix(z)           setup=(z=rand_zalm())
suite["zalm.k_function_matrix"]    = @benchmarkable zalm.k_function_matrix(z)           setup=(z=rand_zalm())
suite["zalm.loss_bsm_matrix_fid"]  = @benchmarkable zalm.loss_bsm_matrix_fid(z)         setup=(z=rand_zalm())
suite["zalm.spin_density_matrix"]  = @benchmarkable zalm.spin_density_matrix(z, $nvec)  setup=(z=rand_zalm())
suite["zalm.probability_success"]  = @benchmarkable zalm.probability_success(z)         setup=(z=rand_zalm())

results = run(suite)
for (func, trial) in results
    println("$func:")
    display(trial)
    println()
end
BenchmarkTools.save(".benchmarks/jl-bench.json", results)
