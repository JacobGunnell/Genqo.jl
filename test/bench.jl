using Genqo
using BenchmarkTools

# Get optional function filter from command line argument
func_filter = length(ARGS) > 0 ? ARGS[1] : ""

uniform(min_val, max_val) = min_val + (max_val-min_val)*rand(Float64)
log_uniform(min_exp, max_exp) = 10^uniform(min_exp, max_exp)

suite = BenchmarkGroup()


# TMSV benchmarks
rand_tmsv() = tmsv.TMSV(
    log_uniform(-5, 1),
    uniform(0.5, 1.0),
)
suite["tmsv.covariance_matrix"]      = @benchmarkable tmsv.covariance_matrix(t)           setup=(t=rand_tmsv())
suite["tmsv.loss_matrix_pgen"]       = @benchmarkable tmsv.loss_matrix_pgen(t)            setup=(t=rand_tmsv())
suite["tmsv.probability_success"]    = @benchmarkable tmsv.probability_success(t)         setup=(t=rand_tmsv())


# SPDC benchmarks
rand_spdc() = spdc.SPDC(
    log_uniform(-5, 1),
    uniform(0.5, 1.0),
    uniform(0.5, 1.0),
    uniform(0.5, 1.0),
)
nvec = [0,1,0,1]

suite["spdc.covariance_matrix"]      = @benchmarkable spdc.covariance_matrix(s)           setup=(s=rand_spdc())
suite["spdc.loss_bsm_matrix_fid"]    = @benchmarkable spdc.loss_bsm_matrix_fid(s)         setup=(s=rand_spdc())
suite["spdc.spin_density_matrix"]    = @benchmarkable spdc.spin_density_matrix(s, $nvec)  setup=(s=rand_spdc())
suite["spdc.fidelity"]               = @benchmarkable spdc.fidelity(s)                    setup=(s=rand_spdc())

# ZALM benchmarks
rand_zalm() = zalm.ZALM(
    log_uniform(-5, 1),
    #[one(Float64)],
    uniform(0.5, 1.0),
    uniform(0.5, 1.0),
    uniform(0.5, 1.0),
    zero(Float64),
    #one(Float64)
)
nvec = [1,0,1,1,0,0,1,0]

suite["zalm.covariance_matrix"]      = @benchmarkable zalm.covariance_matrix(z)           setup=(z=rand_zalm())
suite["zalm.loss_bsm_matrix_fid"]    = @benchmarkable zalm.loss_bsm_matrix_fid(z)         setup=(z=rand_zalm())
suite["zalm.spin_density_matrix"]    = @benchmarkable zalm.spin_density_matrix(z, $nvec)  setup=(z=rand_zalm())
suite["zalm.probability_success"]    = @benchmarkable zalm.probability_success(z)         setup=(z=rand_zalm())
suite["zalm.fidelity"]               = @benchmarkable zalm.fidelity(z)                    setup=(z=rand_zalm())


# SIGSAG benchmarks

rand_sigsag() = sigsag.SIGSAG(
    log_uniform(-5, 1),
    uniform(0.5, 1.0),
    uniform(0.5, 1.0),
    uniform(0.5, 1.0),
)

suite["sigsag.covariance_matrix"]      = @benchmarkable sigsag.covariance_matrix(s)           setup=(s=rand_sigsag())
suite["sigsag.loss_bsm_matrix_fid"]    = @benchmarkable sigsag.loss_bsm_matrix_fid(s)         setup=(s=rand_sigsag())
suite["sigsag.probability_success"]    = @benchmarkable sigsag.probability_success(s)         setup=(s=rand_sigsag())
suite["sigsag.fidelity"]               = @benchmarkable sigsag.fidelity(s)                    setup=(s=rand_sigsag())


# Other benchmarks
suite["tools.k_function_matrix"]     = @benchmarkable tools.k_function_matrix(cov) setup=(cov=zalm.covariance_matrix(rand_zalm()))
suite["linsweep_1d"]                 = @benchmarkable tmsv.probability_success.(range(1e-4, stop=1e-2, length=100), 0.7)
suite["linsweep_2d"]                 = @benchmarkable tmsv.probability_success.(range(1e-4, stop=1e-2, length=100), [0.2, 0.5, 0.6, 0.7, 0.9]')
suite["logsweep_1d"]                 = @benchmarkable tmsv.probability_success.(logrange(1e-4, 1e-2, length=100), 0.7)


# Filter suite based on func_filter if provided
if !isempty(func_filter)
    filtered_suite = BenchmarkGroup()
    for (name, benchmark) in suite
        if occursin(func_filter, name)
            filtered_suite[name] = benchmark
        end
    end
    suite = filtered_suite
    if isempty(suite)
        @warn "No benchmarks matched filter: $func_filter"
    end
end

results = run(suite)
for (func, trial) in results
    println("$func:")
    display(trial)
    println()
end
BenchmarkTools.save(".benchmarks/jl-bench.json", results)
