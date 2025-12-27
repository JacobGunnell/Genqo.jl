# Install genqo and its dependencies
install:
    julia --project=. -e 'using Pkg; Pkg.instantiate()'

    just venv
    . python/.venv/bin/activate && \
    pip install -r python/requirements.txt

# Run benchmarks for <func>, e.g. just bench spdc.spin_density_matrix (benchmarks all by default)
bench func="":
    @echo "Running benchmarks for Julia and Python genqo..."
    mkdir -p .benchmarks
    julia --project=. test/bench.jl "{{func}}"

    . python/.venv/bin/activate && \
    pytest test/test_gqpy_bench.py{{ if func != "" { "::test_" + replace(func, '.', '__') } else { "" } }} --benchmark-json=.benchmarks/py-bench.json && \
    python test/plot_comparison.py

# Create virtual environment for Python wrapper
venv:
    if [ ! -d python/.venv ]; then \
        python3 -m venv python/.venv; \
    fi
