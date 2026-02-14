# Install genqo.jl and its Python wrapper
install:
    julia --project=. -e 'using Pkg; Pkg.instantiate()'

    just venv
    . python/.venv/bin/activate && \
    pip install -e python/[test] && \
    pip install -e test/genqo_old_pkg

# Run tests comparing Julia and Python genqo implementations
test:
    @echo "Running tests comparing Julia and Python genqo..."
    . python/.venv/bin/activate && \
    pytest test/python/test_compare_with_python.py

# Run benchmarks for <func>, e.g. just bench spdc.spin_density_matrix (benchmarks all by default)
bench func="":
    @echo "Running benchmarks for Julia and Python genqo..."
    mkdir -p .benchmarks
    julia --project=. test/bench.jl "{{func}}"

    . python/.venv/bin/activate && \
    pytest test/python/test_gqpy_bench.py{{ if func != "" { "::test_" + replace(func, '.', '__') } else { "" } }} --benchmark-json=.benchmarks/py-bench.json && \
    python test/python/plot_comparison.py

# Create virtual environment for Python wrapper
venv:
    if [ ! -d python/.venv ]; then \
        python3 -m venv python/.venv; \
    fi
