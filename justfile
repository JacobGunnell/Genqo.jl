install:
    julia --project=. -e 'using Pkg; Pkg.instantiate()'

    just venv
    . python/.venv/bin/activate && \
    pip install -r python/requirements.txt

bench:
    @echo "Running benchmarks for Julia and Python genqo..."
    mkdir -p .benchmarks
    julia --project=. test/bench.jl

    . python/.venv/bin/activate && \
    pytest test/test_gqpy_bench.py --benchmark-json=.benchmarks/py-bench.json && \
    python test/plot_comparison.py

venv:
    if [ ! -d python/.venv ]; then \
        python3 -m venv python/.venv; \
    fi
