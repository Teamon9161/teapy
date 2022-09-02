PYTHON=venv/bin/python


make_venv:
	@python -m venv venv
	@venv/bin/pip install -U pip
	@venv/bin/pip install -r build.requirements.txt
	@unset CONDA_PREFIX && source venv/bin/activate && maturin develop

clean:
	@-rm -r venv
	@cargo clean

pytest-cov: $(PYTHON) -m pytest \
		--cov=teapy \
		--cov-report xml \
		--import-mode=importlib

coverage:
	@bash -c "\
		make_venv
		source <(cargo llvm-cov show-env --export-prefix); \
		export CARGO_TARGET_DIR=\$$CARGO_LLVM_COV_TARGET_DIR; \
		export CARGO_INCREMENTAL=1; \
		cargo llvm-cov clean --workspace; \
		source venv/bin/activate; \
		pytest-cov; \
		cargo llvm-cov --no-run --lcov --output-path coverage.lcov; \
		"