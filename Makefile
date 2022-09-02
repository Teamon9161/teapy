.PHONY: clean pip pytest-cov coverage

make_venv:
	@python -m venv venv
	@venv/bin/pip install -U pip
	@venv/bin/pip install -r build.requirements.txt
	# @source venv/bin/activate && maturin develop

clean:
	@-rm -r venv
	@cargo clean

pytest-cov: make_venv
	@pytest teapy/tests \
	--cov=teapy \
	--cov-report xml \
	--import-mode=importlib

coverage: 
	@bash -c "\
		$(MAKE) make_venv; \
		source venv/bin/activate; \
		source <(cargo llvm-cov show-env --export-prefix); \
		export CARGO_TARGET_DIR=\$$CARGO_LLVM_COV_TARGET_DIR; \
		export CARGO_INCREMENTAL=1; \
		cargo llvm-cov clean --workspace; \
		maturin develop; \
		$(MAKE) pytest-cov; \
		cargo llvm-cov --no-run --lcov --output-path coverage.lcov; \
		"