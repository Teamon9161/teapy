venv: ## make a python venv
	@python -m venv venv
	@venv/bin/pip install -U pip
	@venv/bin/pip install -r build.requirements.txt
	# @source venv/bin/activate && maturin develop

.PHONY: clean
clean: ## clean useless folders
	@rm -rf venv/
	@rm -rf target/
	@rm -rf .hypothesis/
	@rm -rf .pytest_cache/
	@cargo clean

.PHONY: pytest-cov
pytest-cov: venv  ## test with coverage report
	@pytest teapy/tests \
	--cov=teapy \
	--cov-report xml \
	--import-mode=importlib

.PHONY: format
format:  ## format and check
	isort . --profile black
	black .
	cargo fmt --all
	flake8 --ignore E501,F401,F403,W503
	cargo clippy

.PHONY: coverage
coverage: # rust and python coverage
	@bash -c "\
		rustup override set nightly-2022-09-08; \
		$(MAKE) venv; \
		source venv/bin/activate; \
		source <(cargo llvm-cov show-env --export-prefix); \
		export CARGO_TARGET_DIR=\$$CARGO_LLVM_COV_TARGET_DIR; \
		export CARGO_INCREMENTAL=1; \
		cargo llvm-cov clean --workspace; \
		maturin develop; \
		$(MAKE) pytest-cov; \
		cargo llvm-cov --no-run --lcov --output-path coverage.lcov; \
		"

.PHONY: release
release:
	maturin develop --release -- -C target-cpu=native

.PHONY: debug
debug:
	maturin develop
