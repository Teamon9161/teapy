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

.PHONY: test
test:
	@pytest -n auto
	# @pytest -s


.PHONY: pytest-cov
pytest-cov: venv  ## test with coverage report
	@pytest python/teapy/tests \
	-s \
	--cov=teapy \
	--cov-report xml \
	--import-mode=importlib

.PHONY: format
format:  ## format and check
	ruff check --fix
	cargo fmt --all
	ruff check
	cargo clippy -- -D warnings

.PHONY: coverage
coverage: # rust and python coverage
	@bash -c "\
		$(MAKE) venv; \
		source venv/bin/activate; \
		source <(cargo llvm-cov show-env --export-prefix); \
		export CARGO_TARGET_DIR=\$$CARGO_LLVM_COV_TARGET_DIR; \
		export CARGO_INCREMENTAL=1; \
		cargo llvm-cov clean --workspace; \
		cd tea-py; \
		maturin develop; \
		$(MAKE) pytest-cov; \
		cargo llvm-cov report --lcov --output-path coverage.lcov; \
		# cargo llvm-cov report -p tears -p py_teapy --lcov --output-path coverage.lcov; \
		"

.PHONY: release_native
release_native:
	maturin develop --release -- -C target-cpu=native

.PHONY: release
release:
	maturin develop --release

.PHONY: debug
debug:
	maturin develop

.PHONY: publish
publish:
	maturin publish -i python -o wheels -u teamon