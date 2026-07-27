# XAI-Bench — common tasks. Run `make help` for the list.
.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install smoke test lint format check build-results validate figures web assets reproduce serve clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev extras
	$(PY) -m pip install -e ".[dev]"

smoke:  ## Fast end-to-end pipeline check (CPU, ~30 s, no downloads)
	$(PY) scripts/smoke_test.py

test:  ## Run the unit test suite
	$(PY) -m pytest tests -v

lint:  ## Lint with ruff
	$(PY) -m ruff check xai_bench scripts tests

format:  ## Auto-fix lint issues
	$(PY) -m ruff check --fix xai_bench scripts tests

check: lint test smoke  ## Lint + tests + smoke test

build-results:  ## Consolidate raw run summaries into canonical records
	$(PY) scripts/build_results.py

validate:  ## Validate the published results against the schema
	$(PY) scripts/validate_results.py

figures:  ## Generate paper figures from the records
	$(PY) scripts/generate_figures.py

web:  ## Export JSON for the website
	$(PY) scripts/export_web_data.py

assets:  ## Export web-optimised figure assets into website/public/figures
	$(PY) scripts/export_web_assets.py

reproduce: build-results validate figures web assets  ## Records -> validation -> figures -> website
	@echo "Done. Serve the site with: make serve"

serve:  ## Serve the website on http://localhost:8000
	$(PY) -m http.server 8000 --directory website

clean:  ## Remove caches and build artefacts
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
