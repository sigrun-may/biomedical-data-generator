# MIT: https://opensource.org/licenses/MIT

SRC      := biomedical_data_generator
TESTS    := tests
EXAMPLES := examples
PY       := poetry run

.DEFAULT_GOAL := help

.PHONY: help install update check lint typecheck format fmt fix test cov build clean distclean

help: ## Show this help
	@echo "Makefile commands:"
	@awk -F':.*##' '/^[a-zA-Z0-9][a-zA-Z0-9_-]+:.*##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: ## Install with lint+test+doc groups
	poetry lock
	poetry install --with lint,test,doc

update: ## Update dependencies
	poetry update

check: lint typecheck test ## Run full CI-style checks

lint: ## Ruff lint (inkl. Imports & Docstrings) + Black check + Markdown check
	$(PY) ruff check $(SRC) $(EXAMPLES) $(TESTS)
	$(PY) black --check --diff $(SRC) $(EXAMPLES) $(TESTS)
	$(PY) mdformat --check *.md

typecheck: ## Mypy type check
	$(PY) mypy --install-types --non-interactive $(SRC) $(EXAMPLES) $(TESTS)

format fmt fix: ## Auto-fix with Ruff (quick-fixes) + Black + Mdformat
	$(PY) ruff check --fix $(SRC) $(EXAMPLES) $(TESTS)
	$(PY) black $(SRC) $(EXAMPLES) $(TESTS)
	$(PY) mdformat *.md

test: ## Run unit tests
	$(PY) pytest

cov: ## Run tests with coverage
	$(PY) pytest --cov=$(SRC) --cov-report=term-missing

build: ## Build sdist and wheel
	poetry build

clean: ## Remove build and cache artifacts
	rm -rf .mypy_cache .pytest_cache .ruff_cache .coverage* htmlcov
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +

distclean: clean ## Also remove Poetry venv
	-poetry env remove --all
