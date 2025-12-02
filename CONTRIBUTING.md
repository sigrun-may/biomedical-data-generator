# Contributing to `biomedical-data-generator`

Thank you for considering a contribution to **`biomedical-data-generator`**!\
This project aims to provide high-quality synthetic biomedical datasets for teaching, benchmarking, and method development in high-dimensional settings.\
Contributions that improve robustness, clarity, and usability are very welcome.

______________________________________________________________________

## How Can I Contribute?

There are many ways to help:

- **Report bugs** (and ideally add a minimal reproducible example)
- **Suggest enhancements** to the API, configuration, or documentation
- **Add tests** (unit tests, regression tests, edge cases)
- **Improve documentation**: docstrings, Sphinx docs, examples, README
- **Add new generators** (e.g., new correlation structures, batch models)
- **Improve performance** or numerical robustness

If you are unsure whether an idea fits, feel free to open an issue first and discuss it.

______________________________________________________________________

## Code of Conduct

Please be respectful and constructive in all interactions.\
Be kind, assume good intent, and focus on technical issues rather than individuals.

______________________________________________________________________

## Getting Started

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/sigrun-may/biomedical-data-generator.git
cd biomedical-data-generator
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/short-bug-description
```

Short, descriptive names are encouraged.

______________________________________________________________________

## Development Environment

This project uses **Python 3.11** and **Poetry** for dependency management.

### 1. Install Poetry (if needed)

Follow the official Poetry instructions: <https://python-poetry.org/docs/#installation>

### 2. Install Dependencies

From the project root:

```bash
poetry install
```

If you want to work on docs and development tooling (recommended):

```bash
poetry install --with dev,docs
```

Use the virtual environment managed by Poetry:

```bash
poetry shell
```

or prefix commands with `poetry run`, e.g.:

```bash
poetry run python -m pytest
```

______________________________________________________________________

## Running Tests

The test suite is based on **pytest**.

```bash
poetry run pytest
```

To run a subset of tests (e.g. only feature-related tests):

```bash
poetry run pytest tests/test_features*
```

Please ensure that:

- All tests pass before you open a pull request.
- New functionality comes with appropriate tests.
- Bug fixes include a regression test when possible.

______________________________________________________________________

## Code Style and Type Checking

The project aims for clear, maintainable, and type-safe code.

### General Guidelines

- Use **type hints** throughout (`from __future__ import annotations` if applicable).
- Prefer **Google-style docstrings** (or the style used in existing modules).
- Keep functions focused and small; favor clarity over micro-optimizations.
- Maintain compatibility with the public API (`DatasetConfig`, `CorrClusterConfig`, `BatchEffectsConfig`, `DatasetMeta`, etc.), especially for released versions.

### Static Analysis

If the repository includes tools like `ruff` or `mypy` (see `pyproject.toml`), please run them locally:

```bash
# Static type checking
poetry run mypy biomedical_data_generator

# Linting / style checks (example)
poetry run ruff check biomedical_data_generator
```

Fix any reported issues before submitting a PR.

______________________________________________________________________

## Documentation

The documentation is built with **Sphinx** and hosted via GitHub Pages.

### Building Docs Locally

From the project root (with the docs extras installed):

```bash
cd docs
poetry run make html
```

This will generate HTML documentation under `docs/_build/html/`.\
Open `docs/_build/html/index.html` in your browser to inspect the result.

### When to Update Docs

Please update the docs if you:

- Add or change public configuration classes
  - e.g. `DatasetConfig`, `ClassConfig`, `CorrClusterConfig`, `BatchEffectsConfig`
- Modify the behavior of `generate_dataset` or related functions
- Introduce new modules, generators, or CLI options
- Change metadata semantics (`DatasetMeta` fields, indices, labels)

Prefer succinct explanations plus small runnable examples.

______________________________________________________________________

## Examples

The `examples/` directory contains small Python scripts demonstrating typical usage.

If you add new functionality that is pedagogically interesting (e.g. new non-causal variation, new correlation structure), consider:

- Adding or extending an example script in `examples/`
- Keeping examples short and focused (no heavy dependencies)

Run an example like this:

```bash
poetry run python examples/01_basic_usage.py
```

______________________________________________________________________

## Command-Line Interface

If you modify or extend the CLI (e.g. `bdg` entry point):

- Maintain backward compatibility where possible.
- Update CLI help strings and documentation.
- Add tests for new CLI behavior (e.g. using `subprocess` or Click/Typer testing utilities, depending on the implementation).

______________________________________________________________________

## Opening an Issue

Before filing a new issue, please:

1. Search the [issue tracker](https://github.com/sigrun-may/biomedical-data-generator/issues) to avoid duplicates.
1. Include:
   - A clear description of the problem or request.
   - Steps to reproduce (for bugs).
   - Minimal code example, if possible.
   - Information about your environment:
     - OS
     - Python version
     - `biomedical-data-generator` version

Issues that come with a minimal reproducible example are much easier to diagnose and fix.

______________________________________________________________________

## Pull Request Guidelines

When you are ready:

1. Ensure your branch is up to date with `main`:

   ```bash
   git fetch origin
   git checkout main
   git pull
   git checkout feature/your-feature-name
   git rebase main
   ```

1. Run tests (and linters, if configured):

   ```bash
   poetry run pytest
   # optional:
   poetry run mypy biomedical_data_generator
   poetry run ruff check biomedical_data_generator
   ```

1. Commit your changes with clear messages:

   ```bash
   git commit -am "Add class-specific correlation example" 
   ```

   Prefer small, focused commits over one huge commit.

1. Push and open a Pull Request on GitHub:

   - Provide a short summary.
   - Describe the motivation and what changed.
   - Reference any related issues (e.g. `Closes #42`).

The CI (GitHub Actions) will run tests and other checks automatically on your PR.

______________________________________________________________________

## Design Principles

When contributing new features or refactoring existing code, please keep in mind:

- **Reproducibility first**

  - Single RNG flow, controlled `random_state`, and explicit metadata.

- **Explicit ground truth**

  - Whenever new structure is introduced (e.g. new correlated blocks, new batch effects), ensure that relevant indices/labels are recorded in `DatasetMeta`.

- **Small, composable building blocks**

  - Each module should have a clear responsibility:
    - label generation
    - informative features
    - correlated clusters
    - noise features
    - batch effects
    - assembly / orchestration
    - metadata

- **Biomedical realism with pedagogical clarity**

  - Effects should be interpretable and explainable in teaching materials.
  - Complex generators should be documented with examples that highlight their use.

______________________________________________________________________

## Questions?

If you are unsure how to implement something or how it fits into the overall design:

- Open an issue with the label `question` or `discussion`, or
- Draft a small PR and mark it as “WIP” (work in progress) to get early feedback.

Thank you again for contributing to `biomedical-data-generator`!
