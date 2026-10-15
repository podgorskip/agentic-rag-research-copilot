# ADR-001: Project Setup, Tooling Configuration, Testing, and CI

**Date**: 2026-10-15  
**Status**: Accepted  

## Context

The **Agentic RAG Research Copilot** is a modular Python system integrating document ingestion (PDFs, HTML), vector and hybrid stores (Qdrant), cross-encoder rerankers, an agentic self-correction loop, and serving layers (FastAPI, UI). 

To ensure long-term code quality, reproducible environments across developer machines and CI, strict type safety, and protection against regressions, the repository required an established foundational development setup before writing application domain code.

Historically, Python projects have suffered from:
1. **Slow and fragile dependency management**: Traditional `pip` and unpinned `requirements.txt` lead to non-reproducible environments, missing lockfiles, and slow resolution.
2. **Configuration file sprawl**: Separate config files (`setup.cfg`, `setup.py`, `.flake8`, `tox.ini`, `black.toml`, `isort.cfg`) fragment project maintenance.
3. **Slow feedback loops**: Running Flake8, Black, isort, and Bandit separately in CI creates multi-minute bottlenecks.
4. **Lack of CI parity**: Discrepancies between local developer environments and remote CI runners cause "works on my machine" bugs.

## Decision

We have decided on the following foundational architecture for environment management, code quality enforcement, testing, and continuous integration:

### 1. Unified Dependency & Project Management with `uv`
- **Tool**: Standardized on Astral's **`uv`** for managing Python interpreters (pinning CPython 3.12 via `.python-version`), lightning-fast dependency resolution, and deterministic lockfile generation (`uv.lock`).
- **Configuration Hub**: Adopted `pyproject.toml` (PEP 517/518/621) as the single source of truth for project metadata, build settings, and tool configurations.
- **Dependency Separation**: Split runtime dependencies (`[project.dependencies]`) from development tools (`[project.optional-dependencies].dev`), ensuring end-users are not forced to install test runners or linters.
- **Packaging Backend**: Configured `hatchling` as the PEP 517 build backend and structured source code in a `src/` layout (`src/agentic_rag_copilot`) to prevent accidental imports of uninstalled local packages during testing.

### 2. Consolidated Linting and Formatting with `Ruff`
- **Tool**: Replaced Flake8, Black, isort, pyupgrade, and Bandit with **`ruff`**.
- **Settings**:
  - `target-version = "py312"` and `line-length = 100`.
  - Comprehensive rule selection (`select`): `E`, `W` (pycodestyle), `F` (pyflakes), `I` (isort import order), `B` (flake8-bugbear), `C4` (comprehensions), `UP` (modern syntax upgrades), `SIM` (code simplification), `TCH` (type-checking guard imports), `ANN` (type annotations), `PTH` (pathlib usage), and `RUF` (Ruff-specific checks).
  - Pragmatic exemptions (`ignore`): `ANN401` disabled to permit `typing.Any` where dynamic interfaces or third-party boundaries require it.
  - Per-file relaxation (`per-file-ignores`): Relaxed annotation requirements (`ANN`) and permitted Pytest `assert` statements (`S101`) in `tests/**/*.py`.
  - Formatter (`[tool.ruff.format]`): Standardized on double quotes (`"`) and 4-space indentation.

### 3. Static Type Checking with `Pyright`
- Configured `pyright` in strict mode (`typeCheckingMode = "strict"`) targeting Python 3.12 to enforce type safety across domain models and interfaces.

### 4. Testing Framework & Coverage Gate with `Pytest`
- **Runner**: Configured `pytest` with `pytest-asyncio` (`asyncio_mode = "auto"`) to run asynchronous and synchronous tests seamlessly.
- **Directory Layout**: Structured test suite into:
  - `tests/conftest.py`: Shared global fixtures (e.g., `anyio_backend` set to `"asyncio"`).
  - `tests/unit/`: Fast, isolated tests mocking vector stores and LLM calls.
  - `tests/integration/`: Slower integration tests against external services (e.g., Qdrant).
- **Initial Scaffold**: Added `tests/unit/test_placeholder.py` to verify test discovery and prevent Pytest exit code 5 (`NO_TESTS_COLLECTED`) during initial repository scaffolding.
- **Coverage Enforcement**: Configured `pytest-cov` with `--cov=src/agentic_rag_copilot --cov-fail-under=80`, enforcing an 80% coverage threshold in CI while excluding boilerplate (`if TYPE_CHECKING:`, `raise NotImplementedError`, `*/__init__.py`).

### 5. Automated Continuous Integration via GitHub Actions (GHA)
- Configured `.github/workflows/ci.yml` running on pushes to `main`/`develop` and PRs against `main`.
- Parallelized execution into two dedicated jobs:
  1. **`lint`**: Runs `ruff check`, `ruff format --check`, and `pyright`.
  2. **`test`**: Runs `pytest` with coverage report generation.
- Leveraged `astral-sh/setup-uv@v4` with automatic dependency caching for rapid execution.

### 6. Local Task Automation via `justfile`
- Added standard recipe aliases (`install`, `lint`, `fmt`, `typecheck`, `test`, `test-fast`, `check`) via `just` to ensure local execution mirrors CI commands exactly.

## Consequences

### Positive Benefits
- **Sub-Second Feedback**: `uv` and `ruff` execute dependency syncs and linting passes in milliseconds rather than minutes.
- **Deterministic Builds**: `uv.lock` eliminates dependency drift and ensures identical environments between local developer machines and GitHub Actions runners.
- **Unified Configuration**: All metadata and tool settings reside in `pyproject.toml`, removing configuration clutter from the repository root.
- **Strict Quality Enforcement**: Parallel CI workflows catch typing errors, formatting issues, lint violations, and coverage regressions before code merges into protected branches.
- **Developer Ergonomics**: Common developer actions are simplified into single-word `just` recipes (`just check`, `just test`).

### Trade-offs & Mitigations
- **Strict Coverage Threshold**: An 80% coverage gate will fail on newly created modules that do not yet have corresponding test coverage.  
  *Mitigation*: Developers use `just test-fast` (`pytest --no-cov`) during early prototyping and implement unit tests before submitting PRs.
- **Tooling Prerequisites**: Developers must install `uv` (and optionally `just`) locally instead of relying on legacy `pip`.  

