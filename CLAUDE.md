# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`bloqade-python-mis` is a Bloqade plugin implementing quantum-classical hybrid optimization for Maximal Independent Set (MIS) problems on neutral atom quantum computers. It is structured as a namespace package under `bloqade`.

## Commands

**Package manager:** uv

```bash
uv sync              # Install dependencies
uv run pytest tests/ # Run tests
uv run pytest tests/test_bloqade_mis.py::test_name  # Run a single test
uv run coverage run -m pytest tests/ && coverage xml && coverage report  # Coverage report
uv build             # Build distribution packages
```

**Code quality** (run via pre-commit or directly):
```bash
pre-commit run --all-files   # Run all hooks (black, ruff, yaml checks)
uv run black src/ tests/     # Format code
uv run ruff check src/ tests/ # Lint
```

## Architecture

The core abstraction is a three-layer **Problem → Solution → Optimizer** pipeline:

### Problem (`src/bloqade/postprocess/problem/`)
- `Problem` (ABC in `base_problem.py`): defines `cost(solution)` and serialization (`from_file`/`to_file`)
- `maximum_independent_set` / `unit_disk_maximum_independent_set` (`MIS_problem.py`): MIS on general graphs and unit disk graphs (atom positions + blockade radius)
- `k_maxcut` (`k_maxcut.py`): graph partitioning

### Solution (`src/bloqade/postprocess/solution/`)
- `Solution` (ABC in `base_solution.py`): defines `ansatz()` (parameterized Rydberg pulse sequence), `get_solution()`, `objective()`, and `submit()` (dispatches to BraKet Aquila, QuEra hardware, or local emulator)
- `greedy_MIS_solution` (`greedy_MIS.py`): hybrid quantum + classical greedy postprocessing
- `classical_greedy_MIS_solution`: classical-only baseline (warm-starts with zero bitstring)
- `tempering_MIS` (`tempering_MIS.py`): tempering-based strategy
- `spectral_kmaxcut` (`spectral_kmaxcut.py`): spectral clustering for k-maxcut

### Optimizer (`src/bloqade/postprocess/optimizer/`)
- `Optimizer` (`base_optimizer.py`): VQE-style parameter optimization supporting SPSA, Bayesian (scikit-optimize), scipy.optimize.minimize, and random search. Supports parameter bounds, fixed parameters, callback tracking, and job tape caching.

### Utilities (`src/bloqade/utils/`)
- `record.py`: data persistence via `local_record` (JSON on disk) or `null_record` (no-op); AWS S3 stub not implemented
- `maximal_IS.py`: greedy MIS algorithms (`greedy_mIS`, `greedy_weighted_mIS`, `greedy_remove`)
- `udg_graph_mapping.py`: unit disk graph construction from atom positions
- `visualize_graph.py` / `visualize_UDG.py`: graph visualization helpers
- `observables.py`: quantum measurement abstractions

### Namespace package
`src/bloqade/__init__.py` is a namespace package entry point. It imports core Bloqade components, making this plugin extend the main `bloqade` package.

### Reference implementations
`postprocessing_implementations/` contains standalone example scripts (type1/2/3 MIS, k-maxcut) and benchmark data in `DATA/`. These are for reference and experimentation, not part of the installed package.

## Key Design Patterns

- **ABC-based extensibility**: new problem types or solution strategies subclass `Problem`/`Solution`
- **Backend abstraction**: `Solution.submit()` dispatches jobs to multiple backends (BraKet, QuEra, emulator, Python simulator) based on a string argument
- **Parameter flattening**: `Solution` flattens/unflattens parameter dicts for compatibility with scipy/SPSA optimizers
- **Job tape caching**: `Optimizer` caches results for repeated parameter evaluations to avoid redundant quantum circuit submissions
