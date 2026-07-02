# bloqade-python-mis

A [Bloqade](https://bloqade.quera.com) plugin implementing quantum-classical hybrid optimization for **Maximal Independent Set (MIS)** and **k-MaxCut** problems on neutral atom quantum computers.

This repo is the code associated with the paper "[Solving non-native combinatorial optimization problems using hybrid quantum-classical algorithms](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10636813)" (Wurtz, Sack, Wang — IEEE Transactions on Quantum Engineering, 2024). Also on [arXiv:2403.03153](https://arxiv.org/abs/2403.03153).

> **Warning:** This is unmanaged research code intended to reproduce results of the paper.

## Background

Classical combinatorial optimization problems like MIS are not naturally expressed as Rydberg Hamiltonian ground states. This codebase implements the **Non-Native Hybrid Algorithm (NNHA)** framework from the paper, which avoids that limitation: instead of requiring the quantum device to directly produce solutions, it uses quantum measurement outcomes as a resource for classical postprocessing routines.

The framework defines three algorithm types, each using the quantum device differently:

| Type | Quantum role | Classical role | Problem |
|------|-------------|----------------|---------|
| **Type 1** | Individual bitstrings as warm starts | 1-local greedy add/remove to enforce independence | Unit disk MIS |
| **Type 2** | Connected correlation functions as features | Spectral clustering (k-means on eigenvectors) | Max k-Cut |
| **Type 3** | Full measurement distribution as a reservoir | Cluster simulated annealing with sandpile updates | Unit disk MIS |

The quantum ansatz is a piecewise-linear adiabatic state preparation on neutral atom arrays, run on QuEra's Aquila (256-qubit). Pulse parameters (time, initial/final detuning) are optimized variationally.

## Installation

Requires Python ≥ 3.10.

```bash
uv sync
```

## Architecture

### Problem

Defines the combinatorial optimization target and its cost function.

| Class | Module | Description |
|-------|--------|-------------|
| `unit_disk_maximum_independent_set` | `bloqade.postprocess.problem.MIS_problem` | MIS on a unit disk graph defined by atom positions and a blockade radius |
| `maximum_independent_set` | `bloqade.postprocess.problem.MIS_problem` | MIS on an arbitrary graph |
| `k_maxcut` | `bloqade.postprocess.problem.k_maxcut` | Graph k-partitioning (MaxCut generalization) |

### Solution

Wraps a parameterized Rydberg ansatz and a classical postprocessing strategy.

| Class | Type | Description |
|-------|------|-------------|
| `greedy_MIS_solution` | Type 1 | Quantum bitstrings warm-start a greedy MIS algorithm (remove conflicts, then greedily add) |
| `classical_greedy_MIS_solution` | Type 1 baseline | Same postprocessing from all-zeros (no quantum) |
| `spectral_kmaxcut_solution` | Type 2 | Eigenvectors of the quantum connected-correlation matrix feed k-means clustering |
| `tempering_MIS_solution` | Type 3 | Quantum distribution seeds cluster simulated annealing via sandpile-model updates |
| `classical_tempering_MIS_solution` | Type 3 baseline | Same annealing from classical sampling (greedy or all-zeros) |

All solutions share a common interface:
- `ansatz(quantum_parameters)` — builds the parameterized Bloqade program
- `get_solution(quantum_parameters, classical_parameters)` — returns candidate solutions
- `objective(...)` — scalar cost for the optimizer
- `submit(quantum_parameters)` — dispatches to the configured backend

### Optimizer

`bloqade.postprocess.optimizer.base_optimizer.Optimizer` — a VQE-style loop supporting:

- `"SPSA"` — simultaneous perturbation stochastic approximation
- `"Bayesian"` — Gaussian process optimization via scikit-optimize
- `"Random"` — uniform random search over bounds
- Any `scipy.optimize.minimize` method string

Supports parameter bounds, fixed indices, callback tracking, and JSON serialization for resuming runs.

### Backends

Pass a `backend` dict to any Solution constructor:

```python
{"quantum": "python_emulate", "num_shots": 20}   # local Python ODE solver
{"quantum": "braket_emulate", "num_shots": 100}   # Braket local emulator
{"quantum": "braket_aquila",  "num_shots": 100}   # AWS Braket → Aquila hardware
{"quantum": "internal_aquila","num_shots": 100}   # QuEra direct hardware access
```

## Usage

### Type 1: Greedy MIS (local emulator)

```python
from bloqade.postprocess.problem.MIS_problem import unit_disk_maximum_independent_set
from bloqade.postprocess.solution.greedy_MIS import greedy_MIS_solution, classical_greedy_MIS_solution
from bloqade.postprocess.optimizer.base_optimizer import Optimizer
import bloqade.postprocess as postprocess
import numpy as np

# Define a unit disk MIS problem from atom positions
positions = np.array(...)   # shape (N, 2), coordinates in units of blockade radius
problem = unit_disk_maximum_independent_set(positions, threshold=0.72)

# Build a quantum solution using the local Python emulator
solution = greedy_MIS_solution(problem, backend={"quantum": "python_emulate", "num_shots": 20})

# Optimize pulse parameters with SPSA
x_init = solution.flatten_parameters(
    {"time": 1.5, "initial_detuning": -15, "final_detuning": 15}, {}
)
optimizer = Optimizer(solution, method="SPSA", initial_guess=x_init, max_iter=50)
optimizer()
```

See `postprocessing_implementations/` for complete runnable examples:

| Script | Paper type | Problem | Backend | Strategy |
|--------|-----------|---------|---------|----------|
| `type1_MIS.py` | Type 1 | Unit disk MIS | `python_emulate` | Greedy warm-start postprocessing |
| `type2_kmaxcut.py` | Type 2 | Max 3-Cut | `braket_aquila` | Spectral clustering of correlations |
| `type3_MIS.py` | Type 3 | Unit disk MIS | `braket_aquila` | Sandpile cluster simulated annealing |

## Development

```bash
uv sync                          # install dependencies
uv run pytest tests/             # run tests
uv run black src/ tests/         # format
uv run ruff check src/ tests/    # lint
```

## Namespace package

This package extends the `bloqade` namespace. It requires `bloqade >= 0.34.0` (which provides `bloqade.analog`) and contributes `bloqade.postprocess` and `bloqade.utils` subpackages.
