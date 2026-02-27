# BiWES — Bipartite Weighted Edge Shuffling

A Python library for randomising bipartite weighted graphs while preserving their degree and weight sequences.

## Installation

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from biwes import BiWES, BiWES2, pre_compute, find_configurations

# Create a simple bi-adjacency matrix
A = np.array([
    [2, 3, 0],
    [0, 1, 4],
    [1, 0, 2],
], dtype=np.int64)

# Run 1000 randomisation steps (single call)
A_rand = BiWES(A, 1_000)

# --- or reuse pre-computed structures for many runs ---
A2, weight_dict, weight_dict2, adj_list, adj_list2, uw, uw2 = pre_compute(A)
A_rand2 = BiWES2(A2, weight_dict, weight_dict2, uw, uw2, adj_list, adj_list2, 1_000)

# Enumerate all configurations for a tiny graph
solutions = find_configurations([3, 4], [3, 4], [2, 2], [2, 2])
```

## API

| Symbol | Description |
|---|---|
| `pre_compute(A)` | Pre-compute all internal data structures for a bi-adjacency matrix |
| `BiWES(A, n_iter)` | Randomise `A` in a single call |
| `BiWES2(A2, …, n_iter)` | Randomise using already-computed structures (efficient for multiple runs) |
| `find_configurations(s_row, s_col, d_row, d_col)` | Enumerate all integer matrices with given row/column strengths and degrees |
