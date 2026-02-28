# BiWES — Bipartite Weighted Edge Swap

A Python library for randomising bipartite weighted graphs while preserving their degree and weight sequences.

## Installation

```bash
pip install biwes
```

## Quick start

```python
import numpy as np
from biwes import BiWES, BiWES2, pre_compute, find_configurations

# Create a simple bi-adjacency matrix.
# The bia-adjacency matrix must contain only integers
A = np.array([
    [2, 3, 0, 2, 0],
    [0, 1, 4, 0, 0],
    [1, 0, 2, 0, 2],
    [0, 0, 2, 1, 0],
], dtype=np.int64)

# Run 1000 randomisation steps (single call)
#The functions BiWES and BiWES2 use numba, so the first call will be slow because 
A_rand = BiWES(A, 1000)

# --- or reuse pre-computed structures for many runs ---
A2, weight_dict, weight_dict2, adj_list, adj_list2, uw, uw2 = pre_compute(A)
A_rand2 = BiWES2(A2, weight_dict, weight_dict2, uw, uw2, adj_list, adj_list2, 1000)

#The functions BiWES and BiWES2 use numba, so the first call will be slow because
```

## Enumerate all configurations with a given degree and strength sequences for a tiny graph
This function employs a brute-force algorithm to enumerate all possible bi-adjacency matrices while fixing the degree and strength sequences of a bipartite graph. The number of feasible configurations grows rapidly as the number of nodes and edges increases. For graphs with more than 15–16 nodes, the algorithm may not converge within a reasonable amount of time.
```python
strength_a = list(np.sum(A, axis=1))
strength_b = list(np.sum(A, axis=0))
degree_a = list(np.sum(np.where(A > 0, 1, 0), axis=1))
degree_b = list(np.sum(np.where(A > 0, 1, 0), axis=0))
solutions = find_configurations(strength_a, strength_b, degree_a, degree_b)
#solutions is a list of bi-adjacency matrices
```







