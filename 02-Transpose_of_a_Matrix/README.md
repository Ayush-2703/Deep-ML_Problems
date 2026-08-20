# Transpose of a Matrix

- **Difficulty:** Easy
- **Category:** Linear Algebra
- **Problem link:** [Deep-ML | Transpose of a Matrix](https://www.deep-ml.com/problems/2)

## Problem Statement

Given a matrix expressed as a list of rows (a list of lists of numbers),
produce its transpose: a new matrix in which every row of the original
becomes a column, and every column becomes a row. If the input has shape
`(m, n)` (m rows, n columns), the output has shape `(n, m)`.

## Function Signature

```python
def transpose_matrix(a: list[list[float]]) -> list[list[float]]:
    ...
```

Input: `a`, an `m x n` matrix represented as a list of `m` rows, each a
list of `n` numbers (all rows the same length).

Output: the `n x m` transposed matrix, in the same nested-list format.

## Worked Example

**Input:**
```python
a = [[1, 2], [3, 4]]
```

**Output:**
```python
[[1, 3], [2, 4]]
```

**Reasoning:**

The input is a 2x2 matrix:

```
1  2
3  4
```

Transposing flips it across its main diagonal -- row `i`, column `j` of
the input becomes row `j`, column `i` of the output:

```
1  3
2  4
```

Concretely: `a[0][1] = 2` (row 0, col 1) moves to `result[1][0] = 2`
(row 1, col 0), and `a[1][0] = 3` moves to `result[0][1] = 3`. Every
other element in this small example sits on the diagonal, so it stays
put.

## Constraints and Edge Cases

- The input matrix is assumed **rectangular**: every row has the same
  number of columns. Ragged input is not a case any implementation here
  guards against.
- **Non-square matrices** (`m != n`) are fully supported -- the output
  shape is simply `(n, m)`.
- **Single row** (`1 x n`) transposes to a single column (`n x 1`), and
  vice versa.
- **Single element** (`1 x 1`) transposes to itself.
- **Empty matrix** (`[]` or a matrix with zero-length rows) transposes
  to `[]`. All four implementations in `solution.py` and
  `solution_frameworks.py` special-case this explicitly, since indexing
  `a[0]` on an empty list would otherwise raise an `IndexError`.

## Files

| File | Description |
|---|---|
| `README.md` | This file -- problem overview, examples, and usage. |
| `solution.py` | Four pure-Python-level implementations (loops, comprehension, idiomatic, NumPy) with a self-test block. |
| `explanation.md` | Math background, step-by-step trace, and why each pure-Python approach works. |
| `solution_frameworks.py` | Four ML-framework-level implementations (NumPy, PyTorch, CUDA/PyCUDA, tinygrad) with a self-test block. |
| `explanation_frameworks.md` | What each framework does under the hood, a comparison table, and real-world relevance. |

## Complexity

| Implementation | Time | Space |
|---|---|---|
| Nested loops | O(m·n) | O(m·n) |
| Comprehension | O(m·n) | O(m·n) |
| Idiomatic (`zip(*a)`) | O(m·n) | O(m·n) |
| NumPy (`.T`) | O(1) to create the view; O(m·n) if/when materialized via `.tolist()` | O(1) for the view; O(m·n) once materialized |

All four implementations must eventually touch every one of the `m*n`
elements at least once to produce a Python-visible result, so O(m·n) is
the practical floor for anything that returns a nested list. NumPy's
advantage is that the transpose *itself* is a free, O(1) metadata
operation (a view with swapped strides) -- the O(m·n) cost only appears
when the view is copied out into a plain Python list.

## Run Instructions

From this directory:

```bash
# Pure-Python-level implementations + self-test
python3 solution.py

# Framework-level implementations + self-test
# (requires numpy, torch, tinygrad; pycuda is optional -- see explanation_frameworks.md)
python3 solution_frameworks.py
```

Both scripts print a pass/fail table for every implementation against
the sample case and several edge cases, followed by an overall
pass/fail summary.
