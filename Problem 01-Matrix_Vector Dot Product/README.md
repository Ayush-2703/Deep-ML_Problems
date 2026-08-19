# Matrix-Vector Dot Product

> **Deep-ML Problem 1** · Difficulty: `Easy` · Category: `Linear Algebra`
> Problem link: https://www.deep-ml.com/problems/1

## Problem Statement

Write a Python function that computes the dot product of a matrix and a vector, i.e. `A · v`. The function returns a new vector representing the result. If the number of columns in the matrix does not equal the number of elements in the vector, the two cannot be multiplied — the function should return `-1`.

## Function Signature

```python
def matrix_dot_vector(a: list[list[int | float]], b: list[int | float]) -> list[int | float] | int:
    ...
```

## Example

**Input**
```python
a = [[1, 2], [2, 4]]
b = [1, 2]
```

**Output**
```python
[5, 10]
```

**Reasoning**
- Row 1 of `a` is `[1, 2]`, dotted with `b = [1, 2]` → `1*1 + 2*2 = 5`
- Row 2 of `a` is `[2, 4]`, dotted with `b = [1, 2]` → `2*1 + 4*2 = 10`

## Constraints

- `a` is a 2D list representing an `m × n` matrix.
- `b` is a 1D list of length `n`.
- If `len(a[0]) != len(b)`, return `-1`.
- Elements may be `int` or `float`.
- Handle the empty matrix (`a = []`) as a valid edge case.

## Files in This Folder

| File | Description |
|---|---|
| `solution.py` | Four pure-Python-level implementations of `matrix_dot_vector`, plus a self-test block |
| `explanation.md` | Concept walkthrough, step-by-step trace, complexity analysis, and approach comparison |
| `solution_frameworks.py` | Four **framework-based** implementations — NumPy, PyTorch, raw CUDA, Tinygrad |
| `explanation_frameworks.md` | What each framework is actually doing under the hood, and how they compare |

## Solutions Included

### `solution.py` — pure Python
1. **Nested loops** — the most explicit version; no library calls, easiest to trace by hand.
2. **List comprehension** — identical logic, written in a more compact, Pythonic style.
3. **`zip()` + `sum()`** — idiomatic pairwise iteration that avoids manual indexing.
4. **NumPy** — vectorized implementation, the approach you'd actually use in production ML code.

### `solution_frameworks.py` — ML systems stack
1. **NumPy** — BLAS-backed CPU baseline.
2. **PyTorch** — autograd tensor, runs on GPU automatically if `torch.cuda.is_available()`.
3. **Raw CUDA** — hand-written `__global__` kernel (one thread per output row), compiled at runtime via `pycuda`; falls back to a CPU emulation of the same per-row logic when no GPU is present.
4. **Tinygrad** — lazy computation graph, JIT-compiled to the active backend (CPU via `clang`/`gcc`, or GPU).

## Complexity

All eight implementations are **O(m·n)** time / **O(m)** output space — the difference between them is *what executes the inner loop* (interpreted Python, BLAS, cuBLAS, a hand-written CUDA kernel, or a self-generated tinygrad kernel), not the algorithm itself.

`m` = number of rows in `a`, `n` = number of columns in `a` (= length of `b`).

## Requirements

```bash
pip install numpy torch tinygrad          # solution_frameworks.py
pip install pycuda                        # only needed to run the real CUDA kernel (GPU required)
```

`solution.py` needs only `numpy`. `solution_frameworks.py`'s tinygrad CPU backend shells out to a C compiler (`clang` or `gcc` — set `CC=gcc` if `clang` isn't installed).

## Run

```bash
python solution.py               # 4 pure-Python-level implementations
python solution_frameworks.py    # 4 framework implementations
```

Each script runs all four of its implementations against the example case, a second valid case, an invalid-dimension case, and an empty-matrix case, and reports pass/fail per implementation.

---
*Part of Ayush Kumar Singh's Deep-ML solutions portfolio.*
