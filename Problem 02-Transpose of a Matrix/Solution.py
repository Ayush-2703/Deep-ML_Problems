"""
Deep-ML Problem 2: Transpose of a Matrix
https://www.deep-ml.com/problems/2

Four pure-Python-level implementations of matrix transpose, each
demonstrating a different style of solving the same problem:

    1. transpose_loops          - explicit nested loops (ground truth)
    2. transpose_comprehension  - list comprehension
    3. transpose_idiomatic      - zip() unpacking (the "Pythonic" way)
    4. transpose_numpy          - vectorized, NumPy-backed

All four take the same input shape (a list of lists, i.e. a matrix
represented in row-major order) and return the same output shape.
"""

from __future__ import annotations

from typing import List

import numpy as np

Matrix = List[List[float]]


def transpose_loops(a: Matrix) -> Matrix:
    """Transpose a matrix using explicit nested loops.

    This is the ground-truth implementation: it builds the output
    matrix cell by cell, mirroring the mathematical definition
    ``result[j][i] = a[i][j]``.

    Args:
        a: An (m x n) matrix as a list of m rows, each of length n.

    Returns:
        The (n x m) transposed matrix.
    """
    if not a or not a[0]:
        return []

    rows = len(a)
    cols = len(a[0])

    # Pre-allocate an (cols x rows) result grid.
    result: Matrix = [[0.0] * rows for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            result[j][i] = a[i][j]

    return result


def transpose_comprehension(a: Matrix) -> Matrix:
    """Transpose a matrix using a nested list comprehension.

    Functionally identical to `transpose_loops`, but expressed as a
    single declarative expression: for every output column index j,
    pull element j from every input row i.

    Args:
        a: An (m x n) matrix as a list of m rows.

    Returns:
        The (n x m) transposed matrix.
    """
    if not a or not a[0]:
        return []

    cols = len(a[0])
    return [[row[j] for row in a] for j in range(cols)]


def transpose_idiomatic(a: Matrix) -> Matrix:
    """Transpose a matrix the idiomatic Python way, using zip(*a).

    `zip(*a)` unpacks each row of `a` as a separate positional argument
    to `zip`, which then groups elements column-wise. This is the
    standard "Pythonic" one-liner for transposing nested lists.

    Args:
        a: An (m x n) matrix as a list of m rows.

    Returns:
        The (n x m) transposed matrix.
    """
    if not a or not a[0]:
        return []

    return [list(col) for col in zip(*a)]


def transpose_numpy(a: Matrix) -> Matrix:
    """Transpose a matrix using NumPy's vectorized `.T` accessor.

    NumPy stores the matrix as a contiguous buffer and represents the
    transpose as a *view* with swapped strides -- no element-by-element
    Python-level looping happens here at all, which is why this scales
    far better than the pure-Python versions for large matrices.

    Args:
        a: An (m x n) matrix as a list of m rows.

    Returns:
        The (n x m) transposed matrix, as a nested Python list.
    """
    if not a or not a[0]:
        return []

    arr = np.array(a)
    return arr.T.tolist()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    implementations = {
        "nested loops": transpose_loops,
        "comprehension": transpose_comprehension,
        "idiomatic (zip)": transpose_idiomatic,
        "numpy": transpose_numpy,
    }

    test_cases = [
        # (name, input, expected_output)
        (
            "sample case (2x2)",
            [[1, 2], [3, 4]],
            [[1, 3], [2, 4]],
        ),
        (
            "non-square (2x3)",
            [[1, 2, 3], [4, 5, 6]],
            [[1, 4], [2, 5], [3, 6]],
        ),
        (
            "single row (1xN)",
            [[1, 2, 3, 4]],
            [[1], [2], [3], [4]],
        ),
        (
            "single column (Nx1)",
            [[1], [2], [3]],
            [[1, 2, 3]],
        ),
        (
            "single element (1x1)",
            [[42]],
            [[42]],
        ),
        (
            "empty matrix",
            [],
            [],
        ),
    ]

    print(f"{'Implementation':<18} | {'Pass/Fail per case':<40}")
    print("-" * 62)

    overall_pass = True
    for impl_name, impl_fn in implementations.items():
        results = []
        for case_name, input_matrix, expected in test_cases:
            try:
                actual = impl_fn(input_matrix)
                passed = actual == expected
            except Exception as exc:  # noqa: BLE001 - report any failure
                passed = False
                print(f"  [{impl_name}] raised on '{case_name}': {exc}")
            results.append("PASS" if passed else "FAIL")
            overall_pass = overall_pass and passed
        print(f"{impl_name:<18} | {', '.join(results)}")

    print("-" * 62)
    print("Test case order:", ", ".join(name for name, _, _ in test_cases))
    print()
    print("ALL IMPLEMENTATIONS PASSED" if overall_pass else "SOME IMPLEMENTATIONS FAILED")
