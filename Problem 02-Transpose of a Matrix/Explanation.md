# Explanation: Transpose of a Matrix

## The Underlying Math / Concept

For a matrix `A` with shape `(m, n)`, the transpose `A^T` has shape
`(n, m)`, defined element-wise by:

```
A^T[j][i] = A[i][j]      for all valid i in [0, m), j in [0, n)
```

In words: the element that lives at row `i`, column `j` in the original
matrix moves to row `j`, column `i` in the transpose. Geometrically,
this is a reflection of the matrix across its main diagonal (the line
from the top-left corner running down and to the right) -- diagonal
elements (`i == j`, only meaningful when the matrix is square) never
move; everything else swaps position with its mirror image across that
diagonal.

Transpose is a purely structural operation: it rearranges *where*
values live, but never changes the values themselves, and it does not
require the matrix to be square. A `1 x n` row vector transposes to an
`n x 1` column vector and vice versa, which is why transpose shows up
constantly in linear algebra as the tool for converting between row-
and column-vector conventions.

## Step-by-Step Trace of the Example

Input:
```
A = [[1, 2],
     [3, 4]]
```
Shape: `(m, n) = (2, 2)`. Output shape will be `(n, m) = (2, 2)`.

We walk every `(i, j)` pair in the input and place `A[i][j]` at
`result[j][i]`:

| i | j | A[i][j] | destination result[j][i] |
|---|---|---|---|
| 0 | 0 | 1 | result[0][0] |
| 0 | 1 | 2 | result[1][0] |
| 1 | 0 | 3 | result[0][1] |
| 1 | 1 | 4 | result[1][1] |

Filling those in:

```
result[0] = [1, 3]
result[1] = [2, 4]
```

So:
```
result = [[1, 3],
          [2, 4]]
```

which matches the expected output. Notice `1` and `4` (the diagonal)
never moved; `2` and `3` (off-diagonal) swapped positions -- exactly
the "reflect across the diagonal" intuition above.

## Why Each Approach Works

### 1. Nested loops (`transpose_loops`)

This is the most literal translation of the definition
`result[j][i] = a[i][j]` into code. It pre-allocates a `(cols x rows)`
grid of zeros, then walks every `(i, j)` index pair with two `for`
loops and assigns directly. Because it mirrors the mathematical
definition one line at a time, it's the easiest version to verify by
hand and serves as the ground truth the other three are checked
against.

### 2. List comprehension (`transpose_comprehension`)

Same logic as the nested-loop version, just expressed declaratively:
for each output row index `j` (0 to `cols - 1`), build a row by pulling
element `j` out of every input row. `[row[j] for row in a]` is exactly
"the j-th column of `a`, read top to bottom" -- which, by definition,
is the j-th row of the transpose. Wrapping that in an outer
comprehension over `j` produces all rows of the result in one
expression, with no manual index bookkeeping or pre-allocation.

### 3. Idiomatic Python: `zip(*a)` (`transpose_idiomatic`)

`zip(*a)` is the standard Python idiom for transposing nested
sequences. `*a` unpacks the list of rows into separate positional
arguments (i.e. `zip(a[0], a[1], ..., a[m-1])`), and `zip` then groups
the *i-th element of every argument* into a single tuple for each `i`.
Since each argument is one row of `a`, the i-th elements across all
rows are exactly the i-th column of `a` -- so `zip(*a)` yields the rows
of the transpose directly, one tuple per column of the original. The
list comprehension around it just converts each tuple back into a
list to match the expected output format.

### 4. NumPy (`transpose_numpy`)

NumPy stores a matrix as one flat, contiguous buffer plus a `(shape,
strides)` header describing how to walk that buffer in each dimension.
`.T` builds a new header with `shape` and `strides` reversed, pointing
at the *same* underlying buffer -- no element is copied or moved at
this stage. Reading `arr.T[j][i]` therefore reads the same memory as
`arr[i][j]`, satisfying the transpose definition, but the work of
producing that view is O(1) regardless of matrix size. The nested
Python list only gets materialized (and the real O(m·n) copy happens)
when `.tolist()` is called to match this problem's expected return
type.

## Edge Cases Handled and Why

- **Empty matrix (`[]`)**: `a[0]` would raise `IndexError` on an empty
  list, so every implementation checks `if not a or not a[0]: return []`
  up front, before touching row/column indices.
- **Single row / single column**: These are just the `m == 1` or
  `n == 1` special cases of the general `(m, n) -> (n, m)` shape
  change; no special-casing is needed beyond the empty-matrix guard,
  since the loop bounds (`rows`, `cols`) are derived directly from the
  input's actual shape.
- **Single element (`1x1`)**: Transposes to itself -- correctly falls
  out of the general logic with `rows = cols = 1`.
- **Non-square matrices**: All four implementations derive `rows` and
  `cols` independently from the input rather than assuming they're
  equal, so rectangular matrices transpose correctly without any extra
  code path.

## Complexity Analysis

| Implementation | Time | Space | Notes |
|---|---|---|---|
| Nested loops | O(m·n) | O(m·n) | One assignment per element; explicit pre-allocated output. |
| Comprehension | O(m·n) | O(m·n) | Same element count, no explicit pre-allocation -- rows are built incrementally. |
| Idiomatic (`zip(*a)`) | O(m·n) | O(m·n) | `zip` iterates each element once; tuple-to-list conversion touches each element again but stays linear. |
| NumPy | O(1) for the view, O(m·n) once materialized to a list | O(1) for the view, O(m·n) once materialized | The transpose itself is metadata-only; cost appears only when copying out. |

For a problem this size (returning a plain nested list), all four
implementations end up doing O(m·n) work overall once you include
producing the final Python object -- the meaningful difference is
*where* that cost lives (explicit Python-level loop vs. a single
vectorized C-level copy inside NumPy) and how large the constant factor
is, not the asymptotic class.

## When to Use Which Approach

- **Nested loops**: Best when you need to explain or debug the
  operation step by step, or when the transpose logic needs to be
  embedded inside something more complex (e.g. combined with another
  per-element operation) where a one-liner would obscure what's
  happening.
- **List comprehension**: A good middle ground -- more concise than
  explicit loops, still readable, and doesn't require importing
  anything. A solid default for small-to-medium pure-Python code.
- **Idiomatic (`zip(*a)`)**: The most "Pythonic" and shortest option
  for pure-Python code operating on lists of lists. Preferred in
  production Python code that doesn't already depend on NumPy, since
  it needs no imports and reads naturally once you know the idiom.
- **NumPy**: The right choice whenever the data is already a NumPy
  array, the matrix is large, or the transpose is one step in a larger
  numerical pipeline (e.g. feeding into matrix multiplication, linear
  algebra routines, or further vectorized math). It also avoids the
  Python-level looping overhead entirely for the transpose step itself.
