# Explanation: Framework-Based Solutions

`solution_frameworks.py` implements the same operation as `solution.py`
(`A · v`, returning `-1` on a dimension mismatch), but through four different
pieces of the ML systems stack instead of plain Python. The point isn't that
these are "better" at multiplying a 2×2 matrix by a vector — it's to show how
the identical O(m·n) operation is expressed and executed differently as you
move from a math library, to an autograd tensor library, to a hand-written
GPU kernel, to a from-scratch deep-learning framework.

## 1. NumPy — the CPU/BLAS baseline

```python
arr_a = np.array(a, dtype=np.float64)
arr_b = np.array(b, dtype=np.float64)
(arr_a @ arr_b).tolist()
```

`@` dispatches to BLAS's `dgemv` (matrix-vector product) routine — decades-old,
heavily optimized C/Fortran code. No autograd, no GPU, just fast CPU numerics.
This is the reference every other implementation is checked against.

## 2. PyTorch — the same op, but device- and gradient-aware

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor_a = torch.tensor(a, dtype=torch.float32, device=device)
tensor_b = torch.tensor(b, dtype=torch.float32, device=device)
(tensor_a @ tensor_b).cpu().tolist()
```

The only difference from NumPy is that `tensor_a` and `tensor_b` live in
PyTorch's tensor system: `@` dispatches to `torch.matmul`, which routes to
cuBLAS on a GPU or oneDNN/MKL on CPU depending on `device`. Because the
device check happens once, at tensor-creation time, the exact same three
lines run unmodified on a laptop CPU or a Colab T4 — the pattern used
throughout this portfolio's Colab-T4-feasible implementations. As a bonus,
these tensors are autograd-tracked, so this same expression is exactly the
first line of a `forward()` in an actual neural network layer (`y = W @ x`).

## 3. Raw CUDA — writing the GPU kernel by hand

```c
__global__ void matvec_kernel(const float *A, const float *v, float *out,
                               int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col)
            sum += A[row * cols + col] * v[col];
        out[row] = sum;
    }
}
```

This is what NumPy's `dgemv` and PyTorch's `matmul` are doing *underneath*,
made explicit. One GPU **thread** is launched per output row (`rows` threads
total, grouped into blocks of 256); each thread independently walks its row
of `A` and accumulates a dot product with `v` — the classic GEMV
parallelization pattern (parallelize over rows, serial reduction within a
row). The kernel source is compiled at runtime via `pycuda`'s `SourceModule`
(an NVRTC wrapper), then launched with an explicit grid/block configuration.

**Environment note:** running the actual kernel requires an NVIDIA GPU, the
CUDA toolkit, and `pycuda` installed — e.g. a Colab GPU runtime
(`!pip install pycuda`), not a CPU-only machine. `matrix_dot_vector_cuda`
detects this and falls back to a CPU emulation of the *exact same* per-row
loop structure, so the function is still correct and testable everywhere;
the fallback path prints which exception triggered it so it's obvious when
you're looking at emulated vs. real GPU execution.

## 4. Tinygrad — a from-scratch framework's take on the same op

```python
tensor_a = Tensor(a, dtype="float32")
tensor_b = Tensor(b, dtype="float32")
tensor_a.matmul(tensor_b).tolist()
```

Tinygrad builds a lazy computation graph instead of executing eagerly:
`.matmul()` records an operation node, and nothing actually runs until
`.tolist()` forces realization. At that point tinygrad's own compiler
lowers the graph to source code for whichever backend is active — C compiled
via `clang`/`gcc` on CPU, PTX/CUDA on an NVIDIA GPU, Metal on Apple Silicon —
and JIT-compiles and runs it. It's a useful counterpoint to PyTorch: instead
of calling into an existing, massive compiled library (cuBLAS/MKL), tinygrad
generates and compiles the kernel itself, in only a few thousand lines of
Python, which is the entire point of the project (a "tiny" but complete deep
learning framework you can read end to end in an afternoon).

**Environment note:** tinygrad's CPU backend shells out to a C compiler at
runtime (`clang` by default, `gcc` if `CC=gcc` is set) to build the kernel it
generates. If neither is installed, realization fails — this is a real
dependency to be aware of, not a bug in the wrapper code.

## Side-by-Side Summary

| Framework | What actually executes the math | GPU support | Autograd | Dependency footprint |
|---|---|---|---|---|
| NumPy | BLAS (`dgemv`) | No | No | Just `numpy` |
| PyTorch | cuBLAS (GPU) / MKL-oneDNN (CPU) | Yes, transparent | Yes | `torch` (large) |
| Raw CUDA | Hand-written kernel, compiled via NVRTC | Required (falls back to CPU emulation) | No | `pycuda` + CUDA toolkit + GPU |
| Tinygrad | Self-generated + JIT-compiled kernel | Yes, via backend selection | Yes | `tinygrad` + a C compiler (CPU backend) |

## Why This Matters Beyond a 2×2 Example

`A @ v` (and its big sibling, `A @ B`) is the single most repeated operation
in deep learning — every linear layer, every attention projection, every
embedding lookup bottoms out in matrix-vector or matrix-matrix products.
Understanding it at all four of these levels — "the math," "the
production library call," "the actual GPU kernel," and "how a framework
compiles down to that kernel" — is what separates being able to call
`nn.Linear()` from understanding what happens when you do.
