"""
Deep-ML Problem 2: Transpose of a Matrix -- Framework-level implementations
https://www.deep-ml.com/problems/2

Four ML-framework-level implementations of the same matrix transpose,
each showing how a different execution backend handles the operation:

    1. transpose_numpy_framework  - NumPy (BLAS-adjacent, view-based)
    2. transpose_torch            - PyTorch (auto device selection)
    3. transpose_cuda             - hand-written CUDA kernel via PyCUDA,
                                     with an honest CPU-emulation fallback
                                     when no GPU/CUDA toolkit is present
    4. transpose_tinygrad         - tinygrad (its own micro-autograd engine)

Environment note (checked at runtime, not assumed):
    This sandbox has no NVIDIA GPU and no CUDA toolkit installed
    (`nvidia-smi` and `nvcc` are both absent, and `pip install pycuda`
    fails to build because `cuda.h` cannot be found). `transpose_cuda`
    below detects this and falls back to a CPU emulation of the *exact
    same indexing logic* a CUDA kernel would use, and prints that it is
    doing so. It does not pretend to run on a GPU.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

Matrix = List[List[float]]


# ---------------------------------------------------------------------------
# 1. NumPy (BLAS-adjacent baseline)
# ---------------------------------------------------------------------------
def transpose_numpy_framework(a: Matrix) -> Matrix:
    """Transpose using NumPy's ndarray.T.

    Under the hood: NumPy does not move any data. An ndarray is a
    (pointer, shape, strides) triple, and `.T` simply returns a new
    ndarray view over the *same buffer* with the shape and strides
    reversed. Reading element [j][i] of the view reads element [i][j]
    of the original buffer. This is O(1) to construct; the O(n*m) cost
    only shows up if/when the view is later copied into a
    C-contiguous array (e.g. via `.tolist()` or `.copy()`).

    Args:
        a: An (m x n) matrix as a list of m rows.

    Returns:
        The (n x m) transposed matrix as a nested Python list.
    """
    if not a or not a[0]:
        return []
    arr = np.asarray(a, dtype=np.float64)
    return arr.T.tolist()


# ---------------------------------------------------------------------------
# 2. PyTorch (auto device selection)
# ---------------------------------------------------------------------------
_TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transpose_torch(a: Matrix) -> Matrix:
    """Transpose using PyTorch, on GPU if available, else CPU.

    Under the hood: `torch.Tensor.T` (for 2D tensors) is, like NumPy,
    a metadata-only view: it swaps the tensor's stride/shape info
    without touching the underlying storage. The resulting tensor is
    part of PyTorch's autograd graph, so if `a` required gradients,
    gradients would flow back through this transpose during
    backprop -- something the pure-Python and NumPy versions cannot do.

    Args:
        a: An (m x n) matrix as a list of m rows.

    Returns:
        The (n x m) transposed matrix as a nested Python list.
    """
    if not a or not a[0]:
        return []
    tensor = torch.tensor(a, dtype=torch.float64, device=_TORCH_DEVICE)
    return tensor.T.cpu().tolist()


# ---------------------------------------------------------------------------
# 3. Raw CUDA kernel (PyCUDA), with CPU-emulation fallback
# ---------------------------------------------------------------------------
def _cuda_available() -> bool:
    """Check whether PyCUDA can actually initialize a CUDA context."""
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda  # noqa: F401

        return True
    except Exception:
        return False


_CUDA_KERNEL_SRC = r"""
__global__ void transpose_kernel(const double *in, double *out, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // index into input columns
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // index into input rows

    if (row < rows && col < cols) {
        // in is (rows x cols) row-major; out is (cols x rows) row-major.
        out[col * rows + row] = in[row * cols + col];
    }
}
"""


def _transpose_cuda_emulated(a: Matrix) -> Matrix:
    """CPU emulation of the CUDA kernel's exact indexing logic.

    This mirrors what `transpose_kernel` above does per-thread, just
    executed serially in Python instead of in parallel on a GPU grid.
    Flat buffers are used deliberately (instead of nested lists) so the
    index arithmetic is identical to what the kernel computes.
    """
    rows = len(a)
    cols = len(a[0])

    flat_in = [a[r][c] for r in range(rows) for c in range(cols)]
    flat_out = [0.0] * (rows * cols)

    for row in range(rows):
        for col in range(cols):
            # out[col * rows + row] = in[row * cols + col]
            flat_out[col * rows + row] = flat_in[row * cols + col]

    return [[flat_out[c * rows + r] for r in range(rows)] for c in range(cols)]


def transpose_cuda(a: Matrix) -> Matrix:
    """Transpose using a hand-written CUDA kernel, falling back to a
    CPU emulation of the identical kernel logic if no GPU/CUDA is
    available in the current environment.

    Under the hood (real GPU path): each CUDA thread is responsible for
    exactly one element. Thread (col, row) reads `in[row*cols+col]` from
    global memory and writes it to `out[col*rows+row]`. A 2D grid of
    thread blocks covers the whole (rows x cols) input; the launch
    config below uses 16x16 thread blocks, which is a common default
    that keeps blocks resident without wasting threads on typical
    matrix sizes.

    Args:
        a: An (m x n) matrix as a list of m rows.

    Returns:
        The (n x m) transposed matrix as a nested Python list.
    """
    if not a or not a[0]:
        return []

    rows = len(a)
    cols = len(a[0])

    if not _cuda_available():
        print(
            "[transpose_cuda] No CUDA-capable GPU / toolkit detected in this "
            "environment -- falling back to a CPU emulation of the exact same "
            "kernel indexing logic (no GPU execution is being simulated)."
        )
        return _transpose_cuda_emulated(a)

    # --- Real GPU path (only reached when PyCUDA + a GPU are present) ---
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule

    mod = SourceModule(_CUDA_KERNEL_SRC)
    kernel = mod.get_function("transpose_kernel")

    a_np = np.asarray(a, dtype=np.float64)
    out_np = np.empty((cols, rows), dtype=np.float64)

    block = (16, 16, 1)
    grid = ((cols + block[0] - 1) // block[0], (rows + block[1] - 1) // block[1])

    kernel(
        cuda.In(a_np),
        cuda.Out(out_np),
        np.int32(rows),
        np.int32(cols),
        block=block,
        grid=grid,
    )

    return out_np.tolist()


# ---------------------------------------------------------------------------
# 4. tinygrad
# ---------------------------------------------------------------------------
def transpose_tinygrad(a: Matrix) -> Matrix:
    """Transpose using tinygrad's Tensor.

    Under the hood: tinygrad builds a lazy computation graph of UOps.
    `.transpose()` records a permute/movement op rather than doing any
    element copying immediately. The graph is only realized (compiled
    to an actual kernel -- C on CPU, or PTX/similar on GPU backends)
    when `.numpy()` (or another realize-triggering call) is invoked.
    This "lazy until realize" model is tinygrad's core design idea:
    keep the op graph small and let a single scheduler/codegen pass
    fuse and compile it, rather than eagerly dispatching one kernel per
    op the way PyTorch's default eager mode does.

    Args:
        a: An (m x n) matrix as a list of m rows.

    Returns:
        The (n x m) transposed matrix as a nested Python list.
    """
    if not a or not a[0]:
        return []

    from tinygrad.tensor import Tensor

    t = Tensor(a, dtype="float64")
    return t.transpose().numpy().tolist()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    implementations = {
        "numpy": transpose_numpy_framework,
        "torch": transpose_torch,
        "cuda (or CPU fallback)": transpose_cuda,
        "tinygrad": transpose_tinygrad,
    }

    test_cases = [
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
            "empty matrix",
            [],
            [],
        ),
    ]

    print(f"PyTorch device in use: {_TORCH_DEVICE}")
    print()
    print(f"{'Implementation':<24} | {'Pass/Fail per case':<40}")
    print("-" * 68)

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
        print(f"{impl_name:<24} | {', '.join(results)}")

    print("-" * 68)
    print("Test case order:", ", ".join(name for name, _, _ in test_cases))
    print()
    print("ALL IMPLEMENTATIONS PASSED" if overall_pass else "SOME IMPLEMENTATIONS FAILED")
