"""
Deep-ML Problem 1: Matrix-Vector Dot Product
https://www.deep-ml.com/problems/1

Four framework-based implementations of A @ v, each showing how the same
O(m*n) operation is expressed and executed by a different piece of the
ML systems stack:

    1. matrix_dot_vector_numpy    -> NumPy (CPU, BLAS)
    2. matrix_dot_vector_pytorch  -> PyTorch (autograd-capable tensor, GPU-portable)
    3. matrix_dot_vector_cuda     -> Raw CUDA C kernel via PyCUDA (hand-written GPU code)
    4. matrix_dot_vector_tinygrad -> Tinygrad (minimal deep-learning framework)

All four share the same contract as the pure-Python versions in solution.py:
    - Return -1 if len(a[0]) != len(b)
    - Return [] for an empty matrix
"""

from __future__ import annotations

Number = int | float
Matrix = list[list[Number]]
Vector = list[Number]


# ---------------------------------------------------------------------------
# 1. NumPy
# ---------------------------------------------------------------------------
def matrix_dot_vector_numpy(a: Matrix, b: Vector) -> Vector | int:
    """CPU reference implementation, BLAS-backed via NumPy's `@` operator."""
    import numpy as np

    if len(a) == 0:
        return []
    if len(a[0]) != len(b):
        return -1

    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    return (arr_a @ arr_b).tolist()


# ---------------------------------------------------------------------------
# 2. PyTorch
# ---------------------------------------------------------------------------
def matrix_dot_vector_pytorch(a: Matrix, b: Vector) -> Vector | int:
    """Tensor-based implementation. Runs on GPU automatically if `torch.cuda`
    is available (e.g. a Colab T4), otherwise falls back to CPU — no code
    change needed, only the `.to(device)` call differs.
    """
    import torch

    if len(a) == 0:
        return []
    if len(a[0]) != len(b):
        return -1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor_a = torch.tensor(a, dtype=torch.float32, device=device)
    tensor_b = torch.tensor(b, dtype=torch.float32, device=device)
    result = tensor_a @ tensor_b
    return result.cpu().tolist()


# ---------------------------------------------------------------------------
# 3. Raw CUDA kernel (PyCUDA)
# ---------------------------------------------------------------------------
_CUDA_KERNEL_SOURCE = r"""
__global__ void matvec_kernel(const float *A, const float *v, float *out,
                               int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            sum += A[row * cols + col] * v[col];
        }
        out[row] = sum;
    }
}
"""


def matrix_dot_vector_cuda(a: Matrix, b: Vector) -> Vector | int:
    """Hand-written CUDA C kernel, compiled at runtime and launched via
    PyCUDA. One GPU thread is assigned per output row; each thread walks
    its row and accumulates the dot product independently — the same
    parallelization pattern used inside real GEMV/GEMM kernels.

    Requires an NVIDIA GPU + CUDA toolkit + `pycuda` installed (e.g. a
    Colab GPU runtime). Falls back to a NumPy CPU emulation of the exact
    same per-row-thread logic when no CUDA device is present, so the
    function is still runnable/testable on CPU-only machines like this one.
    """
    if len(a) == 0:
        return []
    if len(a[0]) != len(b):
        return -1

    try:
        import numpy as np
        import pycuda.autoinit  # noqa: F401  (initializes the CUDA context)
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule

        rows, cols = len(a), len(a[0])
        arr_a = np.array(a, dtype=np.float32).reshape(rows, cols)
        arr_b = np.array(b, dtype=np.float32)
        out = np.zeros(rows, dtype=np.float32)

        mod = SourceModule(_CUDA_KERNEL_SOURCE)
        matvec_kernel = mod.get_function("matvec_kernel")

        threads_per_block = 256
        blocks_per_grid = (rows + threads_per_block - 1) // threads_per_block

        matvec_kernel(
            cuda.In(arr_a), cuda.In(arr_b), cuda.Out(out),
            np.int32(rows), np.int32(cols),
            block=(threads_per_block, 1, 1),
            grid=(blocks_per_grid, 1),
        )
        return out.tolist()

    except Exception as exc:  # pragma: no cover - environment-dependent
        # No CUDA-capable GPU / toolkit / pycuda available in this
        # environment. Emulate the exact same per-row-thread logic on the
        # CPU so the function still returns a correct, comparable result.
        import numpy as np

        print(
            f"[matrix_dot_vector_cuda] CUDA unavailable ({exc.__class__.__name__}); "
            "falling back to CPU emulation of the kernel logic."
        )
        arr_a = np.array(a, dtype=np.float32)
        arr_b = np.array(b, dtype=np.float32)
        out = np.empty(len(a), dtype=np.float32)
        for row in range(len(a)):          # one "thread" per row
            total = 0.0
            for col in range(len(a[0])):   # the kernel's inner loop
                total += arr_a[row, col] * arr_b[col]
            out[row] = total
        return out.tolist()


# ---------------------------------------------------------------------------
# 4. Tinygrad
# ---------------------------------------------------------------------------
def matrix_dot_vector_tinygrad(a: Matrix, b: Vector) -> Vector | int:
    """Implementation using tinygrad's `Tensor`. Tinygrad lazily builds a
    compute graph and lowers it to whichever backend is configured
    (CPU/CLANG, CUDA, Metal, etc. via the `TINYGRAD` device env var),
    making this the same three lines regardless of target hardware.
    """
    from tinygrad import Tensor

    if len(a) == 0:
        return []
    if len(a[0]) != len(b):
        return -1

    tensor_a = Tensor(a, dtype="float32")
    tensor_b = Tensor(b, dtype="float32")
    result = tensor_a.matmul(tensor_b)
    return result.tolist()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    implementations = {
        "numpy": matrix_dot_vector_numpy,
        "pytorch": matrix_dot_vector_pytorch,
        "cuda": matrix_dot_vector_cuda,
        "tinygrad": matrix_dot_vector_tinygrad,
    }

    test_cases = [
        # (a, b, expected)
        ([[1, 2], [2, 4]], [1, 2], [5, 10]),
        ([[1, 2, 3], [4, 5, 6]], [1, 2, 3], [14, 32]),
        ([[1, 2], [2, 4], [3, 6]], [1, 2, 3], -1),  # dimension mismatch
        ([], [1, 2], []),                            # empty matrix
    ]

    for a, b, expected in test_cases:
        print(f"a={a}, b={b}")
        for name, fn in implementations.items():
            got = fn(a, b)
            # Allow small float tolerance when comparing against int expected values
            if isinstance(got, list) and isinstance(expected, list):
                ok = len(got) == len(expected) and all(
                    abs(float(g) - float(e)) < 1e-4 for g, e in zip(got, expected)
                )
            else:
                ok = got == expected
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] {name:10s} -> {got} (expected {expected})")
        print()
