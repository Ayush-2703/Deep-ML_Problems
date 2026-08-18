<div align="center">

# Deep-ML Problems

**Structured, benchmarked Python solutions to [Deep-ML](https://www.deep-ml.com/problems) practice problems** — every problem explained from first principles and implemented across multiple frameworks, from raw Python up to hand-written CUDA kernels.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](./LICENSE)
[![Deep-ML](https://img.shields.io/badge/Practice_on-Deep--ML-6366F1?style=flat-square)](https://www.deep-ml.com/problems)

</div>

---

## About

[Deep-ML](https://www.deep-ml.com/problems) is a LeetCode-style platform for machine learning and deep learning fundamentals — linear algebra, probability, neural network internals, and classical ML algorithms implemented from scratch.

This repository is my solutions log, built to a consistent standard: **every problem gets a plain-language explanation of the underlying math, a working implementation, and a written walkthrough of *why* the solution works** — not just code that passes the test case. Where a problem is a good excuse to go deeper (e.g. a core linear algebra op), solutions are also implemented across the frameworks that actually run production ML systems: NumPy, PyTorch, raw CUDA, and Tinygrad.

## Repository Structure

Each problem lives in its own folder, named after the problem, containing exactly three files:

```
Deep-ML_Problems/
├── matrix-vector-dot-product/
│   ├── README.md            # Problem statement, signature, example, constraints
│   ├── solution.py           # Working solution(s) in Python
│   └── explanation.md        # Why the solution works — math, trace, complexity
│
├── <next-problem>/
│   ├── README.md
│   ├── solution.py
│   └── explanation.md
│
└── README.md                 # you are here
```

| File | Purpose |
|---|---|
| **`README.md`** | The problem itself — restated clearly, with the exact function signature, a worked example, and constraints, plus a link back to the original Deep-ML problem page. |
| **`solution.py`** | A correct, self-tested Python implementation. Simpler problems use one clear approach; foundational ones (e.g. core tensor/linear-algebra ops) include multiple implementations for comparison. |
| **`explanation.md`** | The reasoning: the underlying math, a step-by-step trace on the example input, complexity analysis, and — where relevant — a comparison of approaches and when to reach for each one. |

## Problems

| # | Problem | Category | Difficulty | Frameworks | Solution |
|---|---|---|---|---|---|
| 1 | [Matrix-Vector Dot Product](https://www.deep-ml.com/problems/1) | Linear Algebra | Easy | Python · NumPy · PyTorch · CUDA · Tinygrad | [`matrix-vector-dot-product/`](./matrix-vector-dot-product) |

> New problems are added as they're solved — this table is the single source of truth for progress. See [Roadmap](#roadmap) for what's next.

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Numerical computing | NumPy |
| Deep learning frameworks | PyTorch, [Tinygrad](https://github.com/tinygrad/tinygrad) |
| GPU programming | CUDA (via PyCUDA), for problems where a hand-written kernel is instructive |
| Testing | Lightweight self-test blocks per solution (`if __name__ == "__main__":`), asserting output against the problem's stated example(s) and edge cases |

## Getting Started

```bash
git clone https://github.com/Ayush-2703/Deep-ML_Problems.git
cd Deep-ML_Problems/<problem-folder>

pip install -r requirements.txt   # numpy at minimum; torch/tinygrad for framework variants
python solution.py                # runs the self-test block
```

Each `solution.py` is runnable standalone and prints pass/fail for every test case it checks itself against — no separate test runner needed.

## Why This Format

Most "LeetCode solutions" repos are a wall of code with no context — useful for nobody, including future-me. The three-file structure here is deliberate:

- **`README.md`** makes each folder self-contained — you don't need the Deep-ML site open to understand the problem.
- **`explanation.md`** is the part most solution repos skip, and the part that actually matters: *why* the approach works, not just that it does.
- **Multi-framework solutions**, where included, exist because the same operation (e.g. a matrix-vector product) looks completely different depending on whether you're calling BLAS, letting an autograd engine trace it, writing the GPU kernel yourself, or watching a minimal framework compile one for you — and seeing all four side by side is the fastest way to actually understand what's happening under `nn.Linear()`.

## Roadmap

- [ ] Work through Deep-ML's Linear Algebra track
- [ ] Work through the Deep Learning / Neural Networks track
- [ ] Work through the Probability & Statistics track
- [ ] Add a progress badge / completion percentage to this README

## Connect

**Ayush Kumar Singh** — B.Tech AI/ML, Amity University Lucknow

[![GitHub](https://img.shields.io/badge/GitHub-Ayush--2703-181717?style=flat-square&logo=github)](https://github.com/Ayush-2703)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-ayushsingh2703-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ayushsingh2703)

## License

Distributed under the [MIT License](./LICENSE).
