<div align="center">
    
![Deep-ML Problems](https://capsule-render.vercel.app/api?type=waving&color=0:F0FDF4,100:BBF7D0&height=250&section=header&text=Deep-ML%20Problems&fontSize=60&fontColor=14532D&fontAlignY=36&animation=fadeIn&desc=A%20curated,%20optimized%20collection%20of%20Deep-ML%20problem%20solution%20traking%20my%20AI%20mastry%20journy&descSize=20&descAlignY=60)

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

**Deep-ML** is a LeetCode-style platform for machine learning and deep learning fundamentals — linear algebra, probability, neural network internals, and classical ML algorithms implemented from scratch.

This repository is my solutions log, built to a consistent standard: **every problem gets a plain-language explanation of the underlying math, a working implementation, and a written walkthrough of *why* the solution works** — not just code that passes the test case. Where a problem is a good excuse to go deeper (e.g. a core linear algebra op), solutions are also implemented across the frameworks that actually run production ML systems: NumPy, PyTorch, raw CUDA, and Tinygrad.

## Repository Structure

Each problem lives in its own folder, named after the problem, containing exactly three files:

```
Deep-ML_Problems/
├── <problem>/
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
| 1 | [Matrix-Vector Dot Product](https://www.deep-ml.com/problems/1) | Linear Algebra | Easy | Python · NumPy · PyTorch · CUDA · Tinygrad | [`Problem 01-Matrix_Vector Dot Product`](./Problem%2001-Matrix_Vector%20Dot%20Product) |
| 2 | [Transpose of a Matrix](https://www.deep-ml.com/problems/2) | Linear Algebra | Easy | Python · NumPy · PyTorch · CUDA · Tinygrad | ['Problem 02-Transpose of a Matrix/'](./Problem%2002-Transpose%20of%20a%20Matrix) |

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

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.  
You're free to use, fork, and build on this for personal and commercial projects.

---

## 👤 Author

<div align="center">

### Ayush Kumar Singh

*Researcher in Adversarial ML, Geospatial AI, and LLM/NLP Systems*

[![GitHub](https://img.shields.io/badge/GitHub-Ayush%20Kumar%20Singh-181717?style=for-the-badge&logo=github)](https://github.com/Ayush-2703)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ayush%20Kumar%20Singh-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/ayushsingh2703)
[![Email](https://img.shields.io/badge/Email-Ayush%20Kumar%20Singh-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:ab49ayush@gmail.com)

</div>

---

<div align="center">

**If this repository helped you, please consider giving it a ⭐**  
*It takes 2 seconds and helps others discover it.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:F0FDF4,100:BBF7D0&height=100&section=footer" width="100%"/>

</div>
