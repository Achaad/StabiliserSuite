# StabiliserSuite

Toolbox for benchmarking and stress‑testing NISQ quantum processors  
*Clifford circuits • topology‑aware routing • gate‑transformation search*

[![CI](https://img.shields.io/github/actions/workflow/status/Achaad/StabiliserSuite/build.yml?branch=master&logo=github)](https://github.com/Achaad/StabiliserSuite/actions)
[![PyPI](https://img.shields.io/pypi/v/stabilisersuite?color=blue)](https://pypi.org/project/stabilisersuite)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/stabilisersuite?color=green)](https://anaconda.org/conda-forge/stabilisersuite)
[![MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Key Features

* **Random Clifford generator** – `O(n²)` tableau routine for uniformly random stabiliser circuits  
* **Topology‑aware routing** – 4‑CNOT “bridge’’ pattern for non‑adjacent CNOTs on heavy‑hex and ion‑trap layouts  
* **Gate‑transformation search** – meet‑in‑the‑middle engine (canonical‑phase hashing, memoisation) finds depth‑bounded Clifford / non‑Clifford substitutions  
* **Unified workflow** – circuit synthesis → classical stabiliser simulation → Qiskit hardware run → fidelity report  
* **Open‑source & extensible** – MIT licence, clean Python API, minimal external deps 

---

## 🚀 Quick Start

### Conda (recommended)

```bash
mamba create -n stabilisersuite -c conda-forge stabiliser_suite
mamba activate stabilisersuite
```

### Pure‑pip (light‑weight alternative)

```bash
python -m venv .venv
source .venv/bin/activate
pip install stabiliser_suite
```

### Example usage

```python
import stabiliser_suite as ss
from qiskit.circuit.library import CXGate

# 1. random 6‑qubit Clifford circuit
circ = ss.clifford.sample_clifford_group(3)
print(circ.draw())

# 2. non‑Clifford substitute for CX at depth ≤3
subs = ss.gate_utils.find_non_clifford_transformations(CXGate(), max_depth=3)
print("First match:", subs[0][0])
```

---

## 🗂 Project Layout

```
stabiliser_suite/
├── benchmark     # simple benchmarking scripts
├── clifford      # Clifford-circuit generation
├── circuit       # 4-CNOT construction
├── gate_utils    # gate transformation search
└── tests/        # pytest suite
```

---

## 📖 Documentation

Full API reference and tutorials will be later put in `docs/`.  
Rendered docs will be hosted at <https://achaad.github.io/StabiliserSuite/>.

---

## 🤝 Contributing

1. Fork → feature branch → pull request against `main`.  
2. Run `pytest` and `mypy` locally (`mamba env update -f environment.yml -n stabilisersuite --prune`).  
3. Code is auto‑formatted with **Black**.

The project uses [semantic versioning 2.0](https://semver.org/), i.e. v\${MAJOR}.\${MINOR}.\${PATCH} (e.g. v0.1.123)

The project is configured with automatic releases on any commits to the master branch.
All the merge commits should be prepended with a tag from the list below _(e.g. "feat: Add ESR support"_):

| Tag      | Description                                                |
|----------|------------------------------------------------------------|
| feat     | New feature                                                |
| fix      | Bug fix                                                    |
| docs     | Documentation only changes                                 |
| style    | Only code style changes                                    |
| refactor | Code refactoring                                           |
| perf     | Performance improvements                                   |
| test     | Adding new or correcting existing tests                    |
| build    | Changes which modify the application build or dependencies |
| ci       | Changes to the CI/CD configuration                         |
| revert   | Previous commit reverts                                    |


---

## 📄 License

StabiliserSuite is released under the MIT License (see `LICENSE`).

---

## 📑 Citation

```bibtex
@misc{stabilisersuite2025,
  author       = {Anton Perepelenko},
  title        = {StabiliserSuite: Toolbox for Quantum Computer Testing},
  year         = {2025},
  howpublished = {\url{https://github.com/Achaad/StabiliserSuite}},
  note         = {MIT License}
}
```

---