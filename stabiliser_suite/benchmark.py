"""
Benchmark for find_non_clifford_substitutions
=============================================
Save as benchmark_transformation.py and run with
$ python benchmark_transformation.py
Requires: matplotlib, pandas, qiskit, and your toolbox in PYTHONPATH.
"""
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit.circuit.library import HGate, CXGate

from stabiliser_suite.gate_utils import find_non_clifford_transformations

targets = [("CX (two qubit)", CXGate())]
depths  = list(range(1, 11))          # test max_depth = 1 … 9
rows    = []

for name, gate in targets:
    for d in depths:
        t0 = time.perf_counter()
        subs = find_non_clifford_transformations(gate, max_depth=d)
        elapsed = time.perf_counter() - t0
        rows.append({
            "gate": name,
            "max_depth": d,
            "runtime_s": elapsed,
            "solutions": len(subs),
        })
        print(f"{name:18}  depth={d}  {elapsed:7.3f}s  solutions={len(subs)}")

df = pd.DataFrame(rows)
print("\n=== Raw results ===")
print(df)

# ------------------------------------------------------------
# Plot runtime
for name in df["gate"].unique():
    sub = df[df["gate"] == name]
    plt.figure()
    plt.plot(sub["max_depth"], sub["runtime_s"], marker="o")
    plt.title(f"Runtime vs depth — {name}")
    plt.xlabel("max_depth")
    plt.ylabel("seconds")
    plt.grid(True)
    plt.tight_layout()

# Plot solutions
for name in df["gate"].unique():
    sub = df[df["gate"] == name]
    plt.figure()
    plt.plot(sub["max_depth"], sub["solutions"], marker="s")
    plt.title(f"Solutions vs depth — {name}")
    plt.xlabel("max_depth")
    plt.ylabel("# solutions")
    plt.grid(True)
    plt.tight_layout()

plt.show()