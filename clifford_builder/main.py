import time

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import CZGate, CXGate

import circuit

from matplotlib import pyplot as plt
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from clifford_builder.clifford import sample_clifford_group
from clifford_builder.gate_utils import generate_two_qubit_gates, find_matching_combinations

if __name__ == '__main__':
    # print("Example of random clifford circuit generation")
    coupling = CouplingMap([[0, 1], [2, 3], [1, 2], [2, 4], [0, 5]])
    start = time.process_time_ns()
    qcirc = sample_clifford_group(6, True, coupling_map=coupling, max_distance=1)
    end = time.process_time_ns()
    plt.figure()
    qcirc.draw("mpl")
    draw = time.process_time_ns()

    print(f"Elapsed time: {(end - start) / 1000000} ms")
    print(f"Draw time: {(draw - end) / 1000000} ms")

    print()
    print("Example of CNOT gate application")
    qc = QuantumCircuit(7)
    coupling = CouplingMap([[0, 1], [2, 3], [1, 2], [2, 4], [0, 5], [5, 6]])
    circuit.cnot(qc, 1, 6, coupling_map=coupling)
    qc.draw("mpl")
    plt.show()

    gates = generate_two_qubit_gates()
    cz = CZGate().to_matrix()
    cx = CXGate().to_matrix()
    target = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex)
    solution = find_matching_combinations(gates, cx, '/tmp/test.out', 3,
                                          allow_global_phase=True)
    print(solution)
