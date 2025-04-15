import time
import circuit

from matplotlib import pyplot as plt
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from clifford_builder.clifford import sample_clifford_group

if __name__ == '__main__':
    print("Example of random clifford circuit generation")
    coupling = CouplingMap([[0, 1], [2, 3], [1, 2], [2, 4], [0, 5]])
    start = time.process_time_ns()
    qcirc = sample_clifford_group(6, True, coupling_map=coupling, max_distance=1)
    end = time.process_time_ns()
    plt.figure()
    plt.subplot(211)
    qcirc.draw("mpl")
    draw = time.process_time_ns()

    print(f"Elapsed time: {(end - start) / 1000000} ms")
    print(f"Draw time: {(draw - end) / 1000000} ms")

    print()
    print("Example of CNOT gate application")
    qc = QuantumCircuit(7)
    coupling = CouplingMap([[0, 1], [2, 3], [1, 2], [2, 4], [0, 5], [5, 6]])
    circuit.cnot(qc, 1, 6, coupling_map=coupling)
    plt.subplot(212)
    qc.draw("mpl")
    plt.show()