import numpy as np
import qiskit
from qiskit import QuantumCircuit
from matplotlib import pyplot as plt
import termtables
from clifford_builder import clifford


def test___apply_hadamard():
    qc = QuantumCircuit(2)
    tableau = np.array([[0, 0, 1, 0, 1], [1, 1, 0, 1, 0]])
    clifford.__apply_hadamard(tableau, qc, 0)
    expected_tableau = np.array([[1, 0, 0, 0, 1], [0, 1, 1, 1, 0]])
    assert len(qc.data) == 1
    assert qc.data[0].name == 'h'
    assert np.array_equal(tableau, expected_tableau)
    assert qc.data[0].qubits[0]._index == 0

    # Perform the same with shift
    qc = QuantumCircuit(2)
    tableau = np.array([[0, 0, 1, 1, 1], [1, 1, 0, 0, 0]])
    clifford.__apply_hadamard(tableau, qc, 0, shift=1)
    expected_tableau = np.array([[1, 0, 0, 1, 1], [0, 1, 1, 0, 0]])
    assert len(qc.data) == 1
    assert qc.data[0].name == 'h'
    assert np.array_equal(tableau, expected_tableau)
    assert qc.data[0].qubits[0]._index == 1


def test___apply_s():
    qc = QuantumCircuit(2)
    tableau = np.array([[0, 0, 1, 0, 1], [1, 1, 0, 1, 0]])
    clifford.__apply_s(tableau, qc, 0)
    expected_tableau = np.array([[0, 0, 1, 0, 1], [1, 1, 1, 1, 0]])
    assert len(qc.data) == 1
    assert qc.data[0].name == 's'
    assert np.array_equal(tableau, expected_tableau)
    assert qc.data[0].qubits[0]._index == 0

    # Perform the same with shift
    qc = QuantumCircuit(2)
    tableau = np.array([[0, 0, 1, 1, 1], [1, 1, 1, 0, 0]])
    clifford.__apply_s(tableau, qc, 0, shift=1)
    expected_tableau = np.array([[0, 0, 1, 1, 1], [1, 1, 0, 0, 0]])
    assert len(qc.data) == 1
    assert qc.data[0].name == 's'
    assert np.array_equal(tableau, expected_tableau)
    assert qc.data[0].qubits[0]._index == 1
