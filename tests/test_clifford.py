import numpy as np
import qiskit
from qiskit import QuantumCircuit
from matplotlib import pyplot as plt
import termtables
from sympy.testing.pytest import raises

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


def test___apply_cnot():
    qc = QuantumCircuit(2)
    tableau = np.array([[0, 0, 1, 0, 1], [1, 1, 0, 1, 0]])
    clifford.__apply_cnot(tableau, qc, 0, 1)
    expected_tableau = np.array([[0, 0, 1, 0, 1], [1, 0, 1, 1, 0]])
    assert len(qc.data) == 1
    assert qc.data[0].name == 'cx'
    assert np.array_equal(tableau, expected_tableau)
    assert qc.data[0].qubits[0]._index == 0
    assert qc.data[0].qubits[1]._index == 1

    # Perform the same with shift
    qc = QuantumCircuit(3)
    tableau = np.array([[0, 0, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 1]])
    clifford.__apply_cnot(tableau, qc, 0, 1, shift=1)
    expected_tableau = np.array([[0, 0, 1, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0, 1]])
    assert len(qc.data) == 1
    assert qc.data[0].name == 'cx'
    assert np.array_equal(tableau, expected_tableau)
    assert qc.data[0].qubits[0]._index == 1
    assert qc.data[0].qubits[1]._index == 2


def test___apply_swap():
    qc = QuantumCircuit(2)
    tableau = np.array([[0, 0, 1, 0, 1], [1, 1, 0, 1, 0]])
    clifford.__apply_swap(tableau, qc, 0, 1)
    expected_tableau = np.array([[0, 0, 0, 1, 1], [1, 1, 1, 0, 0]])
    assert len(qc.data) == 1
    assert qc.data[0].name == 'swap'
    assert np.array_equal(tableau, expected_tableau)
    assert qc.data[0].qubits[0]._index == 0
    assert qc.data[0].qubits[1]._index == 1

    # Perform the same with shift
    qc = QuantumCircuit(3)
    tableau = np.array([[1, 0, 0, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 1]])
    clifford.__apply_swap(tableau, qc, 0, 1, shift=1)
    expected_tableau = np.array([[0, 1, 0, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 1]])
    assert len(qc.data) == 1
    assert qc.data[0].name == 'swap'
    assert np.array_equal(tableau, expected_tableau)
    assert qc.data[0].qubits[0]._index == 1
    assert qc.data[0].qubits[1]._index == 2


def test___get_pauli():
    tableau = np.array([[1, 0, 0, 1, 0], [1, 0, 1, 0, 0, ]])
    assert clifford.__get_pauli(tableau, 0, 0) == 'x'
    assert clifford.__get_pauli(tableau, 0, 1) == 'z'
    assert clifford.__get_pauli(tableau, 1, 0) == 'y'
    assert clifford.__get_pauli(tableau, 1, 1) == 'I'


def test___apply_pauli():
    qc = QuantumCircuit(2)
    tableau = np.array([[0, 0, 1, 0, 1], [1, 1, 0, 1, 0]])
    clifford.__apply_pauli(tableau, qc, 0, 'x')
    expected_tableau = np.array([[0, 0, 1, 0, 0], [1, 1, 0, 1, 0]])
    assert len(qc.data) == 1
    assert qc.data[0].name == 'x'
    assert np.array_equal(tableau, expected_tableau)
    assert qc.data[0].qubits[0]._index == 0

    qc = QuantumCircuit(2)
    tableau = np.array([[0, 0, 1, 0, 1], [1, 1, 0, 1, 0]])
    clifford.__apply_pauli(tableau, qc, 0, 'y', shift=1)
    expected_tableau = np.array([[0, 0, 1, 0, 0], [1, 1, 0, 1, 1]])
    assert len(qc.data) == 1
    assert qc.data[0].name == 'y'
    assert np.array_equal(tableau, expected_tableau)
    assert qc.data[0].qubits[0]._index == 1

    qc = QuantumCircuit(2)
    tableau = np.array([[0, 0, 1, 0, 1], [1, 1, 0, 1, 0]])
    clifford.__apply_pauli(tableau, qc, 0, 'z', shift=1)
    expected_tableau = np.array([[0, 0, 1, 0, 1], [1, 1, 0, 1, 1]])
    assert len(qc.data) == 1
    assert qc.data[0].name == 'z'
    assert np.array_equal(tableau, expected_tableau)
    assert qc.data[0].qubits[0]._index == 1

    with raises(ValueError):
        clifford.__apply_pauli(tableau, qc, 0, 'a', shift=1)


def test_print_tableau(capsys):
    tableau = np.array([[0, 0, 1, 0, 1], [1, 1, 0, 1, 0]])
    clifford.print_tableau(tableau, representation="text")
    captured = capsys.readouterr()
    expected = ("┌────┬────┬────┬────┬───┐\n│ X0 │ X1 │ Z0 │ Z1 │ S │\n╞════╪════╪════╪════╪═══╡\n│ 0  │ 0  │ 1  │ 0  "
                "│ 1 │\n├────┼────┼────┼────┼───┤\n│ 1  │ 1  │ 0  │ 1  │ 0 │\n└────┴────┴────┴────┴───┘\n")
    assert captured.out == expected

    with raises(NotImplementedError):
        clifford.print_tableau(tableau, representation="mpl")

    with raises(ValueError):
        clifford.print_tableau(tableau, representation="aaa")
