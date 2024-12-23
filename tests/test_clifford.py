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


def test___check_normalisation():
    tableau = np.array([[0, 0, 1, 0, 1], [1, 1, 0, 1, 0]])
    assert clifford.__check_normalisation(tableau) == False
    tableau = np.array([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]])
    assert clifford.__check_normalisation(tableau) == True


def test___step_1():
    tableau = np.array([[1, 0, 0, 1, 1, 0, 0], [1, 1, 0, 1, 1, 0, 0]])
    qc = QuantumCircuit(3)
    clifford.__step_1(tableau, qc)
    assert len(qc.data) == 2
    assert qc.data[0].name == 's'
    assert qc.data[0].qubits[0]._index == 0
    assert qc.data[1].name == 'h'
    assert qc.data[1].qubits[0]._index == 1
    expected_tableau = np.array([[1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 1, 0, 0]])
    assert np.array_equal(tableau, expected_tableau)


def test___step_2():
    qc = QuantumCircuit(4)
    tableau = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0]])
    clifford.__step_2(tableau, qc)
    assert len(qc.data) == 3
    assert qc.data[0].name == 'cx'
    assert qc.data[0].qubits[0]._index == 0
    assert qc.data[0].qubits[1]._index == 1
    assert qc.data[1].name == 'cx'
    assert qc.data[1].qubits[0]._index == 2
    assert qc.data[1].qubits[1]._index == 3
    assert qc.data[2].name == 'cx'
    assert qc.data[2].qubits[0]._index == 0
    assert qc.data[2].qubits[1]._index == 2

    expected_tableau = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0]])
    assert np.array_equal(tableau, expected_tableau)


def test___step_3():
    qc = QuantumCircuit(3)
    tableau = np.array([[0, 1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0]])
    clifford.__step_3(tableau, qc)
    assert len(qc.data) == 1
    assert qc.data[0].name == 'swap'
    assert qc.data[0].qubits[0]._index == 0
    assert qc.data[0].qubits[1]._index == 1
    expected_tableau = tableau = np.array([[1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0]])
    assert np.array_equal(tableau, expected_tableau)

    with raises(RuntimeError):
        qc = QuantumCircuit(3)
        tableau = np.array([[1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0]])
        clifford.__step_3(tableau, qc)


def test___step_4():
    qc = QuantumCircuit(3)
    tableau = np.array([[1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0]])
    clifford.__step_4(tableau, qc)
    assert len(qc.data) == 1
    assert qc.data[0].name == 'h'
    assert qc.data[0].qubits[0]._index == 0
    expected_tableau = np.array([[0, 0, 0, 1, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0]])
    assert np.array_equal(tableau, expected_tableau)

    qc = QuantumCircuit(2)
    tableau = np.array([[1, 0, 0, 0, 0], [0, 1, 1, 0, 0]])
    clifford.__step_4(tableau, qc)
    assert len(qc.data) == 0
    expected_tableau = np.array([[1, 0, 0, 0, 0], [0, 1, 1, 0, 0]])
    assert np.array_equal(tableau, expected_tableau)


def test___step_5():
    qc = QuantumCircuit(1)
    tableau = np.array([[1, 0, 0], [0, 1, 1]])
    clifford.__step_5(tableau, qc)
    assert len(qc.data) == 1
    assert qc.data[0].name == 'x'
    assert qc.data[0].qubits[0]._index == 0
    expected_tableau = np.array([[1, 0, 0], [0, 1, 0]])
    assert np.array_equal(tableau, expected_tableau)

    qc = QuantumCircuit(1)
    tableau = np.array([[1, 0, 1], [0, 1, 1]])
    clifford.__step_5(tableau, qc)
    assert len(qc.data) == 1
    assert qc.data[0].name == 'y'
    assert qc.data[0].qubits[0]._index == 0
    expected_tableau = np.array([[1, 0, 0], [0, 1, 0]])
    assert np.array_equal(tableau, expected_tableau)

    qc = QuantumCircuit(1)
    tableau = np.array([[1, 0, 1], [0, 1, 0]])
    clifford.__step_5(tableau, qc)
    assert len(qc.data) == 1
    assert qc.data[0].name == 'z'
    assert qc.data[0].qubits[0]._index == 0
    expected_tableau = np.array([[1, 0, 0], [0, 1, 0]])
    assert np.array_equal(tableau, expected_tableau)


def test___generate_random_tableau():
    tab1 = clifford.__generate_random_tableau(8)
    tab2 = clifford.__generate_random_tableau(8)
    assert not np.array_equal(tab1, tab2)


def test___perform_sweeping():
    qc = QuantumCircuit(4)
    tableau = np.array([[1, 1, 1, 1, 0, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 1]])
    clifford.__perform_sweeping(tableau, qc)
    assert len(qc.data) == 9
    assert qc.data[0].name == 's'
    assert qc.data[0].qubits[0]._index == 1
    assert qc.data[1].name == 's'
    assert qc.data[1].qubits[0]._index == 2
    assert qc.data[2].name == 'cx'
    assert qc.data[2].qubits[0]._index == 0
    assert qc.data[2].qubits[1]._index == 1
    assert qc.data[3].name == 'cx'
    assert qc.data[3].qubits[0]._index == 2
    assert qc.data[3].qubits[1]._index == 3
    assert qc.data[4].name == 'cx'
    assert qc.data[4].qubits[0]._index == 0
    assert qc.data[4].qubits[1]._index == 2
    assert qc.data[5].name == 'h'
    assert qc.data[5].qubits[0]._index == 0
    assert qc.data[6].name == 's'
    assert qc.data[6].qubits[0]._index == 0
    assert qc.data[7].name == 'h'
    assert qc.data[7].qubits[0]._index == 0
    assert qc.data[8].name == 'x'
    assert qc.data[8].qubits[0]._index == 0

    expected_tableau = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]])
    assert np.array_equal(tableau, expected_tableau)

    qc = QuantumCircuit(3)
    tableau = np.array([[0, 0, 0, 0, 1, 0, 0], [1, 1, 0, 1, 1, 0, 0]])
    clifford.__perform_sweeping(tableau, qc)
    assert len(qc.data) == 7
    assert qc.data[0].name == 'h'
    assert qc.data[0].qubits[0]._index == 1
    assert qc.data[1].name == 'swap'
    assert qc.data[1].qubits[0]._index == 0
    assert qc.data[1].qubits[1]._index == 1
    assert qc.data[2].name == 'h'
    assert qc.data[2].qubits[0]._index == 0
    assert qc.data[3].name == 's'
    assert qc.data[3].qubits[0]._index == 0
    assert qc.data[4].name == 's'
    assert qc.data[4].qubits[0]._index == 1
    assert qc.data[5].name == 'cx'
    assert qc.data[5].qubits[0]._index == 0
    assert qc.data[5].qubits[1]._index == 1
    assert qc.data[6].name == 'h'
    assert qc.data[6].qubits[0]._index == 0
    expected_tableau = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
    assert np.array_equal(tableau, expected_tableau)

    qc = QuantumCircuit(2)
    tableau = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]])
    clifford.__perform_sweeping(tableau, qc)
    assert len(qc.data) == 1
    assert qc.data[0].name == 'swap'
    assert qc.data[0].qubits[0]._index == 0
    assert qc.data[0].qubits[1]._index == 1
    expected_tableau = np.array([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]])
    assert np.array_equal(tableau, expected_tableau)

    qc = QuantumCircuit(1)
    tableau = np.array([[0, 1, 0], [1, 0, 0]])
    clifford.__perform_sweeping(tableau, qc)
    assert len(qc.data) == 1
    assert qc.data[0].name == 'h'
    assert qc.data[0].qubits[0]._index == 0
    expected_tableau = np.array([[1, 0, 0], [0, 1, 0]])
    assert np.array_equal(tableau, expected_tableau)
