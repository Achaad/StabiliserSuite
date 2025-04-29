from typing import final

from _pytest.python_api import raises
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler import CouplingMap

from clifford_builder import circuit


def test_cnot():
    # Verify that CNOT is applied correctly when no coupling map is given
    qc = QuantumCircuit(3)
    circuit.cnot(qc, 0, 2)
    assert len(qc.data) == 1
    assert qc.data[0].name == 'cx'
    assert qc.data[0].qubits[0]._index == 0
    assert qc.data[0].qubits[1]._index == 2

    # Verify that error is raised in case of impossible operation
    coupling_map = CouplingMap([[0, 1], [2, 3]])
    qc = QuantumCircuit(4)
    with raises(ValueError):
        circuit.cnot(qc, 0, 3, coupling_map=coupling_map)

    # Verify that CNOT is applied correctly with a coupling map with direct connection
    coupling_map = CouplingMap([[0, 1], [1, 2]])
    qc = QuantumCircuit(3)
    circuit.cnot(qc, 1, 2, coupling_map=coupling_map)
    assert len(qc.data) == 1
    assert qc.data[0].name == 'cx'
    assert qc.data[0].qubits[0]._index == 1
    assert qc.data[0].qubits[1]._index == 2

    # Verify that CNOT is applied correctly with a coupling map with indirect connection
    coupling_map = CouplingMap([[0, 1], [2, 3], [1, 2]])
    qc = QuantumCircuit(4)
    circuit.cnot(qc, 0, 2, coupling_map=coupling_map)
    assert len(qc.data) == 4
    assert qc.data[0].name == 'cx'
    assert qc.data[0].qubits[0]._index == 0
    assert qc.data[0].qubits[1]._index == 1
    assert qc.data[1].name == 'cx'
    assert qc.data[1].qubits[0]._index == 1
    assert qc.data[1].qubits[1]._index == 2
    assert qc.data[2].name == 'cx'
    assert qc.data[2].qubits[0]._index == 0
    assert qc.data[2].qubits[1]._index == 1
    assert qc.data[3].name == 'cx'
    assert qc.data[3].qubits[0]._index == 1
    assert qc.data[3].qubits[1]._index == 2

    # Verify that the resulting operation is the same
    coupling_map = CouplingMap([[0, 1], [2, 3], [1, 2]])
    qc_test = QuantumCircuit(4)
    qc_control = QuantumCircuit(4)

    circuit.cnot(qc_test, 0, 2, coupling_map=coupling_map)
    qc_control.cx(0, 2)
    u_test = Operator(qc_test)
    u_control = Operator(qc_control)
    assert u_control.equiv(u_test)


def test_cz_using_iswap():
    # Verify that the resulting operation is the same
    coupling_map = CouplingMap([[0, 1],
                                # [2, 3],
                                [1, 2]])
    qc_test = QuantumCircuit(2)
    qc_control = QuantumCircuit(2)

    circuit.cz_using_rx(qc_test, 0, 1, coupling_map=coupling_map)
    qc_control.cz(0, 1)

    labels = [format(i, '02b') for i in range(16)]
    for label in labels:
        test_init = Statevector.from_label(label)
        test_control = Statevector.from_label(label)

        test = test_init.evolve(qc_test)
        control = test_control.evolve(qc_control)

        assert control.equiv(test)

    u_test = Operator(qc_test)
    u_control = Operator(qc_control)
    assert u_control.equiv(u_test)
