import numpy as np

from qiskit.circuit import Instruction
from qiskit.circuit.library import iSwapGate, HGate, TGate, RZGate, XGate, CXGate, MCXGate, CZGate, SwapGate
from qiskit.circuit.gate import Gate
from sympy.testing.pytest import raises

from stabiliser_suite.gate_utils import __are_equal_up_to_global_phase, rx, rz, __generate_two_qubit_gates, \
    generate_two_qubit_clifford_gates, __multiply_sequence, is_clifford_gate, \
    get_non_clifford_gates, __try_instantiate_gate, get_gates, find_non_clifford_transformations


def test___are_equal_up_to_global_phase():
    matrix1 = np.array([[1, 0], [0, 1]], dtype=complex)
    matrix2 = np.array([[1, 0], [0, 1]], dtype=complex)
    assert __are_equal_up_to_global_phase(matrix1, matrix2)

    matrix1 = np.array([[1, 0], [0, 1]], dtype=complex)
    matrix2 = np.array([[1, 0], [0, 1]], dtype=complex)
    assert __are_equal_up_to_global_phase(matrix1, matrix2)

    matrix1 = np.zeros((2, 2), dtype=complex)
    matrix2 = np.zeros((2, 2), dtype=complex)
    assert __are_equal_up_to_global_phase(matrix1, matrix2)

    matrix1 = np.array([[1e10, 0], [0, 1e10]], dtype=complex)
    matrix2 = np.exp(1j * np.pi / 3) * matrix1
    assert __are_equal_up_to_global_phase(matrix1, matrix2)

    matrix1 = np.array([[np.nan, 0], [0, 1]], dtype=complex)
    matrix2 = np.array([[1, 0], [0, 1]], dtype=complex)
    assert not __are_equal_up_to_global_phase(matrix1, matrix2)


def test_rx():
    theta = 0
    expected = np.array([[1, 0], [0, 1]], dtype=complex)
    assert np.allclose(rx(theta), expected)

    theta = np.pi
    expected = np.array([[0, -1j], [-1j, 0]], dtype=complex)
    assert np.allclose(rx(theta), expected)

    theta = -np.pi / 2
    expected = np.array([[np.sqrt(2) / 2, 1j * np.sqrt(2) / 2], [1j * np.sqrt(2) / 2, np.sqrt(2) / 2]],
                        dtype=complex)
    assert np.allclose(rx(theta), expected)

    rng = np.random.default_rng(123)
    theta = rng.uniform(-2 * np.pi, 2 * np.pi)
    matrix = rx(theta)
    identity = np.eye(2, dtype=complex)
    assert np.allclose(matrix @ matrix.conj().T, identity)


def test_rz():
    theta = 0
    expected = np.array([[1, 0], [0, 1]], dtype=complex)
    assert np.allclose(rz(theta), expected)

    theta = np.pi
    expected = np.array([[np.exp(-1j * np.pi / 2), 0], [0, np.exp(1j * np.pi / 2)]], dtype=complex)
    assert np.allclose(rz(theta), expected)

    theta = -np.pi / 2
    expected = np.array([[np.exp(1j * np.pi / 4), 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
    assert np.allclose(rz(theta), expected)

    rng = np.random.default_rng(123)
    theta = rng.uniform(-2 * np.pi, 2 * np.pi)
    matrix = rz(theta)
    identity = np.eye(2, dtype=complex)
    assert np.allclose(matrix @ matrix.conj().T, identity)


def test___generate_two_qubit_gates():
    gates_dict = {'I': np.eye(2, dtype=complex), 'X': np.array([[0, 1], [1, 0]], dtype=complex)}
    result = __generate_two_qubit_gates(gates_dict)
    expected_keys = {'cz', 'I⊗X', 'cx', 'X⊗X', 'swap', 'iswap', 'X⊗I', 'I⊗I'}
    assert set(result.keys()) == expected_keys

    gates_dict = {}
    result = __generate_two_qubit_gates(gates_dict)
    expected_result = {
        'cx': np.array([[1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0]], dtype=complex),
        iSwapGate().name: iSwapGate().to_matrix(),
        CZGate().name: CZGate().to_matrix(),
        SwapGate().name: SwapGate().to_matrix()
    }
    assert set(result.keys()) == set(expected_result.keys())
    assert np.allclose(result['cx'], expected_result['cx'])
    assert np.allclose(result['iswap'], expected_result['iswap'])

    custom_gate = np.array([[1, 1], [1, -1]], dtype=complex)
    gates_dict = {'Custom': custom_gate}
    result = __generate_two_qubit_gates(gates_dict)
    expected_key = 'Custom⊗Custom'
    assert expected_key in result
    assert np.allclose(result[expected_key], np.kron(custom_gate, custom_gate))


def test_generate_two_qubit_gates():
    result = generate_two_qubit_clifford_gates()
    expected_keys = {'id⊗id', 'x⊗x', 'y⊗y', 'z⊗z', 'h⊗h', 's⊗s'}
    assert set(result.keys()).issuperset(expected_keys)

    result = generate_two_qubit_clifford_gates()
    expected_keys = {'Rx(pi/2)⊗Rx(pi/2)', 'Rz(pi/2)⊗Rz(pi/2)', 'Rx(pi)⊗Rx(pi)', 'Rz(pi)⊗Rz(pi)'}
    assert set(result.keys()).issuperset(expected_keys)

    result = generate_two_qubit_clifford_gates()
    expected_keys = {'cx', 'iswap', 'cz', 'swap'}
    assert set(result.keys()).issuperset(expected_keys)

    gates_dict = {}
    result = __generate_two_qubit_gates(gates_dict)
    expected_keys = {'cx', 'iswap', 'cz', 'swap'}
    assert set(result.keys()) == expected_keys

    custom_gate = np.array([[1, 1], [1, -1]], dtype=complex)
    gates_dict = {'Custom': custom_gate}
    result = __generate_two_qubit_gates(gates_dict)
    expected_key = 'Custom⊗Custom'
    assert expected_key in result
    assert np.allclose(result[expected_key], np.kron(custom_gate, custom_gate))


def test___multiply_sequence():
    matrices = [np.array([[2, 0], [0, 2]], dtype=np.complex128)]
    result = __multiply_sequence(matrices)
    expected = matrices[0]
    assert np.allclose(result, expected)

    matrices = [
        np.array([[1, 2], [3, 4]], dtype=np.complex128),
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.array([[2, 0], [0, 2]], dtype=np.complex128)
    ]
    result = __multiply_sequence(matrices)
    expected = matrices[0] @ matrices[1] @ matrices[2]
    assert np.allclose(result, expected)

    matrices = [
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.complex128),
        np.array([[7, 8], [9, 10], [11, 12]], dtype=np.complex128)
    ]
    result = __multiply_sequence(matrices)
    expected = matrices[0] @ matrices[1]
    assert np.allclose(result, expected)

    matrices = [
        np.array([[1, 2], [3, 4]], dtype=np.complex128),
        np.array([[1, 2, 3]], dtype=np.complex128)
    ]
    try:
        __multiply_sequence(matrices)
        assert False, "Expected ValueError for incompatible matrices"
    except ValueError:
        pass


def test_is_clifford_gate():
    assert is_clifford_gate(HGate())
    assert not is_clifford_gate(TGate())


def test___try_instantiate_gate():
    gates = __try_instantiate_gate(XGate)
    assert len(gates) == 1
    assert isinstance(gates[0], Instruction)
    assert isinstance(gates[0], XGate)

    gates = __try_instantiate_gate(RZGate)
    assert len(gates) == 13
    assert isinstance(gates[0], Instruction)
    assert isinstance(gates[0], RZGate)

    gates = __try_instantiate_gate(np.complex128)
    assert len(gates) == 0

    gates = __try_instantiate_gate(CXGate)
    assert len(gates) == 1
    assert isinstance(gates[0], Instruction)
    assert isinstance(gates[0], CXGate)

    gates = __try_instantiate_gate(MCXGate, num_control_qubits=3)
    assert len(gates) == 1
    assert isinstance(gates[0], Instruction)
    assert isinstance(gates[0], MCXGate)

    gates = __try_instantiate_gate(MCXGate)
    assert len(gates) == 1
    assert isinstance(gates[0], Instruction)
    assert isinstance(gates[0], CXGate)  # Should resolve to 1-control CX gate


def test_get_gates():
    gates = get_gates()
    assert "XGate" in gates
    assert "HGate" in gates
    assert "RZGate(1pi/4)" in gates
    assert isinstance(gates["XGate"], XGate)
    assert isinstance(gates["RZGate(1pi/4)"], RZGate)


def test_get_non_clifford_gates():
    gates = get_non_clifford_gates()
    assert len(gates) > 0
    assert "TGate" in gates
    assert isinstance(gates["TGate"], TGate)
    assert "HGate" not in gates
