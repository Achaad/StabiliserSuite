import numpy as np

from numba.core.types import complex128
from clifford_builder.gate_utils import __are_equal_up_to_global_phase, rx, rz, __generate_two_qubit_gates, \
    generate_two_qubit_gates, __multiply_sequence
from qiskit.circuit.library import iSwapGate


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
    expected_keys = {'I⊗I', 'I⊗X', 'X⊗I', 'X⊗X', 'cx', 'iSwap'}
    assert set(result.keys()) == expected_keys

    gates_dict = {}
    result = __generate_two_qubit_gates(gates_dict)
    expected_result = {
        'cx': np.array([[1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0]], dtype=complex),
        'iSwap': iSwapGate().to_matrix()
    }
    assert set(result.keys()) == set(expected_result.keys())
    assert np.allclose(result['cx'], expected_result['cx'])
    assert np.allclose(result['iSwap'], expected_result['iSwap'])

    custom_gate = np.array([[1, 1], [1, -1]], dtype=complex)
    gates_dict = {'Custom': custom_gate}
    result = __generate_two_qubit_gates(gates_dict)
    expected_key = 'Custom⊗Custom'
    assert expected_key in result
    assert np.allclose(result[expected_key], np.kron(custom_gate, custom_gate))


def test_generate_two_qubit_gates():
    result = generate_two_qubit_gates()
    expected_keys = {'I⊗I', 'I⊗X', 'I⊗Y', 'I⊗Z', 'I⊗H', 'I⊗S',
                     'X⊗I', 'X⊗X', 'X⊗Y', 'X⊗Z', 'X⊗H', 'X⊗S',
                     'Y⊗I', 'Y⊗X', 'Y⊗Y', 'Y⊗Z', 'Y⊗H', 'Y⊗S',
                     'Z⊗I', 'Z⊗X', 'Z⊗Y', 'Z⊗Z', 'Z⊗H', 'Z⊗S',
                     'H⊗I', 'H⊗X', 'H⊗Y', 'H⊗Z', 'H⊗H', 'H⊗S',
                     'S⊗I', 'S⊗X', 'S⊗Y', 'S⊗Z', 'S⊗H', 'S⊗S',
                     'Rx(pi/2)⊗Rx(pi/2)', 'Rz(pi/2)⊗Rz(pi/2)', 'cx', 'iSwap'}
    assert set(result.keys()).issuperset(expected_keys)

    gates_dict = {}
    result = __generate_two_qubit_gates(gates_dict)
    expected_keys = {'cx', 'iSwap'}
    assert set(result.keys()) == expected_keys

    result = generate_two_qubit_gates()
    rx_pi = rx(np.pi)
    rz_pi = rz(np.pi)
    assert np.allclose(result['Rx(pi)⊗Rz(pi)'], np.kron(rx_pi, rz_pi))


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
