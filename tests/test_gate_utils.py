import numpy as np

from clifford_builder.gate_utils import __are_equal_up_to_global_phase, rx


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
