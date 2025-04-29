import numpy as np

from clifford_builder.gate_utils import __are_equal_up_to_global_phase


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