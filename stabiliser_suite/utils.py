import numpy as np


def __find_qubit_count(tableau: np.ndarray) -> int:
    """
    Finds the amount of qubits present in the tableau.

    :param tableau: Tableau representation of the circuit
    :return: Amount of qubits present in the tableau
    """
    return (tableau.shape[1] - 1) // 2


def __do_paulis_anticommute(tableau: np.ndarray) -> bool:
    qubit_count = __find_qubit_count(tableau)

    x1_z2_product = 0
    x2_z1_product = 0
    for i in range(qubit_count):
        x1_z2_product += (tableau[0][i] + tableau[1][i + qubit_count])
        x2_z1_product += (tableau[1][i] + tableau[0][i + qubit_count])

    return False if (x1_z2_product + x2_z1_product) % 2 == 0 else True
