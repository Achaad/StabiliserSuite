from stabiliser_suite import utils

import numpy as np


def test__find_qubit_count() -> None:
    tableau = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    assert utils.__find_qubit_count(tableau) == 2


def test__do_paulis_anticommute() -> None:
    tableau = np.array([[1, 0, 0, 0, 1, 0, 0], [1, 0, 1, 1, 0, 0, 0]])
    assert utils.__do_paulis_anticommute(tableau) == True
