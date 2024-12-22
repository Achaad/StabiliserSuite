import numpy as np
import qiskit
from qiskit import QuantumCircuit
from matplotlib import pyplot as plt
import termtables


def sample_clifford_group(qubit_count: int, add_barriers: bool = False) -> QuantumCircuit:
    """
    Samples random Clifford operator.
    Uses algorithm presented by Ewout van den Berg (https://arxiv.org/pdf/2008.06011).
    :param qubit_count: The amount of qubits in the generated circuit.
    :param add_barriers: Whether the barriers should be added. NB! Barriers might slow down execution.
    :return: Generated quantum circuit.
    """
    qc: QuantumCircuit = qiskit.QuantumCircuit(qubit_count, qubit_count)

    for i in range(qubit_count):
        # This is a test code for having the same results as in the paper
        # if i == 0:
        #     row = np.array([[True, True, True, True, False, True, True, False, False], [True, True, True,
        #                                                                                 True, True, True, True,
        #                                                                                 False, False]])
        # if i == 1:
        #     row = np.array([[0, 0, 0, 0, 1, 0, 0], [1, 1, 0, 1, 1, 0, 0]])
        #
        # if i == 2:
        #     row = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]])
        #
        # if i == 3:
        #     row = np.array([[0, 1, 0], [1, 0, 0]])

        # Generate row
        row: np.ndarray = __generate_random_tableau(qubit_count - i)

        # Perform sweeping
        __perform_sweeping(row, qc, iteration=i)
        if add_barriers:
            qc.barrier()

    return qc


def __perform_sweeping(tableau: np.ndarray, qc: QuantumCircuit, iteration: int = 0) -> None:
    """
    Performs sweeping operation on the tableau.
    :param tableau: The tableau on which the sweeping operation is performed.
    :param qc: Quantum circuit which is being modified.
    :param iteration: Which sweeping operation is currently being performed.
    :return: None
    """
    # Normalisation has to be checked before each step in order to avoid additional computations
    if __check_normalisation(tableau):
        return
    __step_1(tableau, qc, shift=iteration)
    if __check_normalisation(tableau):
        return
    __step_2(tableau, qc, shift=iteration)
    if __check_normalisation(tableau):
        return
    __step_3(tableau, qc, shift=iteration)
    if __check_normalisation(tableau):
        return
    __step_4(tableau, qc, shift=iteration)
    if __check_normalisation(tableau):
        return

    # Apply steps 1 and 2 to the second row
    __step_1(tableau, qc, row_index=1, shift=iteration)
    if __check_normalisation(tableau):
        return
    __step_2(tableau, qc, row_index=1, shift=iteration)
    if __check_normalisation(tableau):
        return

    # Repeat step 4
    __step_4(tableau, qc, shift=iteration)
    if __check_normalisation(tableau):
        return
    __step_5(tableau, qc, shift=iteration)


def __step_1(tableau: np.ndarray, qc: QuantumCircuit, row_index: int = 0, shift: int = 0) -> None:
    """
    Performs the first step of the sweeping algorithm by clearing the Z-block of the given row.
    Column indexing is shifted from actual by shift value.
    :param tableau: Tableau on which the operation is being performed.
    :param qc: Quantum circuit which is being modified.
    :param row_index: Index of the row of the tableau.
    :param shift: Column index shift, corresponds to the sweeping iteration.
    :return: None
    """
    starting_index: int = (tableau.shape[1] - 1) // 2
    for i in range(starting_index):
        if tableau[row_index][starting_index + i]:
            if tableau[row_index][i]:
                __apply_s(tableau, qc, i, shift)
            else:
                __apply_hadamard(tableau, qc, i, shift)


def __step_2(tableau: np.ndarray, qc: QuantumCircuit, row_index: int = 0, shift: int = 0) -> None:
    """
    Performs the second step of the sweeping algorithm by leaving only one operation in X-block.
    Column indexing is shifted from actual by shift value.
    :param tableau: Tableau on which the operation is being performed.
    :param qc: Quantum circuit which is being modified.
    :param row_index: Index of the row of the tableau.
    :param shift: Column index shift, corresponds to the sweeping iteration.
    :return: None
    """
    # Determine the set of indices where the coefficient is non-zero
    coefficients = list()
    for i in range((tableau.shape[1] - 1) // 2):
        if tableau[row_index][i]:
            coefficients.append(i)

    temp = list()
    while len(coefficients) > 1:
        # Apply CNOT gate pairwise to qubits at even locations
        for i in range(len(coefficients) - 1):
            if i % 2 == 0:
                __apply_cnot(tableau, qc, coefficients[i], coefficients[i + 1], shift=shift)

        # Retain indices at even locations
        for i in range(len(coefficients)):
            if i % 2 == 0:
                temp.append(coefficients[i])

        coefficients.clear()
        coefficients.extend(temp)
        temp.clear()


def __step_3(tableau: np.ndarray, qc: QuantumCircuit, shift: int = 0) -> None:
    """
    Performs the third step of the sweeping algorithm by putting non-zero X-element into the first column.
    Column indexing is shifted from actual by shift value.
    :param tableau: Tableau on which the operation is being performed.
    :param qc: Quantum circuit which is being modified.
    :param shift: Column index shift, corresponds to the sweeping iteration.
    :return: None
    """
    # Determine the set of indices where the coefficient is non-zero
    coefficients: list[int] = list()
    for i in range((tableau.shape[1] - 1) // 2):
        if tableau[0][i]:
            coefficients.append(i)

    if len(coefficients) > 1:
        raise RuntimeError(f"Too many coefficients after step 2: {coefficients}")

    if len(coefficients) == 1 and coefficients[0] != 0:
        __apply_swap(tableau, qc, 0, coefficients[0], shift)


def __step_4(tableau: np.ndarray, qc: QuantumCircuit, shift: int = 0) -> None:
    """
    Performs the fourth step of the sweeping algorithm by clearing the Z-block of the second row.
    Column indexing is shifted from actual by shift value.
    :param tableau: Tableau on which the operation is being performed.
    :param qc: Quantum circuit which is being modified.
    :param shift: Column index shift, corresponds to the sweeping iteration.
    :return: None
    """
    z_start: int = (tableau.shape[1] - 1) // 2
    # If the second Pauli is equal to +- Z_1, we skip this step
    if not tableau[1][0] and tableau[1][z_start]:
        return

    __apply_hadamard(tableau, qc, 0, shift=shift)


def __step_5(tableau: np.ndarray, qc: QuantumCircuit, shift: int = 0) -> None:
    """
    Performs the fifth step of the sweeping algorithm by clearing sign bits.
    Column indexing is shifted from actual by shift value.
    :param tableau: Tableau on which the operation is being performed.
    :param qc: Quantum circuit which is being modified.
    :param shift: Column index shift, corresponds to the sweeping iteration.
    :return: None
    """
    phase_index: int = tableau.shape[1] - 1
    if not tableau[0][phase_index] and tableau[1][phase_index]:
        __apply_pauli(tableau, qc, 0, 'x', shift=shift)
    if tableau[0][phase_index]:
        if tableau[1][phase_index]:
            __apply_pauli(tableau, qc, 0, 'y', shift=shift)
        else:
            __apply_pauli(tableau, qc, 0, 'z', shift=shift)


def __generate_random_tableau(qubit_count: int) -> np.ndarray:
    """
    Generates a random tableau row.

    :param qubit_count: The amount of qubits in a tableau.
    :return: Randomly sampled tableau.
    """
    rng = np.random.default_rng()
    return rng.choice([False, True], size=(2, 2 * qubit_count + 1))


def __apply_hadamard(tableau: np.ndarray, qc: QuantumCircuit, qubit: int, shift: int = 0) -> None:
    """
    Applies Hadamard gate to the circuit and modifies tableau.
    :param tableau: Tableau to be modified.
    :param qc: Quantum circuit which is being modified.
    :param qubit: The qubit to which the gate is applied.
    :param shift: Shift in qubit indexing.
    :return: None
    """
    qc.h(qubit + shift)
    z_start: int = (tableau.shape[1] - 1) // 2

    # Swaps columns in the tableau representation
    for row in tableau:
        temp = row[qubit]
        row[qubit] = row[z_start + qubit]
        row[z_start + qubit] = temp


def __apply_s(tableau: np.ndarray, qc: QuantumCircuit, qubit: int, shift: int = 0) -> None:
    """
    Applies S (phase shift) gate to the circuit and modifies tableau.
    :param tableau: Tableau to be modified.
    :param qc: Quantum circuit which is being modified.
    :param qubit: The qubit to which the gate is applied.
    :param shift: Shift in qubit indexing.
    :return: None
    """
    qc.s(qubit + shift)
    z_start: int = (tableau.shape[1] - 1) // 2

    # Swaps columns in the tableau representation
    for row in tableau:
        row[z_start + qubit] = row[z_start + qubit] != row[qubit]


def __apply_cnot(tableau: np.ndarray, qc: QuantumCircuit, control_qubit: int, target_qubit: int,
                 shift: int = 0) -> None:
    """
    Applies Hadamard gate to the circuit and modifies tableau.
    :param tableau: Tableau to be modified.
    :param qc: Quantum circuit which is being modified.
    :param control_qubit: The control qubit to which the gate is applied.
    :param target_qubit: The target qubit to which the gate is applied.
    :param shift: Shift in qubit indexing.
    :return: None
    """
    qc.cx(control_qubit + shift, target_qubit + shift)
    z_start: int = (tableau.shape[1] - 1) // 2

    # Swaps columns in the tableau representation
    for row in tableau:
        row[target_qubit] = row[target_qubit] != row[control_qubit]
        row[z_start + control_qubit] = row[z_start + control_qubit] != row[z_start + target_qubit]


def __apply_swap(tableau: np.ndarray, qc: QuantumCircuit, qubit_1: int, qubit_2: int, shift: int = 0) -> None:
    """
    Applies Swap (three cnot) gate to the circuit and modifies tableau.
    :param tableau: Tableau to be modified.
    :param qc: Quantum circuit which is being modified.
    :param qubit_1: The first qubit to which the gate is applied.
    :param qubit_2: The second qubit to which the gate is applied.
    :param shift: Shift in qubit indexing.
    :return: None
    """
    qc.swap(qubit_1 + shift, qubit_2 + shift)  # Shift already applied to qubit 2
    z_start: int = (tableau.shape[1] - 1) // 2
    for row in tableau:
        # X-swap
        temp = row[qubit_1]
        row[qubit_1] = row[qubit_2]
        row[qubit_2] = temp

        # Z-swap
        temp = row[qubit_1 + z_start]
        row[qubit_1 + z_start] = row[qubit_2 + z_start]
        row[qubit_2 + z_start] = temp


def __check_normalisation(tableau: np.ndarray) -> bool:
    """
    Verifies tableau normalisation.
    :param tableau: Tableau to be checked.
    :return: True if tableau is normalised.
    """
    z_start: int = (tableau.shape[1] - 1) // 2
    return np.array_equal(np.nonzero(tableau[0])[0], [0]) and np.array_equal(np.nonzero(tableau[1])[0], [z_start])


def __get_pauli(tableau: np.ndarray, row: int, qubit: int) -> str:
    """
    Calculates the Pauli gate coded by the given qubit in the tableau.
    :param tableau: Tableau representation the circuit.
    :param row: Row for which the Pauli gate is calculated.
    :param qubit: Index of qubit.
    :return: Name of the Pauli gate.
    """
    z_start: int = (tableau.shape[1] - 1) // 2
    if tableau[row][qubit]:
        if tableau[row][z_start + qubit]:
            return 'y'
        else:
            return 'x'
    else:
        if tableau[row][z_start + qubit]:
            return 'z'

    return 'I'


def __apply_pauli(tableau: np.ndarray, qc: QuantumCircuit, qubit: int, pauli: str, shift: int = 0) -> None:
    """
    Applies Pauli gate to the circuit and modifies tableau.
    :param tableau: Tableau to be modified.
    :param qc: Quantum circuit which is being modified.
    :param qubit: The qubit to which the gate is applied.
    :param pauli: Pauli gate.
    :param shift: Shift in qubit indexing.
    :return: None
    """
    pauli_lower = pauli.lower()
    if pauli_lower == 'x':
        qc.x(qubit + shift)
    elif pauli_lower == 'y':
        qc.y(qubit + shift)
    elif pauli_lower == 'z':
        qc.z(qubit + shift)
    else:
        raise ValueError(f"Unknown pauli: {pauli}")
    for i, row in enumerate(tableau):
        row_pauli = __get_pauli(tableau, i, qubit)
        if row_pauli != pauli_lower and row_pauli != 'I':
            row[-1] = not row[-1]


def print_tableau(tableau: np.ndarray, representation: str = "text") -> None:
    """
    Prints tableau representation of the circuit.
    :param tableau: Tableau to be printed.
    :param representation: The representation model. "text" for text, and "mpl" for matplotlib.
    :return: None
    """
    qubit_count: int = (tableau.shape[1] - 1) // 2

    if representation == "text":
        header: list[str] = list()
        for i in range(qubit_count):
            header.append(f"X{i}")
        for i in range(qubit_count):
            header.append(f"Z{i}")
        header.append("S")

        termtables.print(np.vectorize(lambda t: "1" if t else "0")(tableau), header)
        return
    if representation == "mpl":
        raise NotImplementedError("Matplotlib representation is not implemented. See issue #8")
    else:
        raise ValueError(f"Unknown representation: {representation}")
