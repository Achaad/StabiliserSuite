import numpy as np
import qiskit
from qiskit import quantum_info, QuantumCircuit
from matplotlib import pyplot as plt
import termtables
from qiskit.quantum_info import Clifford


def generate_clifford_group(qubit_count: int, add_barriers: bool = False) -> qiskit.QuantumCircuit:
    qc: QuantumCircuit = qiskit.QuantumCircuit(qubit_count, qubit_count)
    # tableau: np.ndarray = np.array([[True, True, True, True, False, True, True, False, False], [True, True, True,
    #                                                                                             True, True, True, True,
    #                                                                                             False, False]])
    for i in range(qubit_count):
        # TODO: remove this! It's just for testing
        row: np.ndarray = None
        if i == 0:
            row = np.array([[True, True, True, True, False, True, True, False, False], [True, True, True,
                                                                                        True, True, True, True,
                                                                                        False, False]])
        if i == 1:
            row = np.array([[0, 0, 0, 0, 1, 0, 0], [1, 1, 0, 1, 1, 0, 0]])

        if i == 2:
            row = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]])

        if i == 3:
            row = np.array([[0, 1, 0], [0, 1, 0]])

        # Generate row
        # row: np.ndarray = __generate_random_tableau(qubit_count - i)
        # if i == 1:
        #     print(f"row: {i}")
        #     print_tableu(row)
        # Perform sweeping
        __perform_sweeping(row, qc, iteration=i, add_barriers=add_barriers)
        # if i == 1:
        #     print_tableu(row)
    # print("Generated base")
    # print_tableu(tableau)
    # __step_1(tableau, qc)
    # __step_2(tableau, qc)
    # __step_3(tableau, qc)
    # __step_4(tableau, qc)
    #
    # # Apply steps 1 and 2 to the second row
    # __step_1(tableau, qc, row_index=1)
    # __step_2(tableau, qc, row_index=1)
    #
    # # Repeat step 4
    # __step_4(tableau, qc)
    # __step_5(tableau, qc)
    #
    # if add_barriers:
    #     qc.barrier()

    # print("Final tableau")
    # print_tableu(tableau)
    return qc


def __perform_sweeping(tableau: np.ndarray, qc: QuantumCircuit, iteration: int = 0,
                       add_barriers: bool = False) -> None:
    __step_1(tableau, qc, shift=iteration)
    __step_2(tableau, qc, shift=iteration)
    __step_3(tableau, qc)
    __step_4(tableau, qc)

    # Apply steps 1 and 2 to the second row
    __step_1(tableau, qc, row_index=1, shift=iteration)
    __step_2(tableau, qc, row_index=1)

    # Repeat step 4
    __step_4(tableau, qc)
    __step_5(tableau, qc)

    if add_barriers:
        qc.barrier()


def __step_1(tableau: np.ndarray, qc: QuantumCircuit, row_index: int = 0, shift: int = 0) -> None:
    starting_index: int = (tableau.shape[1] - 1) // 2
    for i in range(starting_index):
        if tableau[row_index][starting_index + i]:
            if tableau[row_index][i]:
                __apply_s(tableau, qc, i, shift)
            else:
                __apply_hadamard(tableau, qc, i, shift)


def __step_2(tableau: np.ndarray, qc: QuantumCircuit, row_index: int = 0, shift: int = 0) -> None:
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
                __apply_cnot(tableau, qc, coefficients[i], coefficients[i + 1])

        # Retain indices at even locations
        for i in range(len(coefficients)):
            if i % 2 == 0:
                temp.append(coefficients[i])

        coefficients.clear()
        coefficients.extend(temp)
        temp.clear()

def __step_3(tableau: np.ndarray, qc: QuantumCircuit) -> None:
    # Determine the set of indices where the coefficient is non-zero
    coefficients: list[int] = list()
    for i in range((tableau.shape[1] - 1) // 2):
        if tableau[0][i]:
            coefficients.append(i)

    if len(coefficients) > 1:
        raise RuntimeError(f"Too many coefficients after step 2: {coefficients}")

    print("Step 3 is not yet implemented")


def __step_4(tableau: np.ndarray, qc: QuantumCircuit) -> None:
    z_start: int = (tableau.shape[1] - 1) // 2
    # If the second Pauli is equal to +- Z_1, we skip this step
    # TODO: verify this case
    if not tableau[1][0] and tableau[1][z_start]:
        return

    __apply_hadamard(tableau, qc, 0)


def __step_5(tableau: np.ndarray, qc: QuantumCircuit) -> None:
    phase_index: int = tableau.shape[1] - 1
    if not tableau[0][phase_index] and tableau[1][phase_index]:
        __apply_pauli(tableau, qc, 0, 'x')
    if tableau[0][phase_index]:
        if tableau[1][phase_index]:
            __apply_pauli(tableau, qc, 0, 'y')
        else:
            __apply_pauli(tableau, qc, 0, 'z')


def __generate_random_tableau(qubit_count: int) -> np.ndarray:
    """
    Generates a random tableau row.

    :param qubit_count: The amount of qubits in a tableau
    :return: Randomly sampled tableau
    """
    rng = np.random.default_rng()
    return rng.choice([False, True], size=(2, 2 * qubit_count + 1))


def __apply_hadamard(tableau: np.ndarray, qc: QuantumCircuit, qubit: int, shift: int = 0) -> None:
    qc.h(qubit + shift)
    z_start: int = (tableau.shape[1] - 1) // 2

    # Swaps columns in the tableu representation
    for row in tableau:
        temp = row[qubit]
        row[qubit] = row[z_start + qubit]
        row[z_start + qubit] = temp


def __apply_s(tableau: np.ndarray, qc: QuantumCircuit, qubit: int, shift: int = 0) -> None:
    qc.s(qubit + shift)
    z_start: int = (tableau.shape[1] - 1) // 2

    # Swaps columns in the tableu representation
    for row in tableau:
        row[z_start + qubit] = row[z_start + qubit] != row[qubit]


def __apply_cnot(tableau: np.ndarray, qc: QuantumCircuit, control_qubit: int, target_qubit: int,
                 shift: int = 0) -> None:
    qc.cx(control_qubit + shift, target_qubit + shift)
    z_start: int = (tableau.shape[1] - 1) // 2

    # Swaps columns in the tableu representation
    for row in tableau:
        row[target_qubit] = row[target_qubit] != row[control_qubit]
        row[z_start + control_qubit] = row[z_start + control_qubit] != row[z_start + target_qubit]


# TODO: change to enum
def __get_pauli(tableau: np.ndarray, row: int, qubit: int) -> str:
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


def __apply_pauli(tableau: np.ndarray, qc: QuantumCircuit, qubit: int, pauli: str) -> None:
    pauli_lower = pauli.lower()
    if pauli_lower == 'x':
        qc.x(qubit)
    elif pauli_lower == 'y':
        qc.y(qubit)
    elif pauli_lower == 'z':
        qc.z(qubit)
    else:
        raise ValueError(f"Unknown pauli: {pauli}")

    for i, row in enumerate(tableau):
        row_pauli = __get_pauli(tableau, i, qubit)
        if row_pauli != pauli_lower and row_pauli != 'I':
            row[-1] = not row[-1]


# TODO: representation should be enum?
# TODO: add matplotlib representation
def print_tableu(tableau: np.ndarray, representation: str = "text") -> None:
    qubit_count: int = (tableau.shape[1] - 1) // 2

    if representation == "text":
        header: list[str] = list()
        for i in range(qubit_count):
            header.append(f"X{i}")
        for i in range(qubit_count):
            header.append(f"Z{i}")
        header.append("S")

        termtables.print(np.vectorize(lambda t: "1" if t else "0")(tableau), header)


if __name__ == '__main__':
    # tableau: np.ndarray = __generate_random_tableau(3)
    # tableau: np.ndarray = np.array([[True, True, True, True, False, True, True, False, False], [True, True, True,
    #                                                                                             True, True, True, True,
    #                                                                                             False, False]])
    test = False
    if test:
        qc = QuantumCircuit(2)
        qc.z(0)
        qc.h(0)
        qc.x(0)
        c = Clifford(qc)
        print(c.tableau)
        print(c)

        qc.z(1)
        c = Clifford(qc)
        print(c.tableau)
        print(c)
    else:
        qc = generate_clifford_group(4, True)

    qc.draw("mpl")
    plt.show()
