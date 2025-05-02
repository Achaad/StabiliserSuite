# Define Pauli and standard single-qubit gates
import inspect
import itertools

import numpy as np
from numba import njit
from numba.core.types import complex128
from qiskit import QiskitError
from qiskit.circuit import Gate, Instruction
from qiskit.circuit.library import iSwapGate, XGate, YGate, ZGate, IGate, HGate, SGate, CZGate, SwapGate, standard_gates
from qiskit.quantum_info import Clifford
from tqdm import tqdm

# Define basic gates
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)


@njit(cache=True)
def rx(theta):
    """
        Generate the rotation matrix for the X-axis (Rx) in quantum computing.

        This function creates a 2x2 unitary matrix representing a rotation
        around the X-axis by a given angle `theta`.

        Args:
            theta (float): The rotation angle in radians.

        Returns:
            np.ndarray: A 2x2 complex-valued numpy array representing the Rx matrix.
        """
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)]
    ], dtype=complex128)


@njit(cache=True)
def rz(theta):
    """
    Generate the rotation matrix for the Z-axis (Rz) in quantum computing.

    This function creates a 2x2 unitary matrix representing a rotation
    around the Z-axis by a given angle `theta`.

    Args:
        theta (float): The rotation angle in radians.

    Returns:
        np.ndarray: A 2x2 complex-valued numpy array representing the Rz matrix.
    """
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex128)


def __generate_two_qubit_gates(gates_dict):
    tensor_gates = {}
    for name1, gate1 in gates_dict.items():
        for name2, gate2 in gates_dict.items():
            name = f"{name1}⊗{name2}"
            tensor_gates[name] = np.kron(gate1, gate2)
    tensor_gates['cx'] = np.array([[1, 0, 0, 0],
                                   [0, 0, 0, 1],
                                   [0, 0, 1, 0],
                                   [0, 1, 0, 0]], dtype=complex)
    tensor_gates['iSwap'] = iSwapGate().to_matrix()
    return tensor_gates


def generate_two_qubit_clifford_gates():
    """
    Generate a dictionary of two-qubit Clifford gates.

    This function creates a collection of two-qubit Clifford gates, including
    standard single-qubit gates, rotation gates, and special gates like iSwap and CZ.
    It also generates tensor products of sinqle-qubit gates to form two-qubit gates.

    Returns:
        dict: A dictionary where keys are gate names (str) and values are 2D numpy arrays
              representing the corresponding gate matrices.
    """
    # Define standard single-qubit gates
    gates = {
        IGate().name: IGate().to_matrix(),
        XGate().name: XGate().to_matrix(),
        YGate().name: YGate().to_matrix(),
        ZGate().name: ZGate().to_matrix(),
        HGate().name: HGate().to_matrix(),
        SGate().name: SGate().to_matrix(),
    }

    # Define rotation gates with various angles
    thetas = {
        'pi/2': np.pi / 2,
        'pi': np.pi, '3pi/2': 3 * np.pi / 2, '-3pi/2': -3 * np.pi / 2, '-pi': -np.pi, '-pi/2': - np.pi / 2
    }
    for theta, angle in thetas.items():
        gates.update({
            f'Rx({theta})': rx(angle),
            f'Rz({theta})': rz(angle)
        })

    # Add special two-qubit gates
    gates.update({
        iSwapGate().name: iSwapGate().to_matrix(),
        CZGate().name: CZGate().to_matrix(),
        SwapGate().name: SGate().to_matrix()
    })

    # Generate tensor products of the gates
    return __generate_two_qubit_gates(gates)


@njit
def __are_equal_up_to_global_phase(u1, u2, atol=1e-8):
    """
        Check if two unitary matrices are equal up to a global phase.

        This function determines whether two unitary matrices, `U1` and `U2`,
        are equivalent up to a global phase factor. A global phase factor means
        that the matrices differ only by a scalar multiplication of a complex
        exponential (e.g., e^(i*theta)).

        Args:
            u1 (np.ndarray): The first unitary matrix.
            u2 (np.ndarray): The second unitary matrix.
            atol (float): Absolute tolerance for numerical comparison. Default is 1e-8.

        Returns:
            bool: True if the matrices are equal up to a global phase, False otherwise.
        """
    u = u1 @ u2.conj().T
    phase = np.diag(u)[0]

    return np.allclose(u, phase * np.eye(u.shape[0]), atol=atol)


@njit
def __multiply_sequence(matrices):
    """
    Multiply a sequence of matrices in order.

    This function takes a list of matrices and computes their product
    in the given order. The multiplication is performed iteratively.

    Args:
        matrices (list of np.ndarray): A list of 2D numpy arrays (matrices) to be multiplied.

    Returns:
        np.ndarray: The resulting matrix after multiplying all matrices in the sequence.
    """
    result = np.eye(matrices[0].shape[0], dtype=complex128)
    for m in matrices:
        result = result @ m
    return result


def find_matching_combinations(gate_dict, target_matrix, output, max_depth=3, allow_global_phase=True):
    """
    Try all sequences of gate multiplications to match the target matrix.

    :param gate_dict: dict of name -> 2D np.array
    :param target_matrix: np.array target to match
    :param max_depth: max number of gates to combine
    :param allow_global_phase: allow match up to global phase
    :return: list of (sequence of gate names, result matrix)
    """
    results = []
    with open(output, 'a') as f:
        try:
            gate_names = list(gate_dict.keys())

            for depth in range(1, max_depth + 1):
                print(f"Checking depth {depth} ({len(gate_names) ** depth} combinations)...")
                for sequence in tqdm(itertools.product(gate_names, repeat=depth), total=len(gate_names) ** depth):
                    result = np.eye(target_matrix.shape[0], dtype=complex)
                    for gate_name in sequence:
                        result = result @ gate_dict[gate_name]
                    if (allow_global_phase and __are_equal_up_to_global_phase(result, target_matrix)) or \
                            (not allow_global_phase and np.allclose(result, target_matrix, atol=1e-8)):
                        results.append((sequence, result))

                solutions = [a for a, _ in results]
                print(f"Current solutions at depth {depth}: {solutions}", flush=True)
                f.write(f"Current solutions at depth {depth}: {solutions}\n")
                f.flush()
        finally:
            f.write('\n')
            f.write('COMPLETE RESULTS:')
            f.write(', '.join(f'({a}, {b})' for a, b in results))
            f.flush()
            print(results)


def is_clifford_gate(gate: Gate) -> bool:
    """
    Check if a given quantum gate is a Clifford gate.

    This function determines whether a provided quantum gate belongs to the
    Clifford group. The Clifford group consists of gates that map Pauli operators
    to other Pauli operators under conjugation.

    Args:
        gate (Gate): A quantum gate object from Qiskit.

    Returns:
        bool: True if the gate is a Clifford gate, False otherwise.
    """
    try:
        Clifford(gate)
        return True
    except QiskitError:
        return False


def get_gates() -> dict[str, Gate | Instruction]:
    """
    Retrieve a dictionary of quantum gates and their instances.

    This function iterates through all gate classes in the `standard_gates` module,
    checks if they are subclasses of `Gate`, and attempts to instantiate them.
    If a gate class can be instantiated without parameters, it is added to the dictionary.
    If a gate class requires parameters, instances are created for angles ranging
    from -3π/2 to 3π/2 in steps of π/4, and these instances are added to the dictionary
    with appropriately formatted names.

    Returns:
        dict: A dictionary where keys are gate names (or parameterized gate names)
              and values are the corresponding gate instances.
    """
    gate_classes = {}
    for name, gate_class in vars(standard_gates).items():
        if inspect.isclass(gate_class) and issubclass(gate_class, Gate):
            gates = __try_instantiate_gate(gate_class)
            if gates:
                thetas = [f"{i}pi/4" if i != 0 else "0" for i in range(-6, 7)]
                gate_classes.update(
                    {f"{name}({theta})": gate for theta, gate in zip(thetas, gates)}
                    if len(gates) > 1 else {name: gates[0]}
                )
    return gate_classes


def get_non_clifford_gates() -> dict:
    """
    Retrieve a dictionary of non-Clifford quantum gates.

    This function filters out Clifford gates from the complete set of quantum gates
    obtained using `get_gates()`. It identifies non-Clifford gates by checking each
    gate with the `is_clifford_gate` function.

    Returns:
        dict: A dictionary where keys are the names of non-Clifford gates (str) and
              values are the corresponding gate instances.
    """
    all_gate_classes = get_gates()
    gates = {name: gate_class for name, gate_class in all_gate_classes.items() if not is_clifford_gate(gate_class)}
    return gates


def __try_instantiate_gate(gate_class, control: int = 0, target: int = 1, num_control_qubits: int = 1) -> list[Gate]:
    """
    Attempt to instantiate a quantum gate class or generate parameterized instances.

    This function tries to create an instance of the provided quantum gate class.
    If the class is not a subclass of `Gate`, it returns an empty list. If the class
    requires parameters for instantiation, it generates instances based on the
    required parameters:
    - For gates requiring `num_ctrl_qubits`, it creates an instance with the specified
      number of control qubits.
    - For gates requiring a single parameter `phi`, it generates instances with angles
      ranging from -3π/2 to 3π/2 in steps of π/4.
    - For gates requiring two parameters, it uses the provided `control` and `target` values.

    Args:
        gate_class (type): The class of the quantum gate to instantiate.
        control (int, optional): The control qubit index for two-parameter gates. Defaults to 0.
        target (int, optional): The target qubit index for two-parameter gates. Defaults to 1.
        num_control_qubits (int, optional): The number of control qubits for gates requiring this parameter. Defaults to 1.

    Returns:
        list[Gate]: A list of instantiated quantum gate objects, or an empty list
                    if the class is not a valid quantum gate class or cannot be instantiated.
    """
    if not issubclass(gate_class, Gate):
        return []
    try:
        return [gate_class()]
    except TypeError:

        sig = inspect.signature(gate_class.__init__)
        # Filter out 'self' and kwargs
        required_params = [
            p for p in sig.parameters.values()
            if p.name != 'self' and p.default == inspect.Parameter.empty and p.kind in (
                inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]

        # Generate phis in range (-3π/2, 3π/2) with step π/4
        phis = [i * np.pi / 4 for i in range(-6, 7, 1)]

        try:
            if len(required_params) == 1 and required_params[0].name == 'num_ctrl_qubits':
                return [gate_class(num_control_qubits)]
            elif len(required_params) == 1 and required_params[0].name == 'phi':
                return [gate_class(phi) for phi in phis]
            elif len(required_params) == 2:
                return [gate_class(control, target)]
        except TypeError:
            return []

    return []
