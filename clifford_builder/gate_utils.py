# Define Pauli and standard single-qubit gates
import itertools

import numpy as np
from numba import njit
from numba.core.types import complex128
from qiskit.circuit.library import iSwapGate
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


def rz(theta):
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)


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


def generate_two_qubit_gates():
    gates = {
        'I': I,
        'X': X,
        'Y': Y,
        'Z': Z,
        'H': H,
        'S': S,
    }

    thetas = {
        'pi/2': np.pi / 2,
        'pi': np.pi, '3pi/2': 3 * np.pi / 2, '-3pi/2': -3 * np.pi / 2, '-pi': -np.pi, '-pi/2': - np.pi / 2}
    for theta, angle in thetas.items():
        gates.update({
            f'Rx({theta})': rx(angle),
            f'Rz({theta})': rz(angle)
        })

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
    result = np.eye(matrices[0].shape[0], dtype=np.complex128)
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
            f.write(results)
            f.flush()
            print(results)


def chunked(iterable, size):
    """Yield successive chunks of given size from an iterable (supports generators)."""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def _check_sequence_unpack(args):
    return _check_sequence(*args)


def _check_sequence(sequence, gate_dict, target_matrix, allow_global_phase):
    """Worker function to compute a single combination."""
    result = np.eye(target_matrix.shape[0], dtype=complex)
    for gate_name in sequence:
        result = result @ gate_dict[gate_name]
    if allow_global_phase:
        if __are_equal_up_to_global_phase(result, target_matrix):
            return sequence, result
    else:
        if np.allclose(result, target_matrix, atol=1e-8):
            return sequence, result
    return None
