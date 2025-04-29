# Define Pauli and standard single-qubit gates
import itertools
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit
from qiskit.circuit import Gate
from qiskit.circuit.library import iSwapGate
from qiskit.quantum_info import random_clifford
from tqdm import tqdm

# Define basic gates
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)


def get_clifford_gates() -> list[Gate]:
    """
    Returns a list of Clifford gates.

    :return: List of Clifford gates.
    """

    for i in range(24):
        random_clifford()

# Rotation gates (parameterized)
def Rx(theta):
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)]
    ], dtype=complex)


def Rz(theta):
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)


def __generate_two_qubit_gates(gates_dict):
    """
    Generates all 2-qubit gate matrices as tensor products from a given gate dictionary.

    :param gates_dict: dict with names as keys and 2x2 numpy arrays as values
    :return: dict of tensor products: keys as "A⊗B", values as 4x4 numpy arrays
    """
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
            f'Rx({theta})': Rx(angle),
            f'Rz({theta})': Rz(angle)
        })

    return __generate_two_qubit_gates(gates)


def __are_equal_up_to_global_phase(U1, U2, atol=1e-8):
    """Check if two unitary matrices are equal up to a global phase."""
    inner = np.vdot(U1.flatten(), U2.flatten())
    if np.abs(inner) < atol:
        return False
    phase = inner / np.abs(inner)
    return np.allclose(U1, U2 * phase, atol=atol)


@njit
def __are_equal_up_to_global_phase_numba(U1, U2, atol=1e-8):
    # TODO: create unit tests for that
    # TODO: verify that this computation is correct, maybe there is need for complex conjugate instead of U2,
    #  be careful of complex phase
    """JIT-compiled check for global phase equality between two unitary matrices."""
    dot = np.vdot(U1.flatten(), U2.flatten())
    norm = np.abs(dot)
    if norm < 1e-12:
        return False
    phase = dot / norm
    diff = U1 - phase * U2
    return np.all(np.abs(diff) < atol)


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
                total = len(gate_names) ** depth  # total combinations
                print(f"Checking depth {depth} ({total} combinations)...")
                total = len(gate_names) ** depth

                # with Pool(processes=cpu_count()) as pool:
                #     for chunk in chunked(tasks, 100000):
                #         args = [(seq, gate_dict, target_matrix, allow_global_phase) for seq in chunk]
                #         for res in tqdm(pool.imap_unordered(_check_sequence_unpack, args), total=len(chunk)):
                #             if res is not None:
                #                 results.append(res)

                for sequence in tqdm(itertools.product(gate_names, repeat=depth), total=total):
                    result = np.eye(target_matrix.shape[0], dtype=complex)
                    for gate_name in sequence:
                        result = result @ gate_dict[gate_name]
                    if allow_global_phase:
                        if __are_equal_up_to_global_phase_numba(result, target_matrix):
                            results.append((sequence, result))
                    else:
                        if np.allclose(result, target_matrix, atol=1e-8):
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
    return results


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
