from __future__ import annotations
import inspect
import itertools
import math
from datetime import datetime, timezone
from typing import Tuple, Sequence, Dict, List

import numpy as np
from numba import njit
from numba.core.types import complex128
from numpy import ndarray
from qiskit import QiskitError
from qiskit.circuit import Gate, Instruction, CircuitError
from qiskit.circuit.library import iSwapGate, XGate, YGate, ZGate, IGate, HGate, SGate, CZGate, SwapGate, \
    standard_gates, CXGate
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

    # Add special two-qubit gates
    tensor_gates.update({
        CXGate().name: CXGate().to_matrix(),
        iSwapGate().name: iSwapGate().to_matrix(),
        CZGate().name: CZGate().to_matrix(),
        SwapGate().name: SGate().to_matrix()
    })

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


def find_matching_combinations(gate_dict, target_matrix, output=None, max_depth=3, allow_global_phase=True) -> list[
    tuple[str, np.ndarray]]:
    results = []
    try:
        gate_names = list(gate_dict.keys())

        for depth in range(1, max_depth + 1):
            print(f"Checking depth {depth} ({len(gate_names) ** depth} combinations)...")
            for sequence in tqdm(itertools.product(gate_names, repeat=depth), total=len(gate_names) ** depth):
                result = np.eye(target_matrix.shape[0], dtype=complex)
                for gate_name in sequence:
                    if result.shape == gate_dict[gate_name].shape:
                        result = result @ gate_dict[gate_name]
                if (allow_global_phase and __are_equal_up_to_global_phase(result, target_matrix)) or \
                        (not allow_global_phase and np.allclose(result, target_matrix, atol=1e-8)):
                    results.append((sequence, result))

            solutions = [a for a, _ in results]
            print(f"Current solutions at depth {depth}: {solutions}", flush=True)
    finally:
        if output is not None:
            with open(output, 'a', encoding='utf-8') as f:
                now = datetime.now(timezone.utc).astimezone()
                now_text = now.strftime('%Y-%m-%d %H:%M:%S %Z%z')
                f.write(f'\n{now_text} - {max_depth} depth search\n')
                if len(results) == 0:
                    f.write('NO SOLUTIONS FOUND\n')
                    f.flush()
                else:
                    f.write('COMPLETE RESULTS:')
                    f.write(', '.join(f'({a}, {b})' for a, b in results))
                    f.write('\n')
                    f.flush()
    return results


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


def get_gates(control: int = 0, target: int = 0, num_control_qubits: int = 1) -> dict[str, Gate | Instruction]:
    """
    Retrieve a dictionary of quantum gates with optional control and target qubits.

    This function iterates through the standard gates available in Qiskit,
    attempts to instantiate them, and generates parameterized versions of
    gates if applicable. The resulting dictionary includes gate names as keys
    and their corresponding gate instances as values.

    Args:
        control (int, optional): The control qubit index for gates requiring control qubits. Defaults to 0.
        target (int, optional): The target qubit index for gates requiring target qubits. Defaults to 0.
        num_control_qubits (int, optional): The number of control qubits for gates requiring this parameter. Defaults to 1.

    Returns:
        dict[str, Gate | Instruction]: A dictionary where keys are gate names (optionally parameterized)
                                       and values are the corresponding gate instances.
    """
    gate_classes = {}
    for name, gate_class in vars(standard_gates).items():
        # Check if the item is a class and a subclass of Gate
        if inspect.isclass(gate_class) and issubclass(gate_class, Gate):
            # Attempt to instantiate the gate class
            gates = __try_instantiate_gate(gate_class, control=control, target=target,
                                           num_control_qubits=num_control_qubits)
            if gates:
                # Generate parameterized gate names with angles in steps of π/4
                thetas = [f"{i}pi/4" if i != 0 else "0" for i in range(-6, 7)]
                gate_classes.update(
                    # Add parameterized gates if multiple instances are generated
                    {f"{name}({theta})": gate for theta, gate in zip(thetas, gates)}
                    if len(gates) > 1 else {name: gates[0]}  # Add single instance if only one gate is generated
                )
    return gate_classes


def get_non_clifford_gates(control: int = 0, target: int = 1, num_control_qubits: int = 1) -> dict:
    """
    Retrieve a dictionary of non-Clifford quantum gates.

    This function filters out Clifford gates from the set of all available gates
    and returns only the non-Clifford gates. It uses the `get_gates` function to
    generate all gates and the `is_clifford_gate` function to identify Clifford gates.

    Args:
        control (int, optional): The control qubit index for gates requiring control qubits. Defaults to 0.
        target (int, optional): The target qubit index for gates requiring target qubits. Defaults to 0.
        num_control_qubits (int, optional): The number of control qubits for gates requiring this parameter. Defaults to 1.

    Returns:
        dict: A dictionary where keys are gate names and values are the corresponding
              non-Clifford gate instances.
    """
    all_gate_classes = get_gates(control=control, target=target, num_control_qubits=num_control_qubits)
    gates = {name: gate_class for name, gate_class in all_gate_classes.items() if not is_clifford_gate(gate_class)}
    return gates


def get_clifford_gates(control: int = 0, target: int = 1, num_control_qubits: int = 1) -> dict:
    """
    Retrieve a dictionary of non-Clifford quantum gates.

    This function filters out non-Clifford gates from the set of all available gates
    and returns only the Clifford gates. It uses the `get_gates` function to
    generate all gates and the `is_clifford_gate` function to identify Clifford gates.

    Args:
        control (int, optional): The control qubit index for gates requiring control qubits. Defaults to 0.
        target (int, optional): The target qubit index for gates requiring target qubits. Defaults to 0.
        num_control_qubits (int, optional): The number of control qubits for gates requiring this parameter. Defaults to 1.

    Returns:
        dict: A dictionary where keys are gate names and values are the corresponding
              Clifford gate instances.
    """
    all_gate_classes = get_gates(control=control, target=target, num_control_qubits=num_control_qubits)
    gates = {name: gate_class for name, gate_class in all_gate_classes.items() if is_clifford_gate(gate_class)}
    return gates


def find_non_clifford_transformations(gate: Gate, max_depth: int = 3,
                                      allow_global_phase: bool = True) -> list[tuple[Sequence[str], ndarray]]:
    substitutions = []
    matrices = {}
    if gate.num_qubits == 1:
        all_gates = get_non_clifford_gates()
        matrices = {name: gate.to_matrix() for name, gate in all_gates.items() if gate.num_qubits == 1}
    else:
        all_gates = get_non_clifford_gates()
        matrices = __generate_multi_qubit_gates(all_gates, gate.num_qubits)
    substitutions = mitm_find_transformations(matrices, gate.to_matrix(), max_depth=max_depth,
                                              allow_global_phase=allow_global_phase)
    return substitutions


def find_clifford_transformations(gate: Gate, max_depth: int = 3,
                                  allow_global_phase: bool = True) -> list[tuple[Sequence[str], ndarray]]:
    substitutions = []
    matrices = {}
    if gate.num_qubits == 1:
        all_gates = get_clifford_gates()
        matrices = {name: gate.to_matrix() for name, gate in all_gates.items() if gate.num_qubits == 1}
    else:
        all_gates = get_clifford_gates()
        matrices = __generate_multi_qubit_gates(all_gates, gate.num_qubits)
    substitutions = mitm_find_transformations(matrices, gate.to_matrix(), max_depth=max_depth,
                                              allow_global_phase=allow_global_phase)
    return substitutions


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
        except (TypeError, CircuitError):
            return []

    return []


def __generate_multi_qubit_gates_old(gates_dict: dict[str, Gate], target_qubits: int = 2) -> (
        dict)[str, np.ndarray]:
    """
    Generate multi-qubit gate matrices from a dictionary of gates.

    This function creates a dictionary of multi-qubit gates by combining gates using the Kronecker product. It
    ensures that the resulting gates match the specified target qubit dimension.

    Args:
        gates_dict (dict[str, Gate]): A dictionary where keys are gate names and values
                                      are gate objects.
        target_qubits (int, optional): The number of qubits for the target gates. Defaults to 2.

    Returns:
        dict[str, np.ndarray]: A dictionary where keys are gate names (or combinations of gate names)
                               and values are the corresponding multi-qubit gate matrices.
    """
    multi_qubit_gates = {}
    dim = target_qubits ** 2

    gates = gates_dict.copy()
    gates[IGate().name] = IGate().to_matrix()

    for name1, gate1 in gates.items():
        matrix1 = gate1.to_matrix()
        if matrix1.shape[0] == dim:
            multi_qubit_gates[name1] = matrix1
            continue

        for name2, gate2 in gates.items():
            matrix2 = gate2.to_matrix()
            # Check if the combined dimensions match the target dimension
            if matrix1.shape[0] * matrix2.shape[0] == dim:
                multi_qubit_gates[f"{name1}⊗{name2}"] = np.kron(matrix1, matrix2)

    return multi_qubit_gates


def __generate_multi_qubit_gates(gates_dict: dict[str, Gate],
                                 target_qubits: int = 2) -> dict[str, np.ndarray]:
    """
    Generate arbitrary multi-qubit gates (up to target_qubits) by taking
    tensor products of the supplied gates.

    Parameters
    ----------
    gates_dict : dict[str, Gate]
        Dictionary mapping gate names to Qiskit Gate objects.  The routine
        assumes each gate's ``num_qubits`` attribute is correct.
    target_qubits : int
        The desired width (number of qubits) of every generated matrix.

    Returns
    -------
    dict[str, np.ndarray]
        Keys are Kronecker-product strings such as "H⊗X⊗S", values are the
        corresponding unitary matrices of dimension ``2**target_qubits``.
    """
    # ————————————————————————————————————————————————————————————————
    # 1)  Build a catalogue of (name, matrix, arity) including the identity
    # ————————————————————————————————————————————————————————————————
    gate_specs: list[tuple[str, np.ndarray, int]] = []

    for name, gate in gates_dict.items():
        mat = gate.to_matrix()
        gate_specs.append((name, mat, gate.num_qubits))

    # ensure the identity is present for padding
    i_name, i_mat = IGate().name, IGate().to_matrix()
    gate_specs.append((i_name, i_mat, 1))

    # Map arity → list of specs of that size (helps pruning)
    by_arity: dict[int, list[tuple[str, np.ndarray, int]]] = {}
    for spec in gate_specs:
        by_arity.setdefault(spec[2], []).append(spec)

    # ————————————————————————————————————————————————————————————————
    # 2)  DFS to build all tensor products that sum to target_qubits
    # ————————————————————————————————————————————————————————————————
    multi_qubit_gates: dict[str, np.ndarray] = {}

    def dfs(current_name: str,
            current_mat: np.ndarray | None,
            qubits_left: int) -> None:
        """Recursively build Kronecker products totalling target_qubits."""
        if qubits_left == 0:
            multi_qubit_gates[current_name] = current_mat
            return

        # iterate over every gate whose arity ≤ qubits_left
        for arity in range(1, qubits_left + 1):
            for name, mat, _ in by_arity.get(arity, []):
                new_name = name if current_name == "" else f"{current_name}⊗{name}"
                new_mat = mat if current_mat is None else np.kron(current_mat, mat)
                dfs(new_name, new_mat, qubits_left - arity)

    dfs("", None, target_qubits)
    return multi_qubit_gates


def _canonicalise(mat: np.ndarray, *, allow_phase: bool) -> np.ndarray:
    """Return matrix divided by global phase (if allowed)."""
    if not allow_phase:
        return mat
    # find first non-zero entry
    idx = np.flatnonzero(mat)
    if idx.size == 0:
        return mat
    r, c = divmod(idx[0], mat.shape[0])
    phase = mat[r, c] / abs(mat[r, c])
    return mat / phase


def _hash_key(mat: np.ndarray, *, allow_phase: bool) -> Tuple[float, ...]:
    """Convert (canonical) matrix to a hashable tuple with rounding."""
    canon = _canonicalise(mat, allow_phase=allow_phase)
    rounded = np.round(canon.real, 12) + 1j * np.round(canon.imag, 12)
    return tuple(rounded.ravel())


def _matrix_equal(a: np.ndarray, b: np.ndarray,
                  *, allow_phase: bool) -> bool:
    """Return True if a == e^{iθ} b (or exactly equal if phase not allowed)."""
    if allow_phase:
        return np.allclose(_canonicalise(a, allow_phase=True),
                           _canonicalise(b, allow_phase=True),
                           atol=1e-8)
    return np.allclose(a, b, atol=1e-8)


def _enumerate_layers(gate_dict: Dict[str, np.ndarray],
                      depth: int,
                      allow_phase: bool
                      ) -> Dict[Tuple[float, ...], Tuple[Sequence[str], np.ndarray]]:
    """
    Enumerate all products of exactly `depth` gates.

    Returns a dict {hash_key → (sequence, matrix)} keeping the lexicographically
    smallest sequence per hash key.
    """
    if depth == 0:
        ident = np.eye(next(iter(gate_dict.values())).shape[0], dtype=complex)
        key = _hash_key(ident, allow_phase=allow_phase)
        return {key: (tuple(), ident)}

    layer: Dict[Tuple[float, ...], Tuple[Sequence[str], np.ndarray]] = {}
    gate_items = list(gate_dict.items())

    # start from depth-1 layer
    prev = _enumerate_layers(gate_dict, depth - 1, allow_phase)

    for seq, mat in prev.values():
        for gname, gmat in gate_items:
            new_seq = seq + (gname,)
            new_mat = gmat @ mat  # right-multiplication
            key = _hash_key(new_mat, allow_phase=allow_phase)
            # keep lexicographically minimal sequence for determinism
            if key not in layer or new_seq < layer[key][0]:
                layer[key] = (new_seq, new_mat)
    return layer


def mitm_find_transformations(gate_dict: Dict[str, np.ndarray],
                              target_matrix: np.ndarray,
                              *,
                              max_depth: int = 3,
                              allow_global_phase: bool = True
                              ) -> List[Tuple[Sequence[str], np.ndarray]]:
    """
    Meet-in-the-middle search for gate sequences whose product equals
    `target_matrix` (up to optional global phase).

    Parameters
    ----------
    gate_dict
        Mapping {gate_name → unitary matrix}.
    target_matrix
        Target unitary.
    max_depth
        Maximum total length of a candidate sequence.
    allow_global_phase
        If True the match is checked up to a global phase.

    Returns
    -------
    List of (sequence, product_matrix) for every solution found.
    """
    results: List[Tuple[Sequence[str], np.ndarray]] = []
    dim = target_matrix.shape[0]
    ident = np.eye(dim, dtype=complex)
    gates = gate_dict.copy()
    gates[IGate().name] = ident

    # split depth (ceil/ﬂoor)
    left_depth = math.ceil(max_depth / 2)
    right_depth = max_depth - left_depth

    # pre-compute all left products up to left_depth
    left_tables = []
    for d in range(left_depth + 1):
        left_tables.append(_enumerate_layers(gates, d,
                                             allow_phase=allow_global_phase))

    # enumerate right side from depth 0 … right_depth
    for d_r in range(right_depth + 1):
        right_table = _enumerate_layers(gates, d_r,
                                        allow_phase=allow_global_phase)

        for r_key, (r_seq, r_mat) in right_table.items():
            # Needed left matrix to satisfy L · R = U_target
            needed = target_matrix @ np.linalg.inv(r_mat)
            needed_key = _hash_key(needed, allow_phase=allow_global_phase)

            # check every possible left depth up to remaining budget
            max_left_index = min(left_depth, max_depth - d_r)
            for d_l in range(max_left_index, -1, -1):
                if needed_key in left_tables[d_l]:
                    l_seq, l_mat = left_tables[d_l][needed_key]
                    # sanity check
                    prod = r_mat @ l_mat
                    if _matrix_equal(prod, target_matrix,
                                     allow_phase=allow_global_phase):
                        results.append((l_seq + r_seq, prod))

    return results
