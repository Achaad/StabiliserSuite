from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, CouplingError


def cnot(qc: QuantumCircuit, control: int, target: int, coupling_map: CouplingMap = None) -> None:
    """
    Applies a CNOT gate to the given quantum circuit.
    Attempts to find a bridge between the control and target qubits if the qubits cannot interact directly.

    :param qc: Circuit to apply the CNOT gate to
    :param control: Control qubit
    :param target: Target qubit
    :param coupling_map: Coupling map of the quantum circuit
    :return: None
    """
    # If no coupling map is provided, apply the CNOT gate directly
    if coupling_map is None:
        qc.cx(control, target)
        return

    # Check if the control and target qubits can be connected
    try:
        distance = coupling_map.distance(control, target)
    except CouplingError:
        raise ValueError('Control and target qubits are not connected in the coupling map.')

    # If the qubits are directly connected, apply the CNOT gate
    if distance == 1:
        qc.cx(control, target)
        return

    # If the qubits are not directly connected, find a bridge
    path = coupling_map.shortest_undirected_path(control, target)

    # Apply the CNOT gate using the bridge to every path triplet
    for i in range(len(path) - 2):
        intermediate_control = path[i]
        bridge = path[i + 1]
        intermediate_target = path[i + 2]

        # Apply a CNOT gate between the intermediate control and intermediate target using the bridge
        qc.cx(intermediate_control, bridge)
        qc.cx(bridge, intermediate_target)
        qc.cx(intermediate_control, bridge)
        qc.cx(bridge, intermediate_target)
