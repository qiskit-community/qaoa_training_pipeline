#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""These functions help to work with quantum circuits."""

from typing import List

from qiskit import QuantumCircuit
from qiskit.circuit import Gate


def split_circuit(circuit: QuantumCircuit) -> List[QuantumCircuit]:
    """Split the quantum circuit into single-qubits.

    This utility function allows us to split a circuit with `n` qubits and only
    single-qubit rotations into `n` circuits with a single qubit. This is used
    to split up user provided mizer operators with single-qubit rotations.

    Args:
        circuit: A quantum circuit with only single-qubit rotations.

    Raises:
        ValueError: if the circuit has instructions that act on more than one qubit.
    """
    inst_map = {qubit: [] for qubit in circuit.qubits}

    for inst in circuit.data:
        if len(inst.qubits) > 1:
            raise ValueError("Only single-qubit instructions are supported.")

        if not isinstance(inst.operation, Gate):
            raise ValueError(f"Circuit contains operations that are not instances of {Gate}.")

        inst_map[inst.qubits[0]].append(inst.operation)

    circuits = []
    for operations in inst_map.values():
        sq_circ = QuantumCircuit(1)
        for op in operations:
            sq_circ.append(op, [0])

        circuits.append(sq_circ)

    return circuits
