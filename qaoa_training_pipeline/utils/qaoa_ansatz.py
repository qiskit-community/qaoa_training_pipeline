#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Conventions consistent QAOA Ansatz."""


from qiskit.circuit.library import qaoa_ansatz as qiskit_qaoa_ansatz
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator


def qaoa_ansatz(
    cost_operator: BaseOperator,
    reps: int = 1,
    initial_state: QuantumCircuit | None = None,
    mixer_operator: BaseOperator | None = None,
    insert_barriers: bool = False,
    name: str = "QAOA",
    flatten: bool = True,
) -> QuantumCircuit:
    r"""A QAOA ansatz that follows the conventions in CONVENTIONS.md.

    Qiskit's ``qaoa_ansatz`` defaults to the mixer :math:`\sum_iX_i`. This function wraps it
    with the default mixer :math:`H_M=-\sum_iX_i`, whose ground state is the default initial
    state :math:`|+\rangle^{\otimes n}`. The mixer layer is thus made of :math:`R_X(-2\beta)`.

    Args:
        cost_operator: The operator representing the cost of the optimization problem, denoted as
            :math:`U(C, \gamma)` in [1].
        reps: The integer determining the depth of the circuit, called :math:`p` in [1].
        initial_state: An optional initial state to use, which defaults to a layer of
            Hadamard gates preparing the :math:`|+\rangle^{\otimes n}` state.
            If a custom mixer is chosen, this circuit should be set to prepare its ground state,
            to appropriately fulfill the annealing conditions.
        mixer_operator: An optional custom mixer, which defaults to global Pauli-:math:`-X`
            rotations. This is denoted as :math:`U(B, \beta)` in [1]. If this is set,
            the ``initial_state`` might also require modification.
        insert_barriers: Whether to insert barriers in-between the cost and mixer operators.
        name: The name of the circuit.
        flatten: If ``True``, a flat circuit is returned instead of nesting it inside multiple
            layers of gate objects. Setting this to ``False`` is significantly less performant,
            especially for parameter binding, but can be desirable for a cleaner visualization.
    """

    if mixer_operator is None:
        mixer_operator = default_mixer_operator(cost_operator.num_qubits)

    return qiskit_qaoa_ansatz(
        cost_operator=cost_operator,
        reps=reps,
        initial_state=initial_state,
        mixer_operator=mixer_operator,
        insert_barriers=insert_barriers,
        name=name,
        flatten=flatten,
    )


def default_mixer_operator(num_qubits: int) -> SparsePauliOp:
    r"""The default mixer :math:`H_M=-\sum_iX_i` of the conventions."""
    return SparsePauliOp.from_sparse_list([("X", [i], -1) for i in range(num_qubits)], num_qubits)
