#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""MPS-based QAOA optimizer."""

from typing import Dict, List, Optional

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.utils.graph_utils import (
    circuit_to_graph,
    make_swap_strategy,
    operator_to_list_of_hyperedges,
)
from qaoa_training_pipeline.utils.tns_utils.qaoa_circuit_mps import (
    QAOACircuitMPSRepresentation,
    QAOACircuitVidalRepresentation,
)
from qaoa_training_pipeline.utils.tns_utils.qaoa_cost_function import QAOACostFunction


class MPSEvaluator(BaseEvaluator):
    r"""Matrix Product State-based evaluator of QAOA circuits

    This class encodes the QAOA circuit as a matrix product state (MPS),
    and the cost function as a matrix product operator (MPO). It evaluates
    the cost function associated with the QAOA problem through tensor network contraction.
    The accuracy of the simulator is tuned by two parameters, i.e.,

     - the bond dimension, which defines the maximum dimension of the tensors
       composing the MPS
     - the truncation threshold, which defines the fidelity to which we are
       approximating the circuit

    More details can be found in Ann. Phys., 326, 96 (2011).
    The simulator becomes exact if the truncation threshold is set
    to 0, and if the max dimension is > than :math: `2^{N/2}`, :math: `N` being
    the number of qubits of the circuit.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        use_vidal_form: bool = False,
        threshold_circuit: Optional[float] = None,
        bond_dim_circuit: Optional[int] = None,
        threshold_mpo: Optional[float] = None,
        bond_dim_mpo: Optional[int] = None,
        use_swap_strategy: bool = False,
        store_schmidt_values: bool = False,
    ):
        """Initializes the MPS evaluator.

        We have to set the truncation threshold and the bond dimension for both the
        circuit and the operator simulation. However, the MPO is usually so small
        (at least for sparse quadratic operators) that the MPO compression is negligible.

        Args:
            use_vidal_form (bool): Whether to express the Matrix Product State in the Vidal
                form (see doc of `circuit_mps_vidal.py` for more detail). If true, uses
                the vidal form and, otherwise, uses the canonical gauge form. Note that
                the choice of the gauge affects only the efficiency of the simulator, but
                not the accuracy. This means that, for a given truncation threshold and maximum
                bond dimension, both simulators yield the same accuracy.
            threshold_circuit (Optional[float], optional): Truncation threshold for the circuit
                simulation. Defaults to None. If None is given the MPS is equivalent to a
                 very slow statevector simulator. The threshold should correlate with the fidelity.
            bond_dim_circuit (Optional[int], optional): Maximum bond dimension for the circuit
                simulation. Defaults to None.
            threshold_mpo (Optional[float], optional): Truncation threshold for constructing the
                MPO representation of the cost function. Defaults to None.
            bond_dim_mpo (Optional[int], optional): Maximum bond dimension for the MPO representation
                of the cost function. Defaults to None.
            use_swap_strategy (bool): Use a line swap strategy to apply the gates on the tensor
                network. Defaults to False.
            store_schmidt_values (bool): If true then the Schmidt values are stored in a JSON
                serializable format.
        """
        self._threshold_circuit = threshold_circuit
        self._max_bond_circuit = bond_dim_circuit
        self._threshold_cost = threshold_mpo
        self._max_bond_cost = bond_dim_mpo
        self._use_swap_strategy = use_swap_strategy
        self._store_schmidt = store_schmidt_values
        self._schmidt_values = None

        # This is initialized when `evaluate` is called for the first time.
        self._cost_op = None

        # Swap strategy. Attached to the class for traceability.
        self._swap_strategy = None

        # Type variable remembering which type to use for the MPS representation of the circuit
        self._circuit_type = (
            QAOACircuitVidalRepresentation if use_vidal_form else QAOACircuitMPSRepresentation
        )

        self._results_last_iteration = {}

    # pylint: disable=arguments-differ, pylint: disable=too-many-positional-arguments
    def evaluate(
        self,
        cost_op: SparsePauliOp,
        params: List[float],
        mixer: Optional[QuantumCircuit] = None,
        initial_state: Optional[QuantumCircuit] = None,
        ansatz_circuit: Optional[QuantumCircuit] = None,
    ) -> float:
        r"""Evaluates the energy.

        Args:
            cost_op (SparsePauliOp): The cost operator that defines :math:`H_C`.
                Only quadratic cost functions are currently supported.
            params (List[float]): The parameters for QAOA. The length of this list will
                determine the depth of the QAOA.
                The params are given in the order
                :math:`[\beta_0, \beta_1, \ldots, \gamma_0, \gamma_1, \ldots]`.
            mixer (Optional[QuantumCircuit], optional):
                Circuit to be used as the mixer part of the QAOA circuit. Defaults to None.
                If equal to None, just uses a layer of Rx gates.
            initial_state (Optional[QuantumCircuit], optional): Circuit used for the state
                initialization. Defaults to None.
            ansatz_circuit (Optional[QuantumCircuit], optional):
                Ansatz circuit for the QAOA. Defaults to None. If equal to None, just uses
                the Trotterized propagator associated with the underlying Hamiltonian. When
                specified, this ansatz is given as a network of Rzz gates only with a single
                parameter which is gamma. This will then be used to construct a cost operator
                layer in the evaluator.
            use_vidal_form (bool): if true, uses the Vidal canonization for representing
                the MPS.

        Raises:
            NotImplementedError: if a user-defined mixer is provided as input.
            NotImplementedError: if a user-defined initial_state is provided as input.
            NotImplementedError: if a user-defined ansatz is provided as input.
            KeyError: if the size of the `params` vector is an odd number.

        Returns:
            float: Energy calculated for the `params` parameters.
        """
        if len(params) % 2 != 0:
            raise KeyError("Number of parameters must be an even integer")

        # Must come before the cost_op is permuted by the swap strategy.
        if ansatz_circuit is None:
            edges = operator_to_list_of_hyperedges(cost_op)
        else:
            qc_graph = circuit_to_graph(ansatz_circuit)
            edges = [[[u, v], w.get("weight", 1)] for u, v, w in qc_graph.edges(data=True)]

        # If there are any hyperedges, then the swap strategy cannot be implemented.
        if any(len(i_edge[0]) > 2 for i_edge in edges) and self._use_swap_strategy:
            raise NotImplementedError("Swap strategy not supported for HOBO problems")

        # Make the swap strategy.
        if self._use_swap_strategy:
            self._swap_strategy = make_swap_strategy(
                [val[0] for val in edges],
                cost_op.num_qubits,
            )

            # If we use a SWAP strategy and the QAOA depth is odd we must permute the cost op.
            if (len(params) // 2) % 2 == 1:
                inv_perm = self._swap_strategy.inverse_composed_permutation(
                    len(self._swap_strategy)
                )
                permutation = [inv_perm.index(idx) for idx in range(len(inv_perm))]
                cost_op = cost_op.apply_layout(permutation)

        if self._cost_op is None or not cost_op.equiv(self._cost_op.sparse_pauli):
            self._cost_op = QAOACostFunction(cost_op, self._threshold_cost, self._max_bond_cost)

        # Construct the circuit
        beta_parameters = params[: len(params) // 2]
        gamma_parameters = params[len(params) // 2 :]

        circuit = self._circuit_type.construct_from_list_of_edges(
            edges,
            truncation_threshold=self._threshold_circuit,
            max_bond_dim=self._max_bond_circuit,
            swap_strategy=self._swap_strategy,
            mixer=mixer,
            initial_state=initial_state,
        )

        circuit.apply_qaoa_layer(beta_parameters, gamma_parameters)

        cost_function_estimate = circuit.compute_cost_function(self._cost_op)

        # Updates important results
        underlying_mps = circuit.get_underlying_tn()
        self._results_last_iteration.update(
            {
                "circuit_bond_dimension": [
                    underlying_mps.ind_sizes()[i] for i in underlying_mps.inner_inds()
                ]
            }
        )

        if self._store_schmidt:
            self._schmidt_values = [
                circuit.get_schmidt_values(i) for i in range(circuit.n_qubits - 1)
            ]
            self._results_last_iteration.update({"schmidt_values": self._schmidt_values})

        return np.real(cost_function_estimate)

    def get_results_from_last_iteration(self) -> Dict:
        """Gets important results obtained at the last optimization iteration.

        For now, the only result that is stored are the bond dimensions of the MPS.

        This is relevant because, when one runs a matrix product state-based simulation
        by fixing the truncation threshold to be used in the singular value decomposition,
        it is important to know the maximum bond dimension of the resulting matrix product
        state to extrapolate, for instance, the bond dimension in the zero truncation
        limit.

        Returns:
            Dict: dictionary with relevant data of the simulator obtained at the
                last iteration
        """
        return self._results_last_iteration

    @property
    def schmidt_values(self) -> List[List[float]]:
        """Get the Schmidt values for a given bond of the MPS.

        Returns:
            List[List[float]]: list of the Schmidt values, calculated
                for each MPS bond.
        """

        return self._schmidt_values

    def reset_cost_function(self, new_cost_op: SparsePauliOp) -> None:
        """Resets the cost operator to be used.

        Args:
            new_cost_op (SparsePauliOp): new `SparsePauliOp` object to be used
        """
        self._cost_op = QAOACostFunction(new_cost_op)

    @property
    def swap_strategy(self):
        """The swap strategy used by the evaluator."""
        return self._swap_strategy

    @classmethod
    def from_config(cls, config: dict) -> "MPSEvaluator":
        """Create a MPS evaluator from a config."""
        return cls(**config)

    def to_config(self) -> Dict:
        """Json serializable config to keep track of how results are generated."""
        config = super().to_config()

        config["threshold_circuit"] = self._threshold_circuit
        config["bond_dim_circuit"] = self._max_bond_circuit
        config["threshold_cost"] = self._threshold_cost
        config["bond_dim_mpo"] = self._max_bond_cost

        return config

    @classmethod
    def parse_init_kwargs(cls, init_kwargs: Optional[str] = None) -> dict:
        """Parse initialization kwargs.

        This allows us to override any defaults set in the methods files.
        The input string should be formatted as the variables `use_vidal_form` (bool),
        `threshold_circuit` (float), `bond_dim_circuit` (int), `threshold_mpo` (float),
        `bond_dim_mpo` (int), `use_swap_strategy` (bool) and `store_schmidt_values` (bool),
        all separated by an underscrore and in that order.
        """
        if init_kwargs is None:
            return dict()

        init_args = init_kwargs.split("_")

        threshold_circuit = None if init_args[1].lower() == "none" else float(init_args[1])
        bond_dim_circuit = None if init_args[2].lower() == "none" else int(init_args[2])
        threshold_mpo = None if init_args[3].lower() == "none" else float(init_args[3])
        bond_dim_mpo = None if init_args[4].lower() == "none" else int(init_args[4])
        use_swap_strategy = None if init_args[5].lower() == "none" else bool(init_args[5])
        store_schmidt_values = None if init_args[6].lower() == "none" else bool(init_args[6])

        return {
            "use_vidal_form": init_args[0].lower() in ["1", "true"],
            "threshold_circuit": threshold_circuit,
            "bond_dim_circuit": bond_dim_circuit,
            "threshold_mpo": threshold_mpo,
            "bond_dim_mpo": bond_dim_mpo,
            "use_swap_strategy": use_swap_strategy,
            "store_schmidt_values": store_schmidt_values,
        }
