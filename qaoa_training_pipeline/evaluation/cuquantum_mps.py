#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""cuQuantum MPS evaluator for QAOA.

This module provides an optional NVIDIA cuQuantum/cuTensorNet backend.

"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import cupy as cp
import numpy as np
from cuquantum.tensornet.experimental import MPSConfig, NetworkState, TNConfig
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator
from qaoa_training_pipeline.utils.graph_utils import (
    make_swap_strategy,
    operator_to_list_of_hyper_edges,
)

CUQUANTUM_OBSERVABLE_STRATEGIES = {"pauli_products"}


@dataclass(frozen=True)
class _DiagonalZTerm:
    """A real diagonal Z or ZZ cost term in qubit-index order."""

    qubits: tuple[int, ...]
    coefficient: float


class CuQuantumMPSEvaluator(BaseEvaluator):
    """Evaluate default QAOA energies with cuQuantum's MPS state simulator."""

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        max_bond_dim: int | None = 64,
        abs_cutoff: float | None = None,
        rel_cutoff: float | None = None,
        precision: str = "fp32",
        svd_algo: str | None = None,
        mpo_application: str | None = None,
        observable_strategy: str = "pauli_products",
        device_id: int | None = None,
        network_options: dict | None = None,
        release_workspace: bool = False,
        gauge_option: str | None = None,
        normalization: str | None = None,
        use_swap_strategy: bool | None = True,
        mode: str | None = "mps",
        backend: str = "gpu",
    ) -> None:
        """Initialize the evaluator without importing cuQuantum.

        Args:
            max_bond_dim: Maximum MPS bond dimension. This maps to cuTensorNet
                ``MPSConfig.max_extent``. ``None`` disables extent truncation.
            abs_cutoff: Absolute SVD truncation cutoff. ``None`` disables this cutoff.
            rel_cutoff: Relative SVD truncation cutoff. ``None`` disables this cutoff.
            precision: ``"fp64"`` maps to ``complex128`` and ``"fp32"`` maps to
                ``complex64``.
            svd_algo: cuTensorNet MPS SVD algorithm. Valid values are ``"gesvd"``,
                ``"gesvdj"``, ``"gesvdp"``, and ``"gesvdr"``. ``None`` leaves the
                package default unchanged.
            mpo_application: cuTensorNet MPS-MPO operation mode, either ``"exact"``,
                ``"approximate"``, or ``None`` to leave the package default.
            observable_strategy: ``"pauli_products"`` builds one cached cuQuantum
                ``NetworkOperator`` from identity-free Pauli product components. The
                option is kept explicit to leave room for a future cuQuantum-native
                compressed MPO strategy.
            device_id: Optional CUDA device id passed through ``NetworkOptions``.
            network_options: Optional dictionary passed as cuTensorNet network options.
                ``device_id`` overrides the same key if both are supplied.
            release_workspace: Whether to release cuTensorNet workspace after the
                expectation call.
            backend: desired backend to use. Defaults to GPU
        """

        super().__init__()

        if max_bond_dim is not None and max_bond_dim <= 0:
            raise ValueError("max_bond_dim must be positive or None.")
        if abs_cutoff is not None and abs_cutoff < 0.0:
            raise ValueError("abs_cutoff must be non-negative or None.")
        if rel_cutoff is not None and rel_cutoff < 0.0:
            raise ValueError("rel_cutoff must be non-negative or None.")
        if precision not in {"fp64", "fp32"}:
            raise ValueError("precision must be either 'fp64' or 'fp32'.")

        if svd_algo is not None:
            svd_algo = svd_algo.lower()
            if svd_algo not in {"gesvd", "gesvdj", "gesvdp", "gesvdr"}:
                raise ValueError("svd_algo must be one of gesvd, gesvdj, gesvdp, or gesvdr.")

        if mpo_application is not None:
            mpo_application = mpo_application.lower()
            if mpo_application not in {"exact", "approximate"}:
                raise ValueError("mpo_application must be 'exact', 'approximate', or None.")

        if observable_strategy not in CUQUANTUM_OBSERVABLE_STRATEGIES:
            raise ValueError(
                "observable_strategy must be one of " f"{sorted(CUQUANTUM_OBSERVABLE_STRATEGIES)}."
            )

        self._max_bond_dim = max_bond_dim
        self._abs_cutoff = abs_cutoff
        self._rel_cutoff = rel_cutoff
        self._precision = precision
        self._dtype = "complex64" if precision == "fp32" else "complex128"
        self._np_dtype = np.complex64 if precision == "fp32" else np.complex128
        self._svd_algo = svd_algo
        self._mpo_application = mpo_application
        self._observable_strategy = observable_strategy
        self._device_id = device_id
        self._network_options = dict(network_options or {})
        self._release_workspace = release_workspace
        self._terms = None
        self._pauli_terms = None
        self._use_swap_strategy = use_swap_strategy
        self._cuquantum_classes = None
        self._operator = None
        self._operator_cost_op: SparsePauliOp | None = None
        self._state = None

        if mode == "mps":
            self._config = MPSConfig(
                max_extent=max_bond_dim,
                abs_cutoff=abs_cutoff,
                rel_cutoff=rel_cutoff,
                # algorithm="gesvd",
                mpo_application=mpo_application,
                gauge_option=gauge_option,
                algorithm=svd_algo,
                normalization=normalization,
            )
        elif mode == "tn":
            self._config = TNConfig(num_hyper_samples=1)

        self._set_backend(backend=backend)

    # pylint: disable=too-many-positional-arguments,arguments-differ
    def evaluate(
        self,
        cost_op: SparsePauliOp,
        params: list[float],
        mixer: QuantumCircuit | None = None,
        initial_state: QuantumCircuit | None = None,
        ansatz_circuit: QuantumCircuit | SparsePauliOp | None = None,
    ) -> float:
        """Evaluate the default QAOA energy for the given diagonal cost operator."""

        self._terms_from_cost_operator(cost_op, params)
        self._pauli_strings_from_terms(cost_op.num_qubits, self._observable_terms)

        assert len(self._terms) == len(self._observable_terms)

        state = NetworkState(
            [2 for _ in range(cost_op.num_qubits)],
            config=self._config,
            dtype=self._dtype,
        )
        self._state = state
        self._apply_qaoa_state(self._state, cost_op.num_qubits, self._terms, params)
        expectation = self._state.compute_expectation(self._pauli_terms)
        self.free_state()

        return expectation

    def _apply_qaoa_state(
        self,
        state,
        n_qubits: int,
        terms: Sequence[_DiagonalZTerm],
        params: Sequence[float],
    ) -> None:
        """Build the QAOA state by applying gates directly to a NetworkState."""

        layer_count = len(params) // 2
        h_gate = self._h_gate()
        swap_gate = self._swap_gate()

        self._swap_layer_pairs = defaultdict(list)
        if self._use_swap_strategy:
            for term in terms:
                if len(term.qubits) == 2:
                    q0, q1 = sorted(term.qubits)
                    distance = self._swap_strategy.distance_matrix[q0, q1]
                    self._swap_layer_pairs[distance].append((min(q0, q1), max(q0, q1)))
        for qubit in range(n_qubits):
            state.apply_tensor_operator([qubit], h_gate, unitary=True, immutable=False)

        # plus_state = self._plus_state(n_qubits)
        # state.set_initial_mps(plus_state)

        # Apply QAOA layers
        rep = 1
        for layer in range(layer_count):
            gamma = params[layer_count + layer]
            # Apply each of the terms with naive SWAP strategy if there is no SWAP strategy usage
            if not self._use_swap_strategy:
                for term in terms:
                    if len(term.qubits) == 1:
                        theta = 2.0 * gamma * term.coefficient
                        state.apply_tensor_operator(
                            [term.qubits[0]],
                            self._rz_gate(theta),
                            unitary=True,
                        )
                    elif len(term.qubits) == 2:

                        theta = 2.0 * gamma * term.coefficient
                        q0, q1 = sorted(term.qubits)

                        # --- bring qubits together ---
                        # Move q1 left until it is next to q0
                        for k in range(q1 - 1, q0, -1):
                            state.apply_tensor_operator(
                                [k, k + 1],
                                self._swap_gate(),
                                unitary=True,
                            )
                        # # --- apply RZZ on adjacent qubits ---
                        state.apply_tensor_operator(
                            [q0, q0 + 1], self._rzz_gate(theta), unitary=True, immutable=False
                        )

                        # # --- undo swaps ---
                        for k in range(q0 + 1, q1):
                            state.apply_tensor_operator(
                                [k, k + 1],
                                self._swap_gate(),
                                unitary=True,
                            )

            else:
                # If we use SWAP strategy, apply the previously computed strategy
                layer_order = list(range(len(self._swap_strategy) + 1))
                if rep % 2 == 0:
                    layer_order = layer_order[::-1]

                for layer_idx in layer_order:
                    permutation = self._swap_strategy.inverse_composed_permutation(layer_idx)
                    # 1. Apply the gates.
                    for i, (node0, node1) in enumerate(self._swap_layer_pairs[layer_idx]):
                        theta = (
                            2.0
                            * gamma
                            * next(
                                term.coefficient for term in terms if term.qubits == (node0, node1)
                            )
                        )
                        positions = [
                            min(permutation.index(node0), permutation.index(node1)),
                            max(permutation.index(node0), permutation.index(node1)),
                        ]

                        state.apply_tensor_operator(
                            positions, self._rzz_gate(theta), unitary=True, immutable=False
                        )

                    if rep % 2 == 0:
                        swap_layer_idx = layer_idx - 1
                    else:
                        swap_layer_idx = layer_idx

                    # 2. Apply the SWAPs.
                    if 0 <= swap_layer_idx < len(self._swap_strategy):
                        for swap_pairs in self._swap_strategy.swap_layer(swap_layer_idx):
                            if swap_pairs[1] != swap_pairs[0] + 1:
                                raise ValueError("Inconsistency found in SWAP strategy")
                            state.apply_tensor_operator(
                                [swap_pairs[0], swap_pairs[1]],
                                swap_gate,
                                unitary=True,
                                immutable=False,
                            )
            rep += 1
            beta = params[layer]
            rx_gate = self._rx_gate(2.0 * beta)
            for qubit in range(n_qubits):
                state.apply_tensor_operator([qubit], rx_gate, unitary=True, immutable=False)

    def _terms_from_cost_operator(
        self, cost_op: SparsePauliOp, params
    ) -> tuple[list[_DiagonalZTerm], float]:
        """Extract supported diagonal terms and the identity offset from a cost operator."""

        if self._terms is None:
            terms = []
            identity_offset = 0.0

            for pauli_str, coefficient in cost_op.to_list():
                coefficient = complex(coefficient)
                if abs(coefficient.imag) > 1.0e-12:
                    raise NotImplementedError("Complex Hamiltonian coefficients are not supported.")
                if any(char not in {"I", "Z"} for char in pauli_str):
                    raise NotImplementedError(
                        "CuQuantumMPSEvaluator supports only diagonal I/Z cost operators."
                    )

                qubits = tuple(idx for idx, char in enumerate(pauli_str[::-1]) if char == "Z")
                if len(qubits) == 0:
                    identity_offset += float(coefficient.real)
                elif len(qubits) <= 2:
                    terms.append(_DiagonalZTerm(qubits=qubits, coefficient=float(coefficient.real)))
                else:
                    raise NotImplementedError(
                        "CuQuantumMPSEvaluator currently supports only one- and two-local Z terms."
                    )
            self._terms = terms
            terms_observable = terms
            if self._use_swap_strategy:
                edges = operator_to_list_of_hyper_edges(cost_op)
                self._swap_strategy = make_swap_strategy(
                    [tuple(val[0]) for val in edges],
                    cost_op.num_qubits,
                )

                # If we use a SWAP strategy and the QAOA depth is odd we must permute the cost op.
                if (len(params) // 2) % 2 == 1:
                    inv_perm = self._swap_strategy.inverse_composed_permutation(
                        len(self._swap_strategy)
                    )
                    permutation = [inv_perm.index(idx) for idx in range(len(inv_perm))]
                    cost_op = cost_op.apply_layout(permutation)
                    terms_observable = []
                    for pauli_str, coefficient in cost_op.to_list():
                        coefficient = complex(coefficient)
                        if abs(coefficient.imag) > 1.0e-12:
                            raise NotImplementedError(
                                "Complex Hamiltonian coefficients are not supported."
                            )
                        if any(char not in {"I", "Z"} for char in pauli_str):
                            raise NotImplementedError(
                                "CuQuantumMPSEvaluator supports only diagonal I/Z cost operators."
                            )

                        qubits = tuple(
                            idx for idx, char in enumerate(pauli_str[::-1]) if char == "Z"
                        )
                        if len(qubits) == 0:
                            identity_offset += float(coefficient.real)
                        elif len(qubits) <= 2:
                            terms_observable.append(
                                _DiagonalZTerm(qubits=qubits, coefficient=float(coefficient.real))
                            )
                        else:
                            raise NotImplementedError(
                                "CuQuantumMPSEvaluator currently supports only one- and two-local Z terms."
                            )
            self._observable_terms = terms_observable

    def _plus_state(self, n_qubits):
        xp = self._backend()
        dtp = xp.complex64 if self._dtype == "complex64" else xp.complex128
        plus = xp.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=dtp)
        mps = [plus.reshape(2, 1)]
        mps += [plus.reshape(1, 2, 1) for _ in range(n_qubits - 2)]
        mps += [plus.reshape(1, 2)]

        return mps

    def _h_gate(self):
        """Return the Hadamard gate."""
        xp = self._backend()
        dtp = xp.complex64 if self._dtype == "complex64" else xp.complex128
        return xp.array(
            [[1.0 / xp.sqrt(2.0), 1.0 / xp.sqrt(2.0)], [1.0 / xp.sqrt(2.0), -1.0 / xp.sqrt(2.0)]],
            dtype=dtp,
        )

    def _z_gate(self):
        """Return the Pauli-Z gate."""
        xp = self._backend()
        dtp = xp.complex64 if self._dtype == "complex64" else xp.complex128
        return xp.array([[1.0, 0.0], [0.0, -1.0]], dtype=dtp)

    def _rx_gate(self, theta: float):
        """Return the RX(theta) gate."""
        xp = self._backend()
        dtp = xp.complex64 if self._dtype == "complex64" else xp.complex128
        return xp.array(
            [
                [xp.cos(theta / 2.0), -1.0j * xp.sin(theta / 2.0)],
                [-1.0j * xp.sin(theta / 2.0), xp.cos(theta / 2.0)],
            ],
            dtype=dtp,
        )

    def _rz_gate(self, theta: float):
        """Return the RZ(theta) gate."""
        xp = self._backend()
        dtp = xp.complex64 if self._dtype == "complex64" else xp.complex128
        return xp.array(
            [
                [xp.exp(-0.5j * theta), 0.0],
                [0.0, xp.exp(0.5j * theta)],
            ],
            dtype=dtp,
        )

    def _rzz_gate(self, theta: float):
        """Return the RZZ(theta) gate in cuQuantum tensor-operator axis order."""
        xp = self._backend()
        dtp = xp.complex64 if self._dtype == "complex64" else xp.complex128
        # Diagonal entries
        diag = xp.array(
            [
                xp.exp(-0.5j * theta),
                xp.exp(0.5j * theta),
                xp.exp(0.5j * theta),
                xp.exp(-0.5j * theta),
            ],
            dtype=dtp,
        )

        # Build operator explicitly in (out0, out1, in0, in1)
        gate = xp.zeros((2, 2, 2, 2), dtype=dtp)

        for i in range(2):
            for j in range(2):
                idx = 2 * i + j
                gate[i, j, i, j] = diag[idx]

        return gate

    def _swap_gate(self):
        xp = self._backend()
        dtp = xp.complex64 if self._dtype == "complex64" else xp.complex128
        swap = xp.zeros((2, 2, 2, 2), dtype=dtp)
        for i in range(2):
            for j in range(2):
                swap[j, i, i, j] = 1.0
        return swap

    def _pauli_strings_from_terms(
        self,
        n_qubits: int,
        terms: Sequence[_DiagonalZTerm],
    ) -> dict[str, float]:
        """Build cuQuantum mode-order Pauli strings from diagonal Z terms."""
        if self._pauli_terms == None:
            xp = self._backend()
            pauli_strings = {}
            for term in terms:
                label = ["I"] * n_qubits
                for qubit in term.qubits:
                    label[qubit] = "Z"
                pauli_label = "".join(label)
                pauli_strings[pauli_label] = pauli_strings.get(pauli_label, 0.0) + xp.float64(
                    term.coefficient
                )
            self._pauli_terms = {
                pauli_label: coefficient
                for pauli_label, coefficient in pauli_strings.items()
                if abs(coefficient) > 1.0e-15
            }

    def _set_backend(self, backend: str):
        if backend == "gpu":
            self._xp = cp
        else:
            self._xp = np

    def _backend(self):
        """"""
        return self._xp

    def free_state(self):
        self._state.free()
