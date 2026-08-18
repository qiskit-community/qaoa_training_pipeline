"""Fixed, reproducible cost operators for inference checks.

Kept deterministic (no RNG) so the golden baselines in
``test/inference/baselines/`` are stable across machines. Each entry is a small
QAOA MaxCut-style cost operator over a named graph topology.
"""

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp


def _zz(num_qubits: int, i: int, j: int, weight: float = 1.0) -> tuple[str, float]:
    """A single weighted ZZ term on qubits (i, j) of an n-qubit register."""
    label = ["I"] * num_qubits
    label[i] = "Z"
    label[j] = "Z"
    return "".join(label), weight


def _ring(n: int, weight: float = 1.0) -> SparsePauliOp:
    return SparsePauliOp.from_list([_zz(n, k, (k + 1) % n, weight) for k in range(n)])


def _line(n: int, weight: float = 1.0) -> SparsePauliOp:
    return SparsePauliOp.from_list([_zz(n, k, k + 1, weight) for k in range(n - 1)])


def _complete(n: int, weight: float = 1.0) -> SparsePauliOp:
    return SparsePauliOp.from_list(
        [_zz(n, i, j, weight) for i in range(n) for j in range(i + 1, n)]
    )


def _weighted_ring(n: int) -> SparsePauliOp:
    # Deterministic, non-uniform weights so weight-sensitive models are exercised.
    return SparsePauliOp.from_list([_zz(n, k, (k + 1) % n, 0.5 + 0.25 * k) for k in range(n)])


# Ordered so indices are stable; used as the benchmark/baseline case list.
BENCH_OPS: dict[str, SparsePauliOp] = {
    "triangle_3": _complete(3),
    "line_4": _line(4),
    "ring_4": _ring(4),
    "complete_4": _complete(4),
    "ring_6": _ring(6),
    "line_8": _line(8),
    "weighted_ring_6": _weighted_ring(6),
    "complete_5": _complete(5),
}
