#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The two scaling steps that make the inference models scale-invariant.

:func:`normalize_cost_op` divides the operator before the network sees it, and
:func:`undo_cost_op_normalization` divides the gammas by the same factor afterwards.
:meth:`~qaoa_training_pipeline.inference.AIInference.provide_params` runs both,
around the network step, and is the only place that does.

Needs no model and no ``onnxruntime``: numpy and Qiskit only.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def rescaling_factor(cost_op: SparsePauliOp) -> float:
    """Return the QAOA cost-operator rescaling factor (RMS of per-order weights).

    Homogeneous of degree one, which is what makes the normalized operator --
    and hence the network's output -- the same at every scale. A degenerate
    operator (no non-identity terms, or all-zero coefficients) gives 0 and
    falls back to ``1.0`` rather than dividing by zero.
    """
    terms = defaultdict(list)

    for term in cost_op:
        order = sum(term.paulis[0].z)
        terms[order].append(np.real(term.coeffs[0]) ** 2)

    factor = 0.0
    for squared_weights in terms.values():
        factor += sum(squared_weights) / len(squared_weights)

    factor = float(np.sqrt(factor))
    return factor if factor > 0 else 1.0


def normalize_cost_op(cost_op: SparsePauliOp) -> tuple[SparsePauliOp, float]:
    """Divide ``cost_op`` by its :func:`rescaling_factor`.

    Returns ``(normalized_cost_op, rescale_a)``; the factor comes back because
    the later steps need it.
    """
    rescale_a = rescaling_factor(cost_op)
    return cost_op / rescale_a, rescale_a


def undo_cost_op_normalization(angles: Sequence[float], p: int, rescale_a: float) -> list[float]:
    """Divide the gammas of ``angles`` by ``rescale_a``. Step three of the chain.

    Only the gammas move: with ``U = prod_k exp(-i beta_k H_B) exp(-i gamma_k
    H_C)``, sending ``H_C -> c H_C`` preserves the state only if
    ``gamma -> gamma / c``, while beta must not change.

    Args:
        angles: ``p`` betas followed by ``p`` gammas, as the network predicted
            them for the normalized operator.
        p: The QAOA depth, i.e. where the gammas start.
        rescale_a: The factor the operator was normalized by in step 1.
    """
    divisor = float(rescale_a)
    values = [float(value) for value in angles]
    return values[:p] + [value / divisor for value in values[p:]]
