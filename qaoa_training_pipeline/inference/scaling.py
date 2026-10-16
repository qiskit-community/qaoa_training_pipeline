#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The scale-invariance round trip: normalize the operator, un-rescale the gammas.

The inference models are scale-invariant, and this module is where that is
arranged. A prediction passes through three steps -- scale, predict, unscale --
and the two scaling ones live here so they can be read, and reasoned about,
together:

1. :func:`normalize_cost_op` divides the cost operator by
   :func:`rescaling_factor`, so the network always sees the same normalized
   problem whatever the operator's scale.
2. The network predicts for that normalized problem (the predictor's job), and
   its output is denormalized by the bundle's own output scale
   (:func:`denormalize_qaoa_params_np`).
3. :func:`undo_gamma_rescale` divides the gammas -- and only the gammas -- by
   the same factor, which undoes step 1 for the operator the caller passed.

:class:`~qaoa_training_pipeline.inference.AIInference` calls the three in that
order, and
:meth:`~qaoa_training_pipeline.inference.onnx_predictor.OnnxQAOAPredictor.predict`
does the same for a caller who wants one call.

Nothing here needs ``onnxruntime`` or a model: it is numpy and Qiskit only, so
the scaling can be imported, tested and reasoned about without a bundle.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def rescaling_factor(cost_op: SparsePauliOp) -> float:
    """Return the QAOA cost-operator rescaling factor (RMS of per-order weights).

    The one quantity the round trip computes: the same factor divides the
    operator in step 1 and the gammas in step 3.

    It is homogeneous of degree one -- ``rescaling_factor(c * H) ==
    c * rescaling_factor(H)`` for ``c > 0`` -- which is what makes the
    normalized operator, and hence the network's raw output, the same at every
    scale. It also makes normalizing twice a no-op, since the second factor
    comes out as 1.0.

    For a degenerate operator with no non-identity terms or all-zero
    coefficients the RMS is 0; it falls back to ``1.0``, so normalization and
    denormalization become no-ops instead of dividing by zero.
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
    """Divide ``cost_op`` by its :func:`rescaling_factor`. Step one of the chain.

    Returns ``(normalized_cost_op, rescale_a)``. The factor comes back with the
    operator because the later steps need it: the features report it, and
    :func:`undo_gamma_rescale` divides the gammas by it.
    """
    rescale_a = rescaling_factor(cost_op)
    return cost_op / rescale_a, rescale_a


def denormalize_qaoa_params_np(
    qaoa_params_norm: np.ndarray, scale: float = math.pi / 2
) -> np.ndarray:
    """Denormalize predicted QAOA params by ``scale`` (default pi/2).

    Part of step two, not of the operator scaling: this undoes the output
    normalization the model was trained with, which the bundle's config
    declares as ``output_scale``.
    """
    return qaoa_params_norm * scale


def undo_gamma_rescale(angles: Sequence[float], p: int, rescale_a: float) -> list[float]:
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
