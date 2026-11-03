#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The max-cut convention the inference models were trained under.

What the models accept as a cost operator, and how a caller is told when
theirs does not fit: :func:`validate_cost_op` rejects the operators that
cannot be a max-cut graph at all, and
:func:`warn_on_unexpected_sign_convention` flags the one mistake that is
detectable rather than fatal.
"""

from __future__ import annotations

import warnings

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def warn_on_unexpected_sign_convention(cost_op: SparsePauliOp, prefactor: float) -> None:
    """Warn when ``cost_op`` looks like it was built with the wrong pre-factor.

    ``prefactor`` is the convention the bundle declares in the setup file. It is
    required: this package holds no default, so a model trained under another
    convention is checked against its own rather than against max-cut's.

    Nothing in an operator records which pre-factor produced it, so the general
    case cannot be detected.

    The blind spot, by construction: a graph with weights of both signs has
    coefficients of both signs under either convention, so neither spelling of
    it warns. Normalization does not close that gap -- it divides out the
    magnitude of the pre-factor but not its sign.
    """
    coefficients = [
        float(np.real(term.coeffs[0])) for term in cost_op if int(sum(term.paulis[0].z)) > 1
    ]
    # Under a negative prefactor, positive edge weights give negative
    # coefficients -- so all-positive ones mean the opposite sign was used.
    wrong_sign = (
        all(c > 0.0 for c in coefficients) if prefactor < 0 else all(c < 0.0 for c in coefficients)
    )
    if coefficients and wrong_sign:
        warnings.warn(
            "The quadratic coefficients of cost_op all have the sign opposite to this "
            f"model's convention, H = {prefactor} * sum_e w_e Z_i Z_j, i.e. "
            f"graph_to_operator(graph, pre_factor={prefactor}). The operator was most "
            "likely built with the pre_factor=1.0 default; it still yields finite "
            f"angles, but the predicted gammas are wrong. Pass pre_factor={prefactor}, "
            "or ignore this warning if your edge weights really are negative.",
            UserWarning,
            stacklevel=2,
        )


def validate_cost_op(cost_op: SparsePauliOp) -> None:
    """Reject operators the models cannot be given as a max-cut graph.

    Each rejection is something that would otherwise produce finite, plausible
    and wrong angles rather than an error:

    * **Non-real coefficients.** The graph conversion takes ``np.real`` of every
      coefficient, so a non-Hermitian operator would be read as a graph with
      the imaginary parts discarded -- a purely imaginary coefficient becomes a
      zero-weight edge.
    * **One-local ``Z`` terms.** The networks take edge weights only; there is
      no input for a local field. ``operator_to_graph`` would turn each such
      term into a self-loop, which the symmetrization then double-counts into
      the node degrees.
    * **Terms above second order.** No graph represents them.

    Raises:
        ValueError: On any of the above.
    """
    imaginary = [
        (str(term.paulis[0]), complex(term.coeffs[0]))
        for term in cost_op
        if abs(complex(term.coeffs[0]).imag) > 1e-12
    ]
    if imaginary:
        raise ValueError(
            f"cost_op has non-real coefficients: {imaginary}. The inference models take a "
            "real-weighted max-cut graph; a complex coefficient would be silently truncated "
            "to its real part."
        )

    one_local, higher_order = [], []
    for term in cost_op:
        order = int(sum(term.paulis[0].z))
        if order == 1:
            one_local.append(str(term.paulis[0]))
        elif order > 2:
            higher_order.append(str(term.paulis[0]))

    if one_local:
        raise ValueError(
            f"cost_op has one-local Z terms: {one_local}. The inference models predict angles "
            "for a max-cut graph and have no input for a local field, so such a term cannot be "
            "represented: it would become a self-loop and distort the node degrees the model "
            "reads. Drop the linear terms, or use a trainer that supports them."
        )
    if higher_order:
        raise ValueError(
            f"cost_op has terms above second order: {higher_order}. The inference models take "
            "a quadratic operator, H = prefactor * sum_e w_e Z_i Z_j."
        )
