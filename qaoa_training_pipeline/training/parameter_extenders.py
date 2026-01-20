#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This file is a collection of callables that extend QAOA parameters."""

from typing import Iterable, List

import numpy as np


def interpolate(optimized_params: Iterable[float]) -> list[float]:
    """Update parameters from depth p to p+1 via linear interpolation.

    This function implements the parameter interpolation scheme introduced by
    Zhou et al., PRX 10, 021067 (2020).

    Given a parameter vector (gamma or beta) of length `reps`, assumed to be
    defined at integer positions 1..reps, the method constructs a new vector
    of length `reps + 1` whose entries are evenly spaced over the same interval
    [1, reps]. Each new entry is obtained by linear interpolation between
    adjacent parameters of the original vector.
    """
    optimized_params = np.asarray(optimized_params, dtype=float)
    reps = optimized_params.size // 2

    betas_vals = optimized_params[:reps]  # length reps
    gammas_vals = optimized_params[reps:]  # length reps

    # Interior knots and values
    xp = np.arange(1, len(betas_vals))  # 1, 2, ..., reps
    # Query points: 1 .. reps with reps+1 evenly spaced points
    xq = np.linspace(1.0, float(reps), reps + 1)

    new_betas = np.interp(xq, xp, betas_vals).tolist()
    new_gammas = np.interp(xq, xp, gammas_vals).tolist()
    return new_betas + new_gammas


def extend(optimized_params: List[float]) -> List[float]:
    """Simply extend the vector of parameters by one.

    This function implements the parameter extension presented by Lee et al. in the paper
    `Parameters fixing strategy for quantum approximate optimization algorithm` published
    in 2021 IEEE Int. Conf. on QCE.
    """
    reps = len(optimized_params) // 2
    betas = optimized_params[0:reps] + [np.random.rand()]
    gammas = optimized_params[reps:] + [np.random.rand()]

    return betas + gammas


def trivial_extend(optimized_params: List[float]) -> List[float]:
    """Simply extend the vector of parameters by one by adding a zero.

    This function is used, for example, to add an additional Fourier mode to optimize.
    See, for example, Zhou et al. in PRX 10, 021067 (2020).
    """
    reps = len(optimized_params) // 2
    betas = optimized_params[0:reps] + [0]
    gammas = optimized_params[reps:] + [0]

    return betas + gammas


PARAMETEREXTENDERS = {
    "interpolate": interpolate,
    "extend": extend,
    "trivial_extend": trivial_extend,
}
