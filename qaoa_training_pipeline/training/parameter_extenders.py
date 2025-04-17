#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This file is a collection of callables that extend QAOA parameters."""

from typing import List
import numpy as np


def interpolate(optimized_params: List[float]) -> List[float]:
    """Update parameters from p to p+1 by interpolation.

    This function implements the linear interpolation as presented by Zhou et al.
    in PRX 10, 021067 (2020).
    """
    reps = len(optimized_params) // 2
    betas = [0] + optimized_params[0:reps] + [0]
    gammas = [0] + optimized_params[reps:] + [0]

    new_gammas = []
    for idx in range(1, reps + 2):
        new_gamma = (idx - 1) / reps * gammas[idx - 1] + (reps - idx + 1) / reps * gammas[idx]
        new_gammas.append(new_gamma)

    new_betas = []
    for idx in range(1, reps + 2):
        new_beta = (idx - 1) / reps * betas[idx - 1] + (reps - idx + 1) / reps * betas[idx]
        new_betas.append(new_beta)

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


PARAMETEREXTENDERS = {
    "interpolate": interpolate,
    "extend": extend,
}
