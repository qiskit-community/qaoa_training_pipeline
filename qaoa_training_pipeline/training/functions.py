#
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions for angle trainers."""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np


class BaseAnglesFunction(ABC):
    """A base class to define the interface of QAOA angle functions."""

    @abstractmethod
    def __call__(self, x: list) -> list:
        """Compute the QAOA angles from the optimization variables x."""

    def to_config(self) -> dict:
        """Creates a serializeable dictionary of the class."""
        return {"function_name": self.__class__.__name__}


class IdentityFunction(BaseAnglesFunction):
    """Identity function."""

    def __call__(self, x: list) -> list:
        """Identity function."""
        return x

    # pylint: disable=unused-argument
    @classmethod
    def from_config(cls, config: dict) -> None:
        """Initialize the Identity function."""
        return cls()


class FourierFunction(BaseAnglesFunction):
    """Computes the QAOA angles from the optimization variables x using a Fourier series.

    This is the function that is implemented in Zhou et al. PRX 10, 021067 (2020). It computes
    the beta and gamma angles as follows for a depth-`p` QAOA circuit

    .. math::
        \beta_i = \sum_{k=1}^{q} u_k \sin[(k-1/2)(i-1/2)\pi/p]
        \gamma_i = \sum_{k=1}^{q} v_k \cos[(k-1/2)(i-1/2)\pi/p]
    """

    def __init__(self, depth: int) -> None:
        """Initialize the Fourier function."""
        self._depth = depth

    def __call__(self, x: list) -> list:
        """Compute beta and gamma angles from the optimization variables x.

        We assume that the first half of `x` is for `beta` and the second half is for `gamma`.
        Furthermore, this function currently assumes that the QAOA depth is given by the length
        of `x` divided by 2. Therefore, dimensionality reduction is not implemented.
        """
        n_coeffs = len(x) // 2

        betas, gammas = [], []
        for i_idx in range(self._depth):
            beta_i = 0

            # Note the +0.5 which accounts for the indices starting at 0.
            for k_idx, val in enumerate(x[:n_coeffs]):
                beta_i += val * np.cos((k_idx + 0.5) * (i_idx + 0.5) * np.pi / self._depth)

            gamma_i = 0
            for k_idx, val in enumerate(x[n_coeffs:]):
                gamma_i += val * np.sin((k_idx + 0.5) * (i_idx + 0.5) * np.pi / self._depth)

            betas.append(beta_i)
            gammas.append(gamma_i)

        return betas + gammas

    def to_config(self) -> dict:
        """Creates a serializeable dictionary of the class."""
        return {"function_name": self.__class__.__name__}

    @classmethod
    def from_config(cls, config: dict) -> None:
        """Initialize the Fourier function."""
        return cls(config["depth"])

    def plot_angles(
        self, x: list, axis: Optional[plt.Axes] = None, plot_args: Optional[Dict] = None
    ) -> plt.Axes:
        """Plot the QAOA angles.

        Args:
            x: The coefficients in of the Fourrier basis.
            axis: The axis on which to plot.
            plot_args: Additional arguments to pass to the plot. These are split into
                two sub-dictionaries with keys `beta` and `gamma`.
        """
        axis = axis or plt.gca()

        if plot_args is None:
            plot_args = {"beta": {}, "gamma": {}}

        if "ls" not in plot_args["beta"]:
            plot_args["beta"]["ls"] = "--"

        qaoa_angles = self(x)

        xvals = list(range(1, self._depth + 1))
        axis.plot(xvals, qaoa_angles[: self._depth], label=r"$\beta$", **plot_args["beta"])
        axis.plot(xvals, qaoa_angles[self._depth :], label=r"$\gamma$", **plot_args["gamma"])
        axis.legend()

        return axis

    def plot_basis(
        self, x: list, axis: Optional[plt.Axes] = None, plot_args: Optional[Dict] = None
    ) -> plt.Axes:
        """Plot the Fourier basis functions.

        Args:
            x: The coefficients in of the Fourrier basis.
            axis: The axis on which to plot.
            plot_args: Additional arguments to pass to the plot. These are split into
                two sub-dictionaries with keys `beta` and `gamma`.
        """
        axis = axis or plt.gca()

        n_coeffs = len(x) // 2

        ivals = np.linspace(0, self._depth, 50)

        if plot_args is None:
            plot_args = {"beta": {}, "gamma": {}}

        if "ls" not in plot_args["beta"]:
            plot_args["beta"]["ls"] = "--"

        for k_idx in range(n_coeffs):
            coeff = (k_idx + 0.5) * np.pi / self._depth

            lbl = r"$\beta$ basis" if k_idx == 0 else None

            axis.plot(
                ivals,
                [x[k_idx] * np.cos(coeff * (ival + 0.5)) for ival in ivals],
                label=lbl,
                **plot_args["beta"],
            )

            lbl = r"$\gamma$ basis" if k_idx == 0 else None

            axis.plot(
                ivals,
                [x[k_idx + n_coeffs] * np.sin(coeff * (ival + 0.5)) for ival in ivals],
                label=lbl,
                **plot_args["gamma"],
            )

        axis.legend()

        return axis


FUNCTIONS = {
    "IdentityFunction": IdentityFunction,
    "FourierFunction": FourierFunction,
}
