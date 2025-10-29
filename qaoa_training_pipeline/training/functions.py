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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class BaseAnglesFunction(ABC):
    """A base class to define the interface of QAOA angle functions."""

    @abstractmethod
    def __call__(self, x: list) -> list:
        """Compute the QAOA angles from the optimization variables x."""

    def to_config(self) -> dict:
        """Creates a serializeable dictionary of the class."""
        return {"function_name": self.__class__.__name__}

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> None:
        """Initialize the function from a config."""


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
    r"""Computes the QAOA angles from the optimization variables x using a Fourier series.

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
            x: The coefficients in the Fourier basis. The first half store the coefficients for
                beta and the second half store the coefficients for gamma.
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
            x: The coefficients in the Fourier basis. The first half store the coefficients for
                beta and the second half store the coefficients for gamma.
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


class PCAFunction(BaseAnglesFunction):
    """Performs a PCA transform of the QAOA angles.

    The PCA QAOA angle function is inspired from O. Parry and P. McMinn, "QAOA-PCA: Enhancing
    efficiency in the quantum approximate optimization algorithm via principal component
    analysis", arXiv:2504.16755.
    """

    def __init__(self, num_components: int):
        """Initialize a PCA angles function for QAOA.

        Args:
            num_components: The number of PCA components.
        """
        self._num_components = num_components
        self._scaler = StandardScaler()
        self._pca = PCA(self._num_components)
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return true if self has been fitted to data."""
        return self._is_fitted

    def fit(self, data):
        """Fit the PCA function."""
        data_ = self._scaler.fit_transform(data)
        self._pca.fit(data_)
        self._is_fitted = True

    def __call__(self, x: list) -> list:
        """Compute the QAOA angles from the principal components."""
        if not self._is_fitted:
            raise ValueError(f"Fit {self.__class__.__name__} to compute the QAOA angles.")

        # Reconstruct standardized data from PCA
        angles_scaled = self._pca.inverse_transform([x])

        # Convert back to original scale
        qaoa_angles = self._scaler.inverse_transform(angles_scaled)

        return qaoa_angles[0]

    def transform(self, x: list) -> list:
        """Convert QAOA angles to their principle components."""
        if not self._is_fitted:
            raise ValueError(f"Fit {self.__class__.__name__} to compute the principal components.")

        x_scaled = self._scaler.transform([x])
        return self._pca.transform(x_scaled)[0]

    def to_config(self) -> dict:
        """Creates a serializeable dictionary of the class."""
        config = super().to_config()
        config["num_components"] = self._num_components

        if self._is_fitted:
            config["scaler"] = {
                "mean": self._scaler.mean_.tolist(),
                "scale": self._scaler.scale_.tolist(),
                "var": self._scaler.var_.tolist(),
            }

            config["pca"] = {
                "components": self._pca.components_.tolist(),
                "mean": self._pca.mean_.tolist(),
                "explained_variance": self._pca.explained_variance_.tolist(),
                "explained_variance_ratio": self._pca.explained_variance_ratio_.tolist(),
            }

        return config

    @classmethod
    def from_config(cls, config: dict) -> None:
        """Initialize the Fourier function."""

        foo = cls(config["num_components"])

        if "scaler" in config:
            scaler_params = config["scaler"]
            foo._scaler.mean_ = np.array(scaler_params["mean"])
            foo._scaler.scale_ = np.array(scaler_params["scale"])
            foo._scaler.var_ = np.array(scaler_params["var"])
            foo._scaler.n_features_in_ = len(foo._scaler.mean_)

        if "pca" in config:
            pca_params = config["pca"]
            foo._pca.components_ = np.array(pca_params["components"])
            foo._pca.mean_ = np.array(pca_params["mean"])
            foo._pca.explained_variance_ = np.array(pca_params["explained_variance"])
            foo._pca.explained_variance_ratio_ = np.array(pca_params["explained_variance_ratio"])
            foo._pca.n_features_in_ = len(foo._pca.mean_)

        return foo


FUNCTIONS = {
    "IdentityFunction": IdentityFunction,
    "FourierFunction": FourierFunction,
    "PCAFunction": PCAFunction,
}
