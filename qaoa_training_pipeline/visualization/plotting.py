# 
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""These functions help to plot the data."""


from typing import Dict, Optional
import numpy as np

import matplotlib.pyplot as plt


def plot_cdf(
    func_counts: Dict[float, float],
    axis: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
    **plot_args,
):
    """Plot the CDF of the given counts.

    Args:
        func_counts: A dictionary mapping function values to the count or probability
            of sampling that value.
        axis: The axis object on which to plot. If None is given then we create one.
        fig: The figure instance. If no axis are given then we create one.
        plot_args: Key word arguments that are given to axis.plot().

    Returns:
        An axis instance and figure handle. These are the inputs when given.
    """

    if axis is None:
        fig, axis = plt.subplots(1, 1)

    x_vals = sorted(func_counts.keys())
    y_vals = np.cumsum([func_counts[x] for x in x_vals])
    axis.plot(x_vals, y_vals, **plot_args)

    return axis, fig
