from typing import Callable, List, Optional, Sequence

import matplotlib.pyplot as plt

from bsp_epidemic_suppression_model.math_utilities.config import UNITS_IN_ONE_DAY
from bsp_epidemic_suppression_model.math_utilities.discrete_distributions_utils import (
    DiscreteDistributionOnNonNegatives,
)
from bsp_epidemic_suppression_model.math_utilities.general_utilities import RealRange


def plot_functions(
    fs: Sequence[Callable[[float], float]],
    real_range: RealRange,
    custom_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
):
    """
    Util to plot a list of functions with given range and step.
    """
    if custom_labels is None:
        labels = [str(i) for i in range(len(fs))]
    else:
        labels = custom_labels
    for i, f in enumerate(fs):
        plt.plot(
            real_range.x_values, [f(x) for x in real_range.x_values], label=labels[i]
        )
    if len(fs) > 1 or custom_labels is not None:
        plt.legend()
    plt.title(title)
    plt.show()


def plot_discrete_distributions(
    ds: Sequence[DiscreteDistributionOnNonNegatives],
    custom_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    tau_max: Optional[float] = None,
    plot_cdfs: bool = False,
):
    if tau_max is None:
        tau_max = max(d.tau_max for d in ds)
    tau_range = range(0, tau_max + 1)
    if custom_labels is None:
        labels = [str(i) for i in range(len(fs))]
    else:
        labels = custom_labels
    for i, d in enumerate(ds):
        plt.plot(
            [x / UNITS_IN_ONE_DAY for x in tau_range],
            [d.pmf(tau) for tau in tau_range],
            label=labels[i],
        )
    if len(ds) > 1 or custom_labels is not None:
        plt.legend()
    plt.title(title)
    plt.show()
