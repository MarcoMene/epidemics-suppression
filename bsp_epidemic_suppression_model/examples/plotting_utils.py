from typing import Callable, List, Optional, Sequence

import matplotlib.pyplot as plt

from bsp_epidemic_suppression_model.math_utilities.config import UNITS_IN_ONE_DAY
from bsp_epidemic_suppression_model.math_utilities.discrete_distributions_utils import (
    DiscreteDistributionOnNonNegatives,
)
from bsp_epidemic_suppression_model.math_utilities.general_utilities import (
    RealRange,
    effectiveness,
)
from bsp_epidemic_suppression_model.model_utilities.epidemic_data import R0


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
        labels = [str(i) for i in range(len(ds))]
    else:
        labels = custom_labels
    for i, d in enumerate(ds):
        values = (
            [d.cdf(tau) for tau in tau_range]
            if plot_cdfs
            else [d.pmf(tau) for tau in tau_range]
        )
        plt.plot(
            [x / UNITS_IN_ONE_DAY for x in tau_range], values, label=labels[i],
        )
    if len(ds) > 1 or custom_labels is not None:
        plt.legend()
    plt.title(title)
    plt.show()


def plot_homogeneous_time_evolution(
    t_in_days_sequence: Sequence[float],
    R_ts: Sequence[float],
    R0: float,
    FT_infty_sequence: Sequence[float],
    nu_ts: Sequence[float],
    nu0_ts: Sequence[float],
) -> None:
    """
    Plots the time evolution of R_t, FT_t(∞), nu_t in the homogeneous scenario.
    """
    fig = plt.figure(figsize=(10, 15))

    t_max = t_in_days_sequence[-1]

    R_last = R_ts[-1]

    # R
    R_tplot = fig.add_subplot(311)
    R_tplot.set_xlabel("t (days)")
    R_tplot.set_ylabel("R_t")
    R_tplot.grid(True)
    R_tplot.set_xlim(0, t_max)
    R_tplot.set_ylim(0, R0)
    R_tplot.plot(
        t_in_days_sequence,
        R_ts,
        color="black",
        label=f"R_t → {round(R_last, 2)}, Eff_t → {round(effectiveness(R_last, R0), 2)}",
    ),
    R_tplot.legend()

    # FT
    Pplot = fig.add_subplot(312)
    Pplot.set_xlabel("t (days)")
    Pplot.set_ylabel("Probability")
    Pplot.grid(True)
    Pplot.set_xlim(0, t_max)
    Pplot.set_ylim(0, 1)
    Pplot.plot(
        t_in_days_sequence,
        FT_infty_sequence,
        color="black",
        label="Prob. that infected tests positive",
    )

    Pplot.legend()

    # nu
    nuplot = fig.add_subplot(313)
    nuplot.set_xlabel("t (days)")
    nuplot.set_ylabel("Probability")
    nuplot.grid(True)
    nuplot.set_xlim(0, t_max)
    nuplot.plot(
        t_in_days_sequence, nu_ts, color="black", label="Number of infected",
    )
    nuplot.plot(
        t_in_days_sequence,
        nu0_ts,
        color="gray",
        label="Number of infected without measures",
    )

    nuplot.legend()

    plt.show()
