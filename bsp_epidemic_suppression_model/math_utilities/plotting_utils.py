from typing import List, Optional

import matplotlib.pyplot as plt

from bsp_epidemic_suppression_model.math_utilities.functions_utils import RealRange


def plot_functions(
    fs: list,
    real_range: RealRange,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
):
    """
    Util to plot a list of functions in a range, with a step.
    """
    if labels is None:
        labels = [str(i) for i in range(len(fs))]
    for i, f in enumerate(fs):
        plt.plot(
            real_range.x_values, [f(x) for x in real_range.x_values], label=labels[i]
        )
    plt.legend()
    plt.title(title)
    plt.show()
