from typing import List, Optional

import matplotlib.pyplot as plt

from bsp_epidemic_suppression_model.utilities.functions_utils import RealRange

from bsp_epidemic_suppression_model.utilities.model import R0

from bsp_epidemic_suppression_model.algorithm.time_evolution_with_severity import (
    StepData,
)


def plot_functions(fs: list, real_range: RealRange, labels: Optional[List[str]]=None):
    """
    Util to plot a list of functions in a range, with a step.
    """
    if labels is None:
        labels = [str(i) for i in range(len(fs))]
    for i, f in enumerate(fs):
        plt.plot(real_range.x_values, [f(x) for x in real_range.x_values], label=labels[i])
    plt.legend()
    plt.show()


def plot_time_evolution(step_data_list: List[StepData]):
    fig = plt.figure(figsize=(10, 15))
    # fig.suptitle(f"{scenario}")

    t_max = step_data_list[-1].t

    # R
    R_tplot = fig.add_subplot(211)
    R_tplot.set_xlabel("t [days]")
    R_tplot.set_ylabel("R_t")
    R_tplot.grid(True)
    R_tplot.set_xlim(0, t_max)
    R_tplot.set_ylim(0, R0)
    R_tplot.plot(
        [step_data.t for step_data in step_data_list],
        [step_data.R for step_data in step_data_list],
        color="black",
        label="R",
    ),
    R_tplot.plot(
        [step_data.t for step_data in step_data_list],
        [step_data.Rapp for step_data in step_data_list],
        color="green",
        label="R app",
    ),
    R_tplot.plot(
        [step_data.t for step_data in step_data_list],
        [step_data.Rnoapp for step_data in step_data_list],
        color="red",
        label="R no app",
    ),
    R_tplot.legend()

    # # Effectiveness
    # E_tplot = fig.add_subplot(211)
    # E_tplot.set_xlabel("t [days]")
    # E_tplot.set_ylabel("E_t")
    # E_tplot.grid(True)
    # E_tplot.set_xlim(0, t_max)
    # E_tplot.set_ylim(0, 1)
    # E_tplot.plot(
    #     [step_data.t for step_data in step_data_list],
    #     [effectiveness_from_R(R=step_data.R_t) for step_data in step_data_list],
    #     color="black",
    #     label="R - Eff",
    # ),
    # E_tplot.plot(
    #     [step_data.t for step_data in step_data_list],
    #     [effectiveness_from_R(R=step_data.Rapp_t) for step_data in step_data_list],
    #     color="green",
    #     label="R app - Eff",
    # ),
    # E_tplot.plot(
    #     [step_data.t for step_data in step_data_list],
    #     [effectiveness_from_R(R=step_data.Rnoapp_t) for step_data in step_data_list],
    #     color="red",
    #     label="R no app - Eff",
    # ),
    # E_tplot.legend()

    # Other metrics
    Pplot = fig.add_subplot(212)
    Pplot.set_xlabel("t [days]")
    Pplot.set_ylabel("Probability")
    Pplot.grid(True)
    Pplot.set_xlim(0, t_max)
    Pplot.set_ylim(0, 1)
    Pplot.plot(
        [step_data.t for step_data in step_data_list],
        [step_data.papp for step_data in step_data_list],
        color="yellow",
        label="Prob. infected has the app",
    )
    Pplot.plot(
        [step_data.t for step_data in step_data_list],
        [step_data.tildepapp for step_data in step_data_list],
        color="yellow",
        label="Prob. source has the app",
    )
    Pplot.plot(
        [step_data.t for step_data in step_data_list],
        [step_data.FT_infty for step_data in step_data_list],
        color="black",
        label="Prob. that infected tests positive",
    )
    Pplot.plot(
        [step_data.t for step_data in step_data_list],
        [step_data.FTapp_infty for step_data in step_data_list],
        color="green",
        label="Prob. that infected with app tests positive",
    )
    Pplot.plot(
        [step_data.t for step_data in step_data_list],
        [step_data.FTnoapp_infty for step_data in step_data_list],
        color="red",
        label="Prob. that infected without app tests positive",
    )
    Pplot.legend()

    plt.show()
