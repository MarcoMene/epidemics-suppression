from typing import Sequence

from matplotlib import pyplot as plt

from bsp_epidemic_suppression_model.math_utilities.general_utilities import round2
from bsp_epidemic_suppression_model.old_stuff.epidemic_data import R0
from dev.old_stuff.algorithm.model_blocks import effectiveness_from_R


# TODO
def plot_time_evolution(
    t_in_days_sequence: Sequence[float],
    R: Sequence[float],
    FT_infty_sequence: Sequence[float],
    # step_data_list: List[StepData], plot_components: bool = True
) -> None:
    """
    Plots time evolution of epidemics-suppression: effective reproduction numbers and other KPIs.
    """
    fig = plt.figure(figsize=(10, 15))

    t_max = t_in_days_sequence[-1]

    R_last = R[-1]

    # R
    R_tplot = fig.add_subplot(211)
    R_tplot.set_xlabel("t (days)")
    R_tplot.set_ylabel("R_t")
    R_tplot.grid(True)
    R_tplot.set_xlim(0, t_max)
    R_tplot.set_ylim(0, R0)
    R_tplot.plot(
        t_in_days_sequence,
        R,
        color="black",
        label=f"R_t → {round2(R_last)}, Eff_t → {round2(effectiveness_from_R(R_last))}",
    ),
    # if plot_components:
    #     R_tplot.plot(
    #         [step_data.t for step_data in step_data_list],
    #         [step_data.Rapp for step_data in step_data_list],
    #         color="green",
    #         label=f"R_t app → {round2(Rapp_last)}, Eff_t app → {round2(effectiveness_from_R(Rapp_last))}",
    #     ),
    #     R_tplot.plot(
    #         [step_data.t for step_data in step_data_list],
    #         [step_data.Rnoapp for step_data in step_data_list],
    #         color="blue",
    #         label=f"R_t no app → {round2(Rnoapp_last)}, Eff_t no app → {round2(effectiveness_from_R(Rnoapp_last))}",
    #     ),
    R_tplot.legend()

    # Other metrics
    Pplot = fig.add_subplot(212)
    Pplot.set_xlabel("t (days)")
    Pplot.set_ylabel("Probability")
    Pplot.grid(True)
    Pplot.set_xlim(0, t_max)
    Pplot.set_ylim(0, 1)
    # if plot_components:
    #     Pplot.plot(
    #         [step_data.t for step_data in step_data_list],
    #         [step_data.papp for step_data in step_data_list],
    #         color="yellow",
    #         label="Prob. infected has the app",
    #     )
    #     Pplot.plot(
    #         [step_data.t for step_data in step_data_list],
    #         [step_data.tildepapp for step_data in step_data_list],
    #         color="red",
    #         label="Prob. source has the app",
    #     )
    Pplot.plot(
        t_in_days_sequence,
        FT_infty_sequence,
        color="black",
        label="Prob. that infected tests positive",
    )
    # if plot_components:
    #     Pplot.plot(
    #         [step_data.t for step_data in step_data_list],
    #         [step_data.FTapp_infty for step_data in step_data_list],
    #         color="green",
    #         label="Prob. that infected with app tests positive",
    #     )
    #     Pplot.plot(
    #         [step_data.t for step_data in step_data_list],
    #         [step_data.FTnoapp_infty for step_data in step_data_list],
    #         color="blue",
    #         label="Prob. that infected without app tests positive",
    #     )
    Pplot.legend()

    plt.show()
