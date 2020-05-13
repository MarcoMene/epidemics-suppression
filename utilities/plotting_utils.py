import matplotlib.pyplot as plt

from utilities.model import R0
from utilities.utils import RealRange, round2
from utilities.model import R0, effectiveness_from_R


def plot_functions(fs: list, real_range: RealRange):
    """
    Util to plot a list of functions in a range, with a step.
    """
    for i, f in enumerate(fs):
        plt.plot(real_range.x_values, [f(x) for x in real_range.x_values], label=str(i))
    plt.legend()


def plot_time_evolution(scenario, time_evolution_result):
    fig = plt.figure(figsize=(10, 15))
    fig.suptitle(f"{scenario}")

    R_tplot = fig.add_subplot(211)
    R_tplot.set_xlabel('t [days]')
    R_tplot.set_ylabel('R_t')
    R_tplot.grid(True)
    R_tplot.set_xlim(0, time_evolution_result.last_t)
    R_tplot.set_ylim(0, R0)
    R_tplot.plot(time_evolution_result.t_list, time_evolution_result.R_list,
                 color="black", label=f"R - Eff. {round2(effectiveness_from_R(time_evolution_result.last_R))}")
    R_tplot.plot(time_evolution_result.t_list, time_evolution_result.Rapp_list,
                 color="green", label=f"R app - Eff. {round2(effectiveness_from_R(time_evolution_result.last_Rapp))}")
    R_tplot.plot(time_evolution_result.t_list, time_evolution_result.Rnoapp_list,
                 color="red",
                 label=f"R no app - Eff. {round2(effectiveness_from_R(time_evolution_result.last_Rnoapp))}")
    R_tplot.legend()

    Pplot = fig.add_subplot(212)
    Pplot.set_xlabel('t [days]')
    Pplot.set_ylabel('P_t')
    Pplot.grid(True)
    Pplot.set_xlim(0, time_evolution_result.last_t)
    Pplot.set_ylim(0, 1)
    Pplot.plot(time_evolution_result.t_list, time_evolution_result.P_list,
               color="yellow", label="Prob. source has the app")
    Pplot.plot(time_evolution_result.t_list, time_evolution_result.FTinfty_list,
               color="black", label="Prob. to spot infected Global")
    Pplot.plot(time_evolution_result.t_list, time_evolution_result.FTinftyapp_list,
               color="green", label="Prob. to spot infected APP")
    Pplot.plot(time_evolution_result.t_list, time_evolution_result.FTinftynoapp_list,
               color="red", label="Prob. to spot infected NO APP")
    Pplot.legend()


def show_plot():
    print("Enjoy the plot!")
    plt.show()
