"""
Examples running the complete algorithm in certain scenarios, and plotting the results.
"""

from bsp_epidemic_suppression_model.model_utilities.epidemic_data import (
    make_scenario_parameters_for_asymptomatics_symptomatics_model,
)
from bsp_epidemic_suppression_model.model_utilities.scenario import Scenario
from bsp_epidemic_suppression_model.math_utilities.functions_utils import (
    DeltaMeasure,
    RealRange,
)
from bsp_epidemic_suppression_model.model_utilities.plot_results import (
    plot_time_evolution,
)
from bsp_epidemic_suppression_model.algorithm.time_evolution_main_function import (
    compute_time_evolution,
)

import warnings

warnings.filterwarnings("ignore")


def time_evolution_low_sensitivities_example():
    """
    Example in which the sensitivities s^S and s^C are pretty low, and the app is used by 60% of the population.
    """

    # gs = [asymptomatic, symptomatic]
    tau_max = 30
    integration_step = 0.1

    n_iterations = 8

    p_gs, beta0_gs = make_scenario_parameters_for_asymptomatics_symptomatics_model()

    scenario = Scenario(
        p_gs=p_gs,
        beta0_gs=beta0_gs,
        t_0=0,
        ssapp=[0, 0.2],
        ssnoapp=[0, 0.2],
        scapp=0.5,
        scnoapp=0.2,
        xi=0.7,
        papp=lambda tau: 0.6,
        p_DeltaATapp=DeltaMeasure(position=2),
        p_DeltaATnoapp=DeltaMeasure(position=4),
    )

    step_data_list = compute_time_evolution(
        scenario=scenario,
        real_range=RealRange(0, tau_max, integration_step),
        n_iterations=n_iterations,
        verbose=True,
    )

    plot_time_evolution(step_data_list=step_data_list)


if __name__ == "__main__":
    time_evolution_low_sensitivities_example()
