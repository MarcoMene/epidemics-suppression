"""
Contains a function that runs the algorithm several times, each with a different choice of the input parameters.
"""

from bsp_epidemic_suppression_model.model_utilities.epidemic_data import (
    make_scenario_parameters_for_asymptomatic_symptomatic_model,
)
from bsp_epidemic_suppression_model.model_utilities.scenario import Scenario
from bsp_epidemic_suppression_model.math_utilities.functions_utils import (
    DeltaMeasure,
    RealRange,
)
from bsp_epidemic_suppression_model.algorithm.model_blocks import effectiveness_from_R
from bsp_epidemic_suppression_model.algorithm.time_evolution_main_function import (
    compute_time_evolution,
)
from bsp_epidemic_suppression_model.math_utilities.functions_utils import round2

import warnings

warnings.filterwarnings("ignore")


def time_evolution_with_varying_parameters():
    """
    Run the algorithm several times, each with a different choice of the parameters s^S, s^C, xi, and p^app.
    """

    tau_max = 30
    integration_step = 0.1

    n_iterations = 8

    # gs = [asymptomatic, symptomatic]
    p_gs, beta0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()

    ssnoapp = 0.2
    scnoapp = 0.2
    DeltaATapp = 2
    DeltaATnoapp = 4

    # Varying parameters
    ssapp_list = [0.2, 0.5, 0.8]
    scapp_list = [0.5, 0.8]
    xi_list = [0.7, 0.9]
    papp_list = [0.2, 0.5, 0.7, 0.9]

    for ssapp in ssapp_list:
        for scapp in scapp_list:
            for xi in xi_list:
                for papp in papp_list:
                    scenario = Scenario(
                        p_gs=p_gs,
                        beta0_gs=beta0_gs,
                        t_0=0,
                        ssapp=[0, ssapp],
                        ssnoapp=[0, ssnoapp],
                        scapp=scapp,
                        scnoapp=scnoapp,
                        xi=xi,
                        papp=lambda tau: papp,
                        p_DeltaATapp=DeltaMeasure(position=DeltaATapp),
                        p_DeltaATnoapp=DeltaMeasure(position=DeltaATnoapp),
                    )

                    step_data_list = compute_time_evolution(
                        scenario=scenario,
                        real_range=RealRange(0, tau_max, integration_step),
                        n_iterations=n_iterations,
                        verbose=False,
                    )

                    Rinfty = step_data_list[-1].R
                    Effinfty = effectiveness_from_R(Rinfty)

                    print(
                        f" {ssapp} & {scapp} & {xi} & {papp} & {round2(Rinfty)} & {round2(Effinfty)} \\\ "
                    )
