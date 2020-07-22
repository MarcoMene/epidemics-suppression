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
from bsp_epidemic_suppression_model.algorithm.time_evolution_with_severity import (
    compute_time_evolution_with_severity,
)

import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # gs = [asymptomatic, symptomatic]
    tau_max = 30
    integration_step = 0.1

    p_gs, r0_gs = make_scenario_parameters_for_asymptomatics_symptomatics_model()

    scenario = Scenario(
        p_gs=p_gs,
        r0_gs=r0_gs,
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

    step_data_list = compute_time_evolution_with_severity(
        scenario=scenario,
        real_range=RealRange(0, tau_max, integration_step),
        n_iterations=8,
        verbose=True,
    )

    plot_time_evolution(step_data_list=step_data_list)
