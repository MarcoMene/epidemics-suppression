from bsp_epidemic_suppression_model.utilities.model import r0asy, r0sy, fraction_symptomatics
from bsp_epidemic_suppression_model.utilities.scenario import Scenario
from bsp_epidemic_suppression_model.utilities.functions_utils import (
    DeltaMeasure,
    RealRange,
)
from bsp_epidemic_suppression_model.utilities.plotting_utils import plot_time_evolution
from bsp_epidemic_suppression_model.algorithm.time_evolution_with_severity import (
    compute_time_evolution_with_severity,
)

import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # gs = [asymptomatic, symptomatic]
    tau_max = 30
    integration_step = 0.1

    scenario = Scenario(
        p_gs=[1-fraction_symptomatics, fraction_symptomatics],
        r0_gs=[lambda t, tau: r0asy(tau), lambda t, tau: r0sy(tau),],
        t_0=0,
        ssapp=[0, 0.2],
        ssnoapp=[0, 0.2],
        scapp=0.8,
        scnoapp=0.,
        xi=.9,
        papp=lambda tau: 0.6,
        p_DeltaATapp=DeltaMeasure(position=0),
        p_DeltaATnoapp=DeltaMeasure(position=0),
    )

    step_data_list = compute_time_evolution_with_severity(
        scenario=scenario,
        real_range=RealRange(0, tau_max, integration_step),
        n_iterations=6,
        verbose=True,
    )

    plot_time_evolution(step_data_list=step_data_list)
