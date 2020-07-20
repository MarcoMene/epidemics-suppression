from bsp_epidemic_suppression_model.utilities.model import (
    R0,
    r0,
    make_scenario_parameters_for_asymptomatics_symptomatics_model,
)
from bsp_epidemic_suppression_model.utilities.scenario import Scenario
from bsp_epidemic_suppression_model.utilities.functions_utils import (
    DeltaMeasure,
    RealRange,
)
from bsp_epidemic_suppression_model.algorithm.time_evolution_with_severity import (
    compute_time_evolution_with_severity,
)


def check_equality_with_precision(x: float, y: float, decimal: int):
    return round(x - y, ndigits=decimal) == 0


class TestCornerCases:
    def test_no_epidemic_control_scenario(self):
        tau_max = 30
        integration_step = 0.1

        scenario = Scenario(
            p_gs=[1],
            r0_gs=[lambda t, tau: r0(tau)],
            t_0=0,
            ssapp=[0],
            ssnoapp=[0],
            scapp=0,
            scnoapp=0,
            xi=1,
            papp=lambda tau: 0.6,
            p_DeltaATapp=DeltaMeasure(position=1),
            p_DeltaATnoapp=DeltaMeasure(position=2),
        )

        step_data_list = compute_time_evolution_with_severity(
            scenario=scenario,
            real_range=RealRange(0, tau_max, integration_step),
            n_iterations=4,
            verbose=False,
        )

        last_step_data = step_data_list[-1]

        precision = 3

        assert check_equality_with_precision(
            x=last_step_data.R, y=R0, decimal=precision
        )
        assert check_equality_with_precision(
            x=last_step_data.FT_infty, y=0, decimal=precision
        )
        assert check_equality_with_precision(
            x=last_step_data.papp, y=last_step_data.tildepapp, decimal=precision
        )

    def test_only_symptoms_control(self):

        tau_max = 30
        integration_step = 0.1

        # gs = [asymptomatic, symptomatic]:
        p_gs, r0_gs = make_scenario_parameters_for_asymptomatics_symptomatics_model()

        scenario = Scenario(
            p_gs=p_gs,
            r0_gs=r0_gs,
            t_0=0,
            ssapp=[0, 0.7],
            ssnoapp=[0, 0.5],
            scapp=0,
            scnoapp=0,
            xi=1,
            papp=lambda tau: 0,
            p_DeltaATapp=DeltaMeasure(position=1),
            p_DeltaATnoapp=DeltaMeasure(position=2),
        )

        step_data_list = compute_time_evolution_with_severity(
            scenario=scenario,
            real_range=RealRange(0, tau_max, integration_step),
            n_iterations=4,
            verbose=False,
        )

        precision = 3
        first_step_data = step_data_list[0]
        last_step_data = step_data_list[-1]
        assert check_equality_with_precision(
            x=first_step_data.R, y=last_step_data.R, decimal=precision
        )


if __name__ == "__main__":

    # TestCornerCases().test_no_epidemic_control_scenario()
    TestCornerCases().test_only_symptoms_control()
