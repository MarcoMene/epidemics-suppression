from epidemic_suppression_algorithms.homogeneous_evolution_algorithm import (
    compute_time_evolution_homogeneous_case,
)
from math_utilities.discrete_distributions_utils import delta_distribution
from model_utilities.epidemic_data import (
    R0,
    make_scenario_parameters_for_asymptomatic_symptomatic_model,
    tauS,
)
from model_utilities.scenarios import HomogeneousScenario


def check_equality_with_precision(x: float, y: float, decimal: int):
    return round(x - y, ndigits=decimal) == 0


class TestCornerCases:
    def test_no_epidemic_control_scenario(self):
        p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()

        scenario = HomogeneousScenario(
            p_gs=p_gs,
            b0_gs=b0_gs,
            tauS=tauS,
            t_0=0,
            ss=(lambda t: 0, lambda t: 0),
            sc=lambda t: 0,
            xi=lambda t: 1,
            DeltaAT=delta_distribution(peak_tau_in_days=2),
        )

        (
            t_in_days_list,
            nu,
            nu0,
            R,
            R_by_severity,
            FT_infty,
        ) = compute_time_evolution_homogeneous_case(
            scenario=scenario,
            t_max_in_days=20,
            nu_start=1000,
            b_negative_times=b0_gs,
            threshold_to_stop=0.001,
            verbose=False,
        )

        assert R[-1] == R0
        assert FT_infty[-1] == 0

    def test_only_symptoms_control(self):
        p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()

        scenario = HomogeneousScenario(
            p_gs=p_gs,
            b0_gs=b0_gs,
            tauS=tauS,
            t_0=0,
            ss=(lambda t: 0, lambda t: 0.5),
            sc=lambda t: 0,
            xi=lambda t: 1,
            DeltaAT=delta_distribution(peak_tau_in_days=2),
        )

        (
            t_in_days_list,
            nu,
            nu0,
            R,
            R_by_severity,
            FT_infty,
        ) = compute_time_evolution_homogeneous_case(
            scenario=scenario,
            t_max_in_days=20,
            nu_start=1000,
            b_negative_times=b0_gs,
            threshold_to_stop=0.001,
            verbose=False,
        )

        assert R[0] == R[-1] < R0
        assert FT_infty[0] == FT_infty[-1] > 0
