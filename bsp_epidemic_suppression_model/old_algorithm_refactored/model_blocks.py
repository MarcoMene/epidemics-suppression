from typing import Sequence, Tuple

from bsp_epidemic_suppression_model.math_utilities.config import TAU_MAX_IN_UNITS
from bsp_epidemic_suppression_model.math_utilities.discrete_distributions_utils import (
    DiscreteDistributionOnNonNegatives,
    generate_discrete_distribution_from_pdf_function,
)
from bsp_epidemic_suppression_model.old_stuff.epidemic_data import R0
from bsp_epidemic_suppression_model.old_stuff.functions_utils import (
    ImproperProbabilityCumulativeFunction,
    ProbabilityCumulativeFunction,
)


def approximate_FA_from_FAs_and_previous_step_data(
    FAsapp_ti_gs: Sequence[ImproperProbabilityCumulativeFunction],
    FAsnoapp_ti_gs: Sequence[ImproperProbabilityCumulativeFunction],
    tildepapp_tim1: float,
    tildeFTapp_tim1: ProbabilityCumulativeFunction,
    tildeFTnoapp_tim1: ProbabilityCumulativeFunction,
    EtauC_tim1_relative_tau_units: float,
    scapp_ti: float,
    scnoapp_ti: float,
) -> (
    Tuple[ImproperProbabilityCumulativeFunction],
    Tuple[ImproperProbabilityCumulativeFunction],
):
    """
    Implements the time evolution formulae, computing the components of F^A at a time t_i given
    the components of F^{A,s} at t_i, the components of F^T at the previous time step t_{i-1},
    the expected value of the contagion time tau^C, and some parameters of the model.
    :param FAsapp_ti_gs: CDF of the symptoms notification time for an individual
      with the app infected at t_i
    :param FAsnoapp_ti_gs: CDF of the symptoms notification time for an individual
      without the app infected at t_i
    :param tildepapp_tim1: Probability that a source infected at t_{i-1} has the app
    :param tildeFTapp_tim1: CDF of the testing time of a source infected at t_{i-1} having the app
    :param tildeFTnoapp_tim1: CDF of the testing time of a source infected at t_{i-1}
      without the app
    :param EtauC_tim1_relative_tau_units: expected value of the contagion time tau^C at t_{i-1}.
    :return: FAapp_ti_gs, FAnoapp_ti_gs: the CDFs of the notification time for an individual
      infected at t_i, given that the have or do not have the app, respectively.
    """
    EtauC_tim1_relative_tau_units = int(EtauC_tim1_relative_tau_units)
    gs = range(len(FAsapp_ti_gs))  # Values of severity G

    # CDF of the time of notification for the source, given that the recipient has the app:
    FNapp_tim1 = lambda tau: (
        scapp_ti * tildepapp_tim1 * tildeFTapp_tim1(tau)
        + scnoapp_ti * (1 - tildepapp_tim1) * tildeFTnoapp_tim1(tau)
    )

    # CDF of the time of notification for the source, given that the recipient does not have the app
    # (this is simply scnoapp * tildeFT_tim1):
    FNnoapp_tim1 = lambda tau: (
        scnoapp_ti
        * (
            tildepapp_tim1 * tildeFTapp_tim1(tau)
            + (1 - tildepapp_tim1) * tildeFTnoapp_tim1(tau)
        )
    )

    def evolution(
        FN_tim1_component: ImproperProbabilityCumulativeFunction,
        FAs_ti_component: ImproperProbabilityCumulativeFunction,
        EtauC: int,
    ) -> ImproperProbabilityCumulativeFunction:
        """
        Implements the approximate evolution equations, computing the improper CDF F^A at
        a time t_i in terms of the improper CDF F^N at the previous time step t_{i-1}.
        """
        FAc_ti_component = lambda tau: FN_tim1_component(tau + EtauC)
        FA_ti_component = lambda tau: (
            FAs_ti_component(tau)
            + FAc_ti_component(tau)
            - FAs_ti_component(tau) * FAc_ti_component(tau)
        )
        return FA_ti_component

    FAapp_ti_gs = []
    FAnoapp_ti_gs = []

    for g in gs:
        FAapp_ti_g = evolution(
            FN_tim1_component=FNapp_tim1,
            FAs_ti_component=FAsapp_ti_gs[g],
            EtauC=EtauC_tim1_relative_tau_units,
        )
        FAapp_ti_gs.append(FAapp_ti_g)
        FAnoapp_ti_g = evolution(
            FN_tim1_component=FNnoapp_tim1,
            FAs_ti_component=FAsnoapp_ti_gs[g],
            EtauC=EtauC_tim1_relative_tau_units,
        )
        FAnoapp_ti_gs.append(FAnoapp_ti_g)

    return tuple(FAapp_ti_gs), tuple(FAnoapp_ti_gs)


# def convolve(f: Callable[[float], float], delta: DeltaMeasure):
#     """
#     Computes the convolution of a function f and a DeltaMeasure delta.
#     """
#     return lambda x: delta.height * f(x - delta.position)


# def compute_FT_from_FA_and_DeltaAT(
#     FAapp_ti_gs: List[ImproperProbabilityCumulativeFunction],
#     FAnoapp_ti_gs: List[ImproperProbabilityCumulativeFunction],
#     p_DeltaATapp: DiscreteDistributionOnNonNegatives,
#     p_DeltaATnoapp: DiscreteDistributionOnNonNegatives,
# ) -> Tuple[
#     List[ImproperProbabilityCumulativeFunction],
#     List[ImproperProbabilityCumulativeFunction],
# ]:
#     """Implements the formula giving each component of the test time CDF F^T as a convolution of the respective
#     components of the notification time CDF F^A and the notification-to-test distribution Delta^{A -> T}."""
#     gs = range(len(FAapp_ti_gs))  # Values of severity G
#
#     FTapp_ti_gs = []
#     FTnoapp_ti_gs = []
#     for g in gs:
#         FTapp_ti_g = convolve(f=FAapp_ti_gs[g], delta=p_DeltaATapp)
#         FTapp_ti_gs.append(FTapp_ti_g)
#         FTnoapp_ti_g = convolve(f=FAnoapp_ti_gs[g], delta=p_DeltaATnoapp)
#         FTnoapp_ti_gs.append(FTnoapp_ti_g)
#
#     return FTapp_ti_gs, FTnoapp_ti_gs


def suppressed_beta_from_tauT(
    beta0_component: DiscreteDistributionOnNonNegatives,
    tauT_component: DiscreteDistributionOnNonNegatives,
    xi: float,
) -> DiscreteDistributionOnNonNegatives:
    """
    Given a component of the default infectiousness beta^0 and the corresponding component of the CDF F^T for the test
    results time, calculates the component of the suppressed infectiousness.
    """
    beta_component = lambda tau: beta0_component.pmf(tau) * (
        1 - tauT_component.cdf(tau) * xi
    )
    return generate_discrete_distribution_from_pdf_function(
        pdf=beta_component,
        max_tau_in_density_measure=TAU_MAX_IN_UNITS,
        time_step=1,
        improper=True,
    )


def compute_beta_and_R_components_from_FT(
    tauTapp_ti_gs: Tuple[DiscreteDistributionOnNonNegatives, ...],
    tauTnoapp_ti_gs: Tuple[DiscreteDistributionOnNonNegatives, ...],
    beta0_ti_gs: Tuple[DiscreteDistributionOnNonNegatives, ...],
    xi_ti: float,
) -> Tuple[
    Tuple[DiscreteDistributionOnNonNegatives, ...],
    Tuple[DiscreteDistributionOnNonNegatives, ...],
    Tuple[float, ...],
    Tuple[float, ...],
]:
    """
    Computes the app and no-app components of the suppressed infectiousness beta and the suppressed effective
    reproduction number R.
    :param FTapp_ti_gs: the list of test results times improper CDFs (one per severity component), for people with
    the app.
    :param FTnoapp_ti_gs: the list of test results times improper CDFs (one per severity component), for people without
    the app.
    :param beta0_ti_gs: the list of default infectiousness distributions (one per severity component).
    :param xi_ti: the probability of self-isolation given a positive test result.
    :param tau_max: maximum relative time when doing numerical integrations.
    """
    gs = range(len(tauTapp_ti_gs))  # Values of severity G

    betaapp_ti_gs = []
    betanoapp_ti_gs = []
    Rapp_ti_gs = []
    Rnoapp_ti_gs = []
    for g in gs:
        betaapp_ti_g = suppressed_beta_from_tauT(
            beta0_component=beta0_ti_gs[g], tauT_component=tauTapp_ti_gs[g], xi=xi_ti
        )
        betaapp_ti_gs.append(betaapp_ti_g)
        betanoapp_ti_g = suppressed_beta_from_tauT(
            beta0_component=beta0_ti_gs[g], tauT_component=tauTnoapp_ti_gs[g], xi=xi_ti
        )
        betanoapp_ti_gs.append(betanoapp_ti_g)

        Rapp_ti_g = betaapp_ti_g.total_mass
        Rapp_ti_gs.append(Rapp_ti_g)
        Rnoapp_ti_g = betanoapp_ti_g.total_mass
        Rnoapp_ti_gs.append(Rnoapp_ti_g)

    return (
        tuple(betaapp_ti_gs),
        tuple(betanoapp_ti_gs),
        tuple(Rapp_ti_gs),
        tuple(Rnoapp_ti_gs),
    )


def compute_aggregate_beta_t_two_components(
    p_gs: Tuple[float, ...],
    papp_t: float,
    betaapp_t_gs: Tuple[DiscreteDistributionOnNonNegatives, ...],
    betanoapp_t_gs: Tuple[DiscreteDistributionOnNonNegatives, ...],
) -> DiscreteDistributionOnNonNegatives:

    gs = range(len(p_gs))

    betaapp_ti = lambda tau: sum(p_gs[g] * betaapp_t_gs[g].pmf(tau) for g in gs)
    betanoapp_ti = lambda tau: sum(p_gs[g] * betanoapp_t_gs[g].pmf(tau) for g in gs)
    beta_ti = lambda tau: papp_t * betaapp_ti(tau) + (1 - papp_t) * betanoapp_ti(tau)

    return generate_discrete_distribution_from_pdf_function(
        pdf=beta_ti,
        max_tau_in_density_measure=TAU_MAX_IN_UNITS,
        time_step=1,
        improper=True,
    )


def effectiveness_from_R(R: float) -> float:
    """
    Effectiveness of the isolation measures, expressed as fraction of R reduction.
    """
    return 1.0 - R / R0
