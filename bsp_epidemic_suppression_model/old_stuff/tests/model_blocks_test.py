from bsp_epidemic_suppression_model.math_utilities.general_utilities import RealRange
from bsp_epidemic_suppression_model.old_stuff.epidemic_data import (
    make_scenario_parameters_for_asymptomatic_symptomatic_model,
)
from bsp_epidemic_suppression_model.old_stuff.functions_utils import (
    DeltaMeasure,
    integrate,
)
from dev.old_stuff.algorithm.model_blocks import (
    compute_beta_and_R_components_from_FT,
    compute_FA_from_FAs_and_previous_step_data,
    compute_FT_from_FA_and_DeltaAT,
)


class TestAlgorithmBlock:
    """Tests each block of the algorithm separately"""

    def test_R_suppression(self):
        _, beta0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()
        beta0_ti_gs = [lambda tau: beta0_gs[0](0, tau), lambda tau: beta0_gs[1](0, tau)]
        xi = 0.8  # Any value in [0,1] will do
        tau_max = 30

        FTapp_ti_gs = [
            lambda tau: 0,
            lambda tau: 1 if tau >= 10 else 0,
        ]
        FTnoapp_ti_gs = [
            lambda tau: 0,
            lambda tau: 1 if tau >= 20 else 0,
        ]

        (
            betaapp_ti_gs,
            betanoapp_ti_gs,
            Rapp_ti_gs,
            Rnoapp_ti_gs,
        ) = compute_beta_and_R_components_from_FT(
            FTapp_ti_gs=FTapp_ti_gs,
            FTnoapp_ti_gs=FTnoapp_ti_gs,
            beta0_ti_gs=beta0_ti_gs,
            xi=xi,
            tau_max=tau_max,
        )

        R0_gs = [integrate(f=beta0_ti_gs[g], a=0, b=tau_max) for g in [0, 1]]

        # No suppression for asymptomatic:
        assert all(
            beta0_ti_gs[0](tau) == betaapp_ti_gs[0](tau) == betanoapp_ti_gs[0](tau)
            for tau in (0, 3, 6, 9, 12, 15)
        )
        assert Rapp_ti_gs[0] == Rnoapp_ti_gs[0] == R0_gs[0]
        # For symptomatic with app, suppression for tau >= 10
        assert all(beta0_ti_gs[1](tau) == betaapp_ti_gs[1](tau) for tau in (0, 3, 6, 9))
        assert all(
            betaapp_ti_gs[1](tau) == (1 - xi) * beta0_ti_gs[1](tau)
            for tau in (10, 13, 16, 19)
        )
        assert (1 - xi) * R0_gs[1] <= Rapp_ti_gs[1] <= R0_gs[1]
        # For symptomatic without app, suppression for tau >= 20
        assert all(
            beta0_ti_gs[1](tau) == betanoapp_ti_gs[1](tau)
            for tau in (0, 3, 6, 9, 12, 15, 18)
        )
        assert all(
            betanoapp_ti_gs[1](tau) == (1 - xi) * beta0_ti_gs[1](tau)
            for tau in (20, 25)
        )
        assert (1 - xi) * R0_gs[1] <= Rnoapp_ti_gs[1] <= R0_gs[1]

    def test_compute_FT(self):
        FAapp_ti_gs = [
            lambda tau: 0.4,
            lambda tau: 1 if tau >= 10 else 0.5,
        ]
        FAnoapp_ti_gs = [
            lambda tau: 0.5 if tau >= 20 else 0,
            lambda tau: 1 if tau >= 20 else 0.3,
        ]

        position_app = 1
        position_noapp = 5
        p_DeltaATapp = DeltaMeasure(position=position_app)
        p_DeltaATnoapp = DeltaMeasure(position=position_noapp)

        real_range = RealRange(x_min=0, x_max=30, step=0.1)

        FTapp_ti_gs, FTnoapp_ti_gs = compute_FT_from_FA_and_DeltaAT(
            FAapp_ti_gs=FAapp_ti_gs,
            FAnoapp_ti_gs=FAnoapp_ti_gs,
            p_DeltaATapp=p_DeltaATapp,
            p_DeltaATnoapp=p_DeltaATnoapp,
        )

        # Check that each FT component is the translation of the respective FA component

        assert all(
            FAapp_ti_gs[0](tau - position_app) == FTapp_ti_gs[0](tau)
            for tau in (-10, -2, 0, 2, 5, 10, 20, 30)
        )
        assert all(
            FAapp_ti_gs[1](tau - position_app) == FTapp_ti_gs[1](tau)
            for tau in (-10, -2, 0, 2, 5, 10, 20, 30)
        )
        assert not all(
            FAapp_ti_gs[1](tau) == FTapp_ti_gs[1](tau) for tau in (8, 10, 12)
        )
        assert all(
            FAnoapp_ti_gs[0](tau - position_noapp) == FTnoapp_ti_gs[0](tau)
            for tau in (-10, 10, 20, 22, 27, 30)
        )
        assert all(
            FAnoapp_ti_gs[1](tau - position_noapp) == FTnoapp_ti_gs[1](tau)
            for tau in (-10, 10, 20, 22, 27, 30)
        )
        assert not all(
            FAnoapp_ti_gs[1](tau) == FTnoapp_ti_gs[1](tau) for tau in (18, 20, 22)
        )

    def test_compute_FA(self):
        # No contact tracing without app

        FAsapp_ti_gs = [lambda tau: 0, lambda tau: 1]
        FAsnoapp_ti_gs = [lambda tau: 0] * 2

        tildeFTapp_tim1 = lambda tau: 1 if tau >= 10 else 0.5

        tildepapp_tim1 = 0.5
        EtauC_tim1 = 3

        FAapp_ti_gs, FAnoapp_ti_gs = compute_FA_from_FAs_and_previous_step_data(
            FAsapp_ti_gs=FAsapp_ti_gs,
            FAsnoapp_ti_gs=FAsnoapp_ti_gs,
            tildepapp_tim1=tildepapp_tim1,
            tildeFTapp_tim1=tildeFTapp_tim1,
            tildeFTnoapp_tim1=lambda tau: 0,
            EtauC_tim1=EtauC_tim1,
            scapp=1,
            scnoapp=0,
        )

        # Expected results
        # FAc_ti is:
        # - tildeFTapp_tim1 multiplied by tildepapp_tim1 and translated by EtauC_tim1 to the left for who has the app
        # - zero otherwise:
        FAcapp_ti = lambda tau: tildepapp_tim1 * tildeFTapp_tim1(tau + EtauC_tim1)
        FAcnoapp_ti = lambda tau: 0

        for g in (0, 1):
            # FA = FAs + FAc - FAs * FAt
            assert all(
                FAapp_ti_gs[g](tau)
                == FAsapp_ti_gs[g](tau)
                + FAcapp_ti(tau)
                - FAsapp_ti_gs[g](tau) * FAcapp_ti(tau)
                for tau in (-10, 0, 8, 10, 12, 15, 20)
            )
            assert all(
                FAnoapp_ti_gs[g](tau)
                == FAsnoapp_ti_gs[g](tau)
                + FAcnoapp_ti(tau)
                - FAsnoapp_ti_gs[g](tau) * FAcnoapp_ti(tau)
                for tau in (-10, 0, 8, 10, 12, 15, 20)
            )
