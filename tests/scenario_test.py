from utilities.scenario import Scenario
from utilities.utils import RealRange, list_from_f, f_from_list


def test_scenario():
    scenario = Scenario(sSapp=0.7, sSnoapp=0.2, sCapp=0.7, sCnoapp=0.2,
                        xi=0.9, epsilon0=0.6)

    # verify defaults
    assert scenario.Deltat_testapp == 0
    assert scenario.Deltat_testnoapp == 4
    assert scenario.t_epsilon == 0

    assert scenario.epsilon(-1) == 0
    assert scenario.epsilon(0) == 0.6
    assert scenario.epsilon(1000) == 0.6

    assert scenario.p_DeltaATapp.height == 1
    assert scenario.p_DeltaATapp.position == 0

    assert scenario.FAsapp(100) > 0.99*scenario.sSapp
    assert scenario.FAsnoapp(100) > 0.99*scenario.sSnoapp

    assert scenario.FAsapp(0) == 0
    assert scenario.FAsnoapp(0) == 0

