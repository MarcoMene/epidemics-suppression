import warnings

from utilities.plotting_utils import show_plot, plot_time_evolution
from utilities.scenario import Scenario
from utilities.time_evolution_with_asymptomatics import compute_time_evolution_with_asymptomatics

warnings.filterwarnings('ignore')

scenario = Scenario(sSapp=1, sSnoapp=0., sCapp=0, sCnoapp=0.0,
                    xi=1, epsilon0=1., Deltat_testapp=2, Deltat_testnoapp=4)

time_evolution_result = compute_time_evolution_with_asymptomatics(scenario,
                                               n_iterations=2,
                                               verbose=True)

plot_time_evolution(scenario, time_evolution_result)

show_plot()
