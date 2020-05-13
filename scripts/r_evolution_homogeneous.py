import warnings

from utilities.plotting_utils import show_plot, plot_time_evolution
from utilities.scenario import Scenario
from utilities.time_evolution import compute_time_evolution

warnings.filterwarnings('ignore')

scenario = Scenario(sSapp=0.7, sSnoapp=0., sCapp=0.7, sCnoapp=0.0,
                    xi=0.9, epsilon0=1., Deltat_testapp=2, Deltat_testnoapp=4)

time_evolution_result = compute_time_evolution(scenario,
                                               n_iterations=6,
                                               verbose=True)

plot_time_evolution(scenario, time_evolution_result)

show_plot()
