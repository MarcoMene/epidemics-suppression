import warnings

from utilities.plotting_utils import show_plot, plot_time_evolution
from utilities.scenario import Scenario
from utilities.time_evolution import compute_time_evolution

warnings.filterwarnings('ignore')

scenario = Scenario(sSapp=0.6, sSnoapp=0.0, sCapp=0.9, sCnoapp=0.0,
                    xi=1., epsilon0=1., Deltat_testapp=0, Deltat_testnoapp=0)

time_evolution_result = compute_time_evolution(scenario,
                                               n_iterations=6,
                                               verbose=True)

plot_time_evolution(scenario, time_evolution_result)

show_plot()
