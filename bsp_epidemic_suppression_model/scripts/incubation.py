from bsp_epidemic_suppression_model.utilities.model import FS
from bsp_epidemic_suppression_model.utilities.plotting_utils import plot_functions
from bsp_epidemic_suppression_model.utilities.functions_utils import RealRange

tau_max = 30
step = 0.05

print("The CDF of tau^S:")
plot_functions([FS], RealRange(x_min=0, x_max=tau_max, step=step))
