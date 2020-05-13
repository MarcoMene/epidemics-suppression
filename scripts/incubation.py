from utilities.model import FS
from utilities.plotting_utils import plot_functions, show_plot
from utilities.utils import RealRange
import scipy.integrate as integrate

tau_max = 30
step = 0.05

print("The CDF of tau^S:")
plot_functions([FS], RealRange(x_min=0, x_max=tau_max, step=step))
show_plot()
