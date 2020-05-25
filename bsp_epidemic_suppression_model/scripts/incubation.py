from bsp_epidemic_suppression_model.utilities import FS
from bsp_epidemic_suppression_model.utilities import plot_functions
from _dev.old_stuff import show_plot
from bsp_epidemic_suppression_model.utilities import RealRange

tau_max = 30
step = 0.05

print("The CDF of tau^S:")
plot_functions([FS], RealRange(x_min=0, x_max=tau_max, step=step))
show_plot()
