from bsp_epidemic_suppression_model.utilities import gamma_density
from bsp_epidemic_suppression_model.utilities import plot_functions
from _dev.old_stuff import show_plot
from bsp_epidemic_suppression_model.utilities import DeltaMeasure, convolve, RealRange

f1 = lambda x: gamma_density(x - 10, 4, 1)
# f2= lambda x: gamma_density(x-10, 20, 20)
f2 = DeltaMeasure(position=5, height=1)

g = convolve(f1, f2, RealRange(0, 30, 1))
plot_functions([f1, g], RealRange(0, 30, 0.1))
show_plot()
