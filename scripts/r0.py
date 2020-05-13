from utilities.model import r0
from utilities.plotting_utils import plot_functions, show_plot
from utilities.utils import RealRange
import scipy.integrate as integrate

tau_max = 30
step = 0.05

print("The function r^0_0:")
plot_functions([r0], RealRange(x_min=0, x_max=tau_max, step=step))

print("Integral of r^0_0:", integrate.quad(r0, 0, 30)[0])  # Should give back R0
show_plot()
