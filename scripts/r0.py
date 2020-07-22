from bsp_epidemic_suppression_model.model_utilities.epidemic_data import beta0
from bsp_epidemic_suppression_model.math_utilities.plotting_utils import plot_functions
from bsp_epidemic_suppression_model.math_utilities.functions_utils import RealRange
import scipy.integrate as integrate

tau_max = 30
step = 0.05

print("The function r^0_0:")
plot_functions([beta0], RealRange(x_min=0, x_max=tau_max, step=step))

print("Integral of r^0_0:", integrate.quad(beta0, 0, 30)[0])  # Should give back R0
