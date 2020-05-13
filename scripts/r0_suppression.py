from utilities.model import r0, suppressed_r_from_test_cdf
from utilities.plotting_utils import plot_functions, show_plot
from utilities.utils import RealRange
import scipy.integrate as integrate
from numpy import heaviside

tau_max = 30
step = 0.05

tau_s = 10
F_Tsimple = lambda tau: heaviside(tau - tau_s, 1)
xi = 0.7  # Probability of (immediate) isolation given positive test

suppressed_r_0 = suppressed_r_from_test_cdf(r0, F_Tsimple, xi)


suppressed_R_0 = integrate.quad(lambda tau: suppressed_r_0(tau), 0, tau_max)[0]
print("suppressed R_0 =", suppressed_R_0)

plot_functions([r0, suppressed_r_0], RealRange(x_min=0, x_max=tau_max, step=step))
show_plot()
