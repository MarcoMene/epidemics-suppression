from utilities.model import r0, suppressed_r_from_test_cdf, R0, FS, effectiveness_from_R
from utilities.plotting_utils import plot_functions, show_plot
from utilities.utils import RealRange, DeltaMeasure, f_from_list, list_from_f, convolve, round2
import scipy.integrate as integrate
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

#   *********************************** CONFIG  ****************************************** #

tau_max = 30
step = 0.05
N_iterations = 6

#   *********************************** INPUTS  ****************************************** #

sSapp = 0.7  # Probability of (immediate) CTA given symptoms
sSnoapp = 0.2
sCapp = 0.7
sCnoapp = 0.
xi = 0.9


def epsilon(t):
    if t < 0:
        return 0
    else:
        return 0.6


Deltat_testapp = 2
Deltat_testnoapp = 4

#   *********************************** FUNCTIONS  ****************************************** #


p_DeltaATapp = DeltaMeasure(position=Deltat_testapp, height=1)
p_DeltaATnoapp = DeltaMeasure(position=Deltat_testnoapp, height=1)


def FAsapp(tau):
    return sSapp * FS(tau)


def FAsnoapp(tau):
    return sSnoapp * FS(tau)


#   *********************************** ITERATIONS  ****************************************** #


# FA_list = []
t_list = []
tildepC_list = []
FTapp_list = []
FTnoapp_list = []
FTinftyapp_list = []
FTinftynoapp_list = []
FTinfty_list = []
R_list = []
Rapp_list = []
Rnoapp_list = []
P_list = []
tauC_exp_list = []

fig = plt.figure(figsize=(10, 15))

r_tplot = fig.add_subplot(311)
r_tplot.set_xlabel('Day')
r_tplot.set_ylabel('r_t')
r_tplot.grid(True)
r_tplot.set_xlim(0, tau_max)
r_tplot.set_ylim(0, R0 / 6)

tau_values = RealRange(0, tau_max, step).x_values

for i in range(0, N_iterations):

    if i == 0:
        t = 0
        FAapp_t = FAsapp
        FAnoapp_t = FAsnoapp

    else:

        FTapp_prev_values = FTapp_list[-1]
        FTapp_prev = f_from_list(FTapp_prev_values, RealRange(0, tau_max, step))
        FTnoapp_prev_values = FTnoapp_list[-1]
        FTnoapp_prev = f_from_list(FTnoapp_prev_values, RealRange(0, tau_max, step))
        tildepC_prev_values = tildepC_list[-1]
        tildepC_prev = f_from_list(tildepC_prev_values, RealRange(0, tau_max, step))

        tildetauC_exp = integrate.quad(lambda tau: tau * tildepC_prev(tau), 0, tau_max)[0]
        t = t_list[-1] + tildetauC_exp

        P_prev = P_list[-1]

        # Evolution equations here:
        LHSapp = lambda tau: sCapp * P_prev * FTapp_prev(tau) + (1 - P_prev) * sCnoapp * FTnoapp_prev(tau)
        FAcapp_t = lambda tau: LHSapp(tau + tildetauC_exp)
        FAapp_t = lambda tau: FAsapp(tau) + FAcapp_t(tau) - FAsapp(tau) * FAcapp_t(tau)

        LHSnoapp = lambda tau: sCnoapp * P_prev * FTapp_prev(tau) + (1 - P_prev) * sCnoapp * FTnoapp_prev(tau)
        FAcnoapp_t = lambda tau: LHSnoapp(tau + tildetauC_exp)
        FAnoapp_t = lambda tau: FAsnoapp(tau) + FAcnoapp_t(tau) - FAsnoapp(tau) * FAcnoapp_t(tau)

    FTapp_t = convolve(FAapp_t, p_DeltaATapp, RealRange(0, tau_max, step))
    rapp_t = suppressed_r_from_test_cdf(lambda tau: r0(tau), FTapp_t, xi)

    FTnoapp_t = convolve(FAnoapp_t, p_DeltaATnoapp, RealRange(0, tau_max, step))
    rnoapp_t = suppressed_r_from_test_cdf(lambda tau: r0(tau), FTnoapp_t, xi)

    r_t = lambda tau: epsilon(t) * rapp_t(tau) + (1 - epsilon(t)) * rnoapp_t(tau)

    Rapp_t = integrate.quad(rapp_t, 0, tau_max)[0]
    Rnoapp_t = integrate.quad(rnoapp_t, 0, tau_max)[0]
    R_t = integrate.quad(r_t, 0, tau_max)[0]

    tauC_exp = integrate.quad(lambda tau: tau * r_t(tau) / R_t, 0, tau_max)[0]

    P_t = epsilon(t) * Rapp_t / R_t

    tilder_t = lambda tau: P_t * rapp_t(tau) + (1 - P_t) * rnoapp_t(tau)
    tildeR_t = integrate.quad(tilder_t, 0, tau_max)[0]
    tildepC_t = lambda tau: tilder_t(tau) / tildeR_t

    FTinftyapp = FTapp_t(tau_max)
    FTinftynoapp = FTnoapp_t(tau_max)
    FTinfty = epsilon(t) * FTinftyapp + (1 - epsilon(t)) * FTinftynoapp

    t_list.append(t)
    r_t_values = list_from_f(r_t, RealRange(0, tau_max, step))
    tildepC_t_values = list_from_f(tildepC_t, RealRange(0, tau_max, step))
    FTapp_t_values = list_from_f(FTapp_t, RealRange(0, tau_max, step))
    FTnoapp_t_values = list_from_f(FTnoapp_t, RealRange(0, tau_max, step))
    tildepC_list.append(tildepC_t_values)
    FTapp_list.append(FTapp_t_values)
    FTnoapp_list.append(FTnoapp_t_values)
    FTinftyapp_list.append(FTinftyapp)
    FTinftynoapp_list.append(FTinftynoapp)
    FTinfty_list.append(FTinfty)
    R_list.append(R_t)
    Rapp_list.append(Rapp_t)
    Rnoapp_list.append(Rnoapp_t)
    P_list.append(P_t)
    tauC_exp_list.append(tauC_exp)

    print(f"t={round2(t)}, \n"
          f" FTapp_t(infty)={round2(FTinftyapp)},  FTnoapp_t(infty)={round2(FTinftynoapp)},  FT_t(infty)={round2(FTinfty)}, \n"
          f"    epsilon_t={epsilon(t)},  P_t={round2(P_t)}"
          f"    R_t={round2(R_t)}"
          f"    E(tauC)={round2(tauC_exp)}"
          )

    r_tplot.plot(tau_values, r_t_values, "r")

    fig.canvas.draw()

R_tplot = fig.add_subplot(312)
R_tplot.set_xlabel('t [days]')
R_tplot.set_ylabel('R_t')
R_tplot.grid(True)
R_tplot.set_xlim(0, t_list[-1])
R_tplot.set_ylim(0, R0)
R_tplot.plot(t_list, R_list, color="black", label=f"R - Eff. {round2(effectiveness_from_R(R_list[-1]))}")
R_tplot.plot(t_list, Rapp_list, color="green", label=f"R app - Eff. {round2(effectiveness_from_R(Rapp_list[-1]))}")
R_tplot.plot(t_list, Rnoapp_list, color="red", label=f"R no app - Eff. {round2(effectiveness_from_R(Rnoapp_list[-1]))}")
R_tplot.legend()

Pplot = fig.add_subplot(313)
Pplot.set_xlabel('t [days]')
Pplot.set_ylabel('P_t')
Pplot.grid(True)
Pplot.set_xlim(0, t_list[-1])
Pplot.set_ylim(0)
Pplot.plot(t_list, P_list, color="yellow", label="Prob. source has the app")
Pplot.plot(t_list, FTinfty_list, color="black", label="Prob. to spot infected Global")
Pplot.plot(t_list, FTinftyapp_list, color="green", label="Prob. to spot infected APP")
Pplot.plot(t_list, FTinftynoapp_list, color="red", label="Prob. to spot infected NO APP")
Pplot.legend()

show_plot()
