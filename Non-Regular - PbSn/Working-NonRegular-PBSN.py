import math
import os
import numpy as np
from scipy.optimize import fsolve  # Could be useful to test some of your code.
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from scipy.misc import derivative
'''abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)'''
comp1 = 'Pb'
comp2 = 'Sn'
Tmelt1 = 327.46 + 273
Tmelt2 = 231.93 + 273
model_type = 'Non-Regular Solution Model'  # Again, this may need to be changed.
model_name = f'{comp1}-{comp2} - {model_type}'
start = time.time()
print('Setting up model and plotting our first GX curves...')
R = 8.31446261815324
LN = ln = Ln = lN = np.log
resolution = 1e-6

"""
INSERT YOUR CODE HERE...

You'll need to define equations from the TDB file and your own understanding of
thermodynamics to set up how you define the full mixing equations below.

IMPORTANT: If I have defined a function or variable as "INSERT YOUR CODE HERE",
make sure you keep the function or variable name unchanged. Otherwise, you will
be unable to use much of my subsequent code.

"""
if True:
    FCC_abs_Pb = lambda T: -10531.095 + 154.243182 * T - 32.4913959 * T * LN(
        T) + .00154613 * T ** 2 + 8.054E+25 * T ** (-9)
    BCT_abs_Sn = lambda T: (-5855.135 + 65.443315 * T - 15.961 * T * LN(T)
                            - .0188702 * T ** 2 + 3.121167E-06 * T ** 3
                            - 61960 * T ** (-1) if T < 505.078
                            else 2524.724 + 4.005269 * T - 8.2590486 * T * LN(T)
                                 - .016814429 * T ** 2 + 2.623131E-06 * T ** 3
                                 - 1081244 * T ** (-1) - 1.2307E+25 * T ** (-9))

    FCC_hanging_Pb = lambda T: 0
    FCC_hanging_Sn = lambda T: 5510 - 8.46 * T
    BCT_hanging_Pb = lambda T: (489 + 3.52 * T)
    BCT_hanging_Sn = lambda T: 0
    Liq_hanging_Pb = lambda T: ((4672.124 - 7.750683 * T - 60.19E-20 * T ** 7) if (T < 600.61)
                                else -5677.958 + 146.176046 * T - 32.4913959 * T * LN(
        T) + 1.54613E-3 * T ** 2 - FCC_abs_Pb(T))
    Liq_hanging_Sn = lambda T: (7103.092 - 14.087767 * T + 147.031E-20 * T ** 7 if T < 505.078
                                else 9496.31 - 9.809114 * T - 8.2590486 * T * LN(T)
                                     - 16.814429E-3 * T ** 2 + 2.623131E-6 * T ** 3
                                     - 1081244 * T ** (-1) - BCT_abs_Sn(T))

    S_ideal = lambda X_2: R * (X_2 * ln(X_2) + (1 - X_2) * ln(1 - X_2))
    G_ex_FCC = lambda X2, T: X2*(1-X2) * (
            (4758.8 + 2.4719 * T) +
            (2293.4-4.9197*T) * (1-2*X2)
    )
    G_ex_BCT = lambda X2, T: X2*(1-X2) * (
            (19693.75-15.89485*T)
    )
    G_ex_L = lambda X2, T: X2*(1-X2) * (
            (5368+0.93414*T) +
            (97.8+0.09354*T) * (1-2*X2)
    )
DeltaG_mix_FCC = lambda X_2, T: (T * S_ideal(X_2) + G_ex_FCC(X_2,T) +
                                 (1 - X_2) * FCC_hanging_Pb(T) + X_2 * FCC_hanging_Sn(T))
DeltaG_mix_BCT = lambda X_2, T: (T * S_ideal(X_2) + G_ex_BCT(X_2,T) +
                                 (1 - X_2) * BCT_hanging_Pb(T) + X_2 * BCT_hanging_Sn(T))
DeltaG_mix_L = lambda X_2, T: (T * S_ideal(X_2) + G_ex_L(X_2,T) +
                               (1 - X_2) * Liq_hanging_Pb(T) + X_2 * Liq_hanging_Sn(T))
plotres = 1e-4
Xrange = np.arange(plotres, 1 - plotres, plotres)
plottemp = 185 + 273.15
plt.figure('Random GX Plot', dpi=300)
plt.title(model_name + f' - $T$ = {plottemp - 273.15:.1f} °C\n')
plt.plot(Xrange, DeltaG_mix_FCC(Xrange, plottemp), color='r', label='FCC')
plt.plot(Xrange, DeltaG_mix_BCT(Xrange, plottemp), color='purple', label='BCT')
plt.plot(Xrange, DeltaG_mix_L(Xrange, plottemp), color='b', label='L')
plt.ylabel(r'Δ$G_\mathrm{mix}$ (J/mol)')
plt.xlabel(f'$X_\\mathrm{{{comp2}}}$ (mol/mol)')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
ymin = min(min(DeltaG_mix_FCC(Xrange, plottemp)), min(DeltaG_mix_BCT(Xrange, plottemp)),
           min(DeltaG_mix_L(Xrange, plottemp)))
plt.ylim(ymin * 1.1, 0)
plt.xlim(0, 1)
plt.legend()
plt.show()  # This line isn't necessary in Spyder, but it can't hurt.

print(f'This cell took {time.time() - start:.2f} seconds to run.')

start = time.time()

print('Solving for and plotting the eutectic GX curves...')
def ddx(function, dX):
    return lambda X, T: (function(X + dX, T) - function(X, T)) / dX


dG_Liq = ddx(DeltaG_mix_L, 10 ** -5)
dG_FCC = ddx(DeltaG_mix_FCC, 10 ** -5)
dG_BCT = ddx(DeltaG_mix_BCT, 10 ** -5)


def func(x):
    X_2_FCC, X_2_L, X_2_BCT, T = x

    eqns = [dG_BCT(X_2_BCT, T) - dG_FCC(X_2_FCC, T), dG_BCT(X_2_BCT, T) - dG_Liq(X_2_L, T),
            DeltaG_mix_BCT(X_2_BCT, T) - X_2_BCT * dG_BCT(X_2_BCT, T) - (
                        DeltaG_mix_L(X_2_L, T) - X_2_L * dG_Liq(X_2_L, T)),
            DeltaG_mix_BCT(X_2_BCT, T) - X_2_BCT * dG_BCT(X_2_BCT, T) - (
                        DeltaG_mix_FCC(X_2_FCC, T) - X_2_FCC * dG_FCC(X_2_FCC, T))]

    return eqns
X_2_FCC_guess = .01
X_2_L_guess = .5  # "INSERT YOUR CODE HERE"
X_2_BCT_guess = .9
T_guess = 185+273
xmin = resolution
xmax = 1 - resolution
tempmin = 0
tempmax = np.inf
res = least_squares(func, [X_2_FCC_guess, X_2_L_guess, X_2_BCT_guess, T_guess],
                    bounds=((xmin, xmin, xmin, tempmin), (xmax, xmax, xmax, tempmax)))
print(res)
eutec_soln = res.x
print(eutec_soln)
X_2_FCC_eutectic = eutec_soln[0]
X_2_L_eutectic = eutec_soln[1]
X_2_BCT_eutectic = eutec_soln[2]
T_eutectic = eutec_soln[3]
plottemp = T_eutectic
plt.figure('Eutectic GX Plot', dpi=300)
plt.title(model_name + f' - $T$ = {plottemp - 273.15:.1f} °C\n')
plt.plot(Xrange, DeltaG_mix_FCC(Xrange, plottemp), color='r', label='FCC')
plt.plot(Xrange, DeltaG_mix_BCT(Xrange, plottemp), color='purple', label='BCT')
plt.plot(Xrange, DeltaG_mix_L(Xrange, plottemp), color='b', label='L')
plt.plot([X_2_FCC_eutectic, X_2_BCT_eutectic],
         [DeltaG_mix_FCC(X_2_FCC_eutectic, T_eutectic), DeltaG_mix_BCT(X_2_BCT_eutectic, T_eutectic)], color='black',
         linestyle='--')
plt.ylabel(r'Δ$G_\mathrm{mix}$ (J/mol)')
plt.xlabel(f'$X_\\mathrm{{{comp2}}}$ (mol/mol)')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
ymin = min(min(DeltaG_mix_FCC(Xrange, plottemp)), min(DeltaG_mix_BCT(Xrange, plottemp)),
           min(DeltaG_mix_L(Xrange, plottemp)))
plt.ylim(ymin * 1.1, 0)
plt.xlim(0, 1)
plt.legend()
plt.show()  # This line isn't necessary in Spyder, but it can't hurt.

print(f'This cell took {time.time() - start:.2f} seconds to run.')

print('Defining the tangent solving functions...')
def FCCsolvus(T):
    if T <= T_eutectic:  # Temperature is below eutectic

        def func(x):
            X_2_FCC, X_2_BCT = x

            eqns = [dG_BCT(X_2_BCT, T) - dG_FCC(X_2_FCC, T),
                    DeltaG_mix_BCT(X_2_BCT, T) - X_2_BCT * dG_BCT(X_2_BCT, T) - (
                                DeltaG_mix_FCC(X_2_FCC, T) - X_2_FCC * dG_FCC(X_2_FCC, T))
                    ]

            return eqns
        X_2_FCC_guess = 0.001  # Giving you a helpful recommendation here.
        X_2_BCT_guess = 0.999  # Giving you a helpful recommendation here.
        xmin = resolution
        xmax = 1 - resolution
        res = least_squares(func, [X_2_FCC_guess, X_2_BCT_guess], bounds=((xmin, xmin), (xmax, xmax)))
        soln = res.x

        return soln[0]


def BCTsolvus(T):  # <--- Solving twice is so inefficient but alas I am lazy
    if T <= T_eutectic:
        def func(x):
            X_2_FCC, X_2_BCT = x

            eqns = [dG_BCT(X_2_BCT, T) - dG_FCC(X_2_FCC, T),
                    DeltaG_mix_BCT(X_2_BCT, T) - X_2_BCT * dG_BCT(X_2_BCT, T) - (
                                DeltaG_mix_FCC(X_2_FCC, T) - X_2_FCC * dG_FCC(X_2_FCC, T))
                    ]

            return eqns
        X_2_FCC_guess = .001
        X_2_BCT_guess = .999
        xmin = resolution
        xmax = 1 - resolution
        res = least_squares(func, [X_2_FCC_guess, X_2_BCT_guess], bounds=((xmin, xmin), (xmax, xmax)))
        soln = res.x

        return soln[1]


def FCCsolidus(T):
    if T_eutectic <= T <= Tmelt1:
        def func(x):
            X_2_FCC, X_2_L = x

            eqns = [dG_Liq(X_2_L, T) - dG_FCC(X_2_FCC, T),
                    DeltaG_mix_L(X_2_L, T) - X_2_L * dG_Liq(X_2_L, T) - (
                                DeltaG_mix_FCC(X_2_FCC, T) - X_2_FCC * dG_FCC(X_2_FCC, T))
                    ]

            return eqns
        X_2_FCC_guess = .01
        X_2_L_guess = .5
        xmin = resolution
        xmax = 1 - resolution
        res = least_squares(func, [X_2_FCC_guess, X_2_L_guess], bounds=((xmin, xmin), (xmax, xmax)))
        soln = res.x

        return soln[0]


def FCCliquidus(T):
    if T_eutectic <= T <= Tmelt1:  # Might need to include melting temp too

        def func(x):
            X_2_FCC, X_2_L = x

            eqns = [dG_Liq(X_2_L, T) - dG_FCC(X_2_FCC, T),
                    DeltaG_mix_L(X_2_L, T) - X_2_L * dG_Liq(X_2_L, T) - (
                                DeltaG_mix_FCC(X_2_FCC, T) - X_2_FCC * dG_FCC(X_2_FCC, T))
                    ]

            return eqns
        X_2_FCC_guess = .01
        X_2_L_guess = .5
        xmin = resolution
        xmax = 1 - resolution
        res = least_squares(func, [X_2_FCC_guess, X_2_L_guess], bounds=((xmin, xmin), (xmax, xmax)))
        soln = res.x

        return soln[1]


def BCTsolidus(T):
    if T_eutectic <= T <= Tmelt2:
        def func(x):
            X_2_BCT, X_2_L = x

            eqns = [dG_Liq(X_2_L, T) - dG_BCT(X_2_BCT, T),
                    DeltaG_mix_L(X_2_L, T) - X_2_L * dG_Liq(X_2_L, T) - (
                                DeltaG_mix_BCT(X_2_BCT, T) - X_2_BCT * dG_BCT(X_2_BCT, T))
                    ]

            return eqns
        X_2_BCT_guess = .999
        X_2_L_guess = .5
        xmin = resolution
        xmax = 1 - resolution
        res = least_squares(func, [X_2_BCT_guess, X_2_L_guess], bounds=((xmin, xmin), (xmax, xmax)))
        soln = res.x

        return soln[0]


def BCTliquidus(T):
    if T_eutectic <= T <= Tmelt2:
        def func(x):
            X_2_BCT, X_2_L = x

            eqns = [dG_Liq(X_2_L, T) - dG_BCT(X_2_BCT, T),
                    DeltaG_mix_L(X_2_L, T) - X_2_L * dG_Liq(X_2_L, T) - (
                                DeltaG_mix_BCT(X_2_BCT, T) - X_2_BCT * dG_BCT(X_2_BCT, T))
                    ]

            return eqns
        X_2_BCT_guess = .999
        X_2_L_guess = .5
        xmin = resolution
        xmax = 1 - resolution
        res = least_squares(func, [X_2_BCT_guess, X_2_L_guess], bounds=((xmin, xmin), (xmax, xmax)))
        soln = res.x

        return soln[1]
start = time.time()

print('Plotting the entire phase diagram... Please be patient...')
temp_start = 0 + 273.15  # Minimum temperature to plot in K, converted from °C.
temp_end = 350 + 273.15  # Maximum temperature to plot in K, converted from °C.
temprange = np.arange(temp_start, temp_end, 1)
FCCsolvusVALS = list(map(FCCsolvus, temprange))
FCCsolidusVALS = list(map(FCCsolidus, temprange))
FCCliquidusVALS = list(map(FCCliquidus, temprange))
BCTsolvusVALS = list(map(BCTsolvus, temprange))
BCTsolidusVALS = list(map(BCTsolidus, temprange))
BCTliquidusVALS = list(map(BCTliquidus, temprange))
plt.figure('Phase Diagram', dpi=300)
plt.title(model_name)
plt.plot(list(map(FCCsolvus, temprange)), temprange - 273.15, color='r', label='FCC')
plt.plot(list(map(FCCsolidus, temprange)), temprange - 273.15, color='r')
plt.plot(list(map(BCTsolvus, temprange)), temprange - 273.15, color='purple', label='BCT')
plt.plot(list(map(BCTsolidus, temprange)), temprange - 273.15, color='purple')
plt.plot(list(map(FCCliquidus, temprange)), temprange - 273.15, color='b', label='L')
plt.plot(list(map(BCTliquidus, temprange)), temprange - 273.15, color='b')
plt.plot(np.linspace(FCCsolvus(T_eutectic), BCTsolvus(T_eutectic),100), np.repeat(T_eutectic - 273.15,100), color='black')
plt.ylabel(r'$T$ (°C)')
plt.xlabel(f'$X_\\mathrm{{{comp2}}}$ (mol/mol)')
plt.xlim(0, 1)
plt.ylim(temp_start - 273.15, temp_end - 273.15)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

plt.show()  # This line isn't necessary in Spyder, but it can't hurt.

print(f'This cell took {time.time() - start:.2f} seconds to run.')
start = time.time()

print('Making a super fancy animation...')
framecount = 100
temp_stepsize = math.ceil((temp_end - temp_start) / framecount)
fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(9, 12))


def plotgxcurves(temp_step):
    plottemp = temp_start + (temp_step * temp_stepsize)
    ymin = 1.1 * min(min(DeltaG_mix_FCC(Xrange, plottemp)), min(DeltaG_mix_BCT(Xrange, plottemp)),
                     min(DeltaG_mix_L(Xrange, plottemp)))
    ax1.clear()
    ax2.clear()
    fig.suptitle(model_name + ', $T$ = ' + f"{(plottemp - 273.15):.1f} °C", fontsize=18)
    plt.xlabel(f'$X_\\mathrm{{{comp2}}}$ (mol/mol)')
    plt.xlim(0, 1)
    ax1.set_ylim(ymin, 0)
    ax1.set_ylabel(r'Δ$G_\mathrm{mix}$ (J/mol)')
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    ax2.set_ylim(temp_start - 273.15, temp_end - 273.15)
    ax2.set_ylabel(r'$T$ (°C)')
    ax1.plot(Xrange, DeltaG_mix_FCC(Xrange, plottemp), color='r', label='FCC')
    ax1.plot(Xrange, DeltaG_mix_BCT(Xrange, plottemp), color='purple', label='BCT')
    ax1.plot(Xrange, DeltaG_mix_L(Xrange, plottemp), color='b', label='L')
    ax1.legend(loc='upper center')
    ax2.plot(FCCsolvusVALS, temprange - 273.15, color='r')
    ax2.plot(FCCsolidusVALS, temprange - 273.15, color='r')
    ax2.plot(BCTsolvusVALS, temprange - 273.15, color='purple')
    ax2.plot(BCTsolidusVALS, temprange - 273.15, color='purple')
    ax2.plot(FCCliquidusVALS, temprange - 273.15, color='b')
    ax2.plot(BCTliquidusVALS, temprange - 273.15, color='b')
    ax2.plot([X_2_FCC_eutectic, X_2_BCT_eutectic], [T_eutectic - 273.15, T_eutectic - 273.15], color='black')
    try:
        xleft = FCCsolvus(plottemp)
        xright = BCTsolvus(plottemp)
        yleft = DeltaG_mix_FCC(xleft, plottemp)
        yright = DeltaG_mix_BCT(xright, plottemp)
        ax1.plot([xleft, xright], [yleft, yright], 'k--')  # Tangent line
        ax1.plot([xleft, xleft], [yleft, ymin], 'k--')  # Left line to x-axis
        ax1.plot([xright, xright], [yright, ymin], 'k--')  # Right line to x-axis
        ax2.scatter([xleft, xright], [plottemp - 273.15, plottemp - 273.15], color='k')  # Points on PD
        ax1.text(xleft - 2e-2, (yleft + ymin) / 2, round(xleft, 4), backgroundcolor='white', ha="right")
        ax1.text(xright + 2e-2, (yright + ymin) / 2, round(xright, 4), backgroundcolor='white', ha="left")
    except:
        pass
    try:
        xleft = FCCsolidus(plottemp)
        xright = FCCliquidus(plottemp)
        yleft = DeltaG_mix_FCC(xleft, plottemp)
        yright = DeltaG_mix_L(xright, plottemp)
        ax1.plot([xleft, xright], [yleft, yright], color='black', linestyle='--')  # Tangent line
        ax1.plot([xleft, xleft], [yleft, ymin], color='black', linestyle='--')  # Left line to x-axis
        ax1.plot([xright, xright], [yright, ymin], color='black', linestyle='--')  # Right line to x-axis
        ax2.scatter([xleft, xright], [plottemp - 273.15, plottemp - 273.15], color='k')  # Points on PD
        ax1.text(xleft - 2e-2, (yleft + ymin) / 2, round(xleft, 4), backgroundcolor='white',
                 horizontalalignment="right", verticalalignment="bottom")
        ax1.text(xright + 2e-2, (yright + ymin) / 2, round(xright, 4), backgroundcolor='white',
                 horizontalalignment="left", verticalalignment="top")
    except:
        pass
    try:
        xleft = BCTliquidus(plottemp)
        xright = BCTsolidus(plottemp)
        yleft = DeltaG_mix_L(xleft, plottemp)
        yright = DeltaG_mix_BCT(xright, plottemp)
        ax1.plot([xleft, xright], [yleft, yright], color='black', linestyle='--')  # Tangent line
        ax1.plot([xleft, xleft], [yleft, ymin], color='black', linestyle='--')  # Left line to x-axis
        ax1.plot([xright, xright], [yright, ymin], color='black', linestyle='--')  # Right line to x-axis
        ax2.scatter([xleft, xright], [plottemp - 273.15, plottemp - 273.15], color='k')  # Points on PD
        ax1.text(xleft - 2e-2, (yleft + ymin) / 2, round(xleft, 4), backgroundcolor='white',
                 horizontalalignment="right", verticalalignment="top")
        ax1.text(xright + 2e-2, (yright + ymin) / 2, round(xright, 4), backgroundcolor='white',
                 horizontalalignment="left", verticalalignment="bottom")
    except:
        pass

    return plt
anim = animation.FuncAnimation(fig, plotgxcurves, interval=200, frames=framecount, repeat=False)
def print_progress(current_frame, total_frames):
    percent_done = round((current_frame / total_frames) * 100)
    progress_str = f'Progress: {percent_done}% complete'.ljust(30, ' ')
    print(f'\r{progress_str}', end='', flush=True)
anim.save(model_name + " - GX and PD.gif", progress_callback=print_progress)
print()

print(f'This cell took {time.time() - start:.2f} seconds to run.')

'''start = time.time()

print('Plotting GX curves and tangents at several different temperatures...')
temps_in_C = [25, 100, 180, 200, 300]
temps = [T + 273.15 for T in temps_in_C]


def plotgxcurves(plottemp):
    plt.figure(plottemp, dpi=300)
    ymin = 1.1 * min(min(DeltaG_mix_FCC(Xrange, plottemp)), min(DeltaG_mix_BCT(Xrange, plottemp)),
                     min(DeltaG_mix_L(Xrange, plottemp)))
    plt.title(model_name + ', $T$ = ' + f"{(plottemp - 273.15):.1f} °C\n")
    plt.xlabel(f'$X_\\mathrm{{{comp2}}}$ (mol/mol)')
    plt.xlim(0, 1)
    plt.ylim(ymin, 0)
    plt.ylabel(r'Δ$G_\mathrm{mix}$ (J/mol)')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    plt.plot(Xrange, DeltaG_mix_FCC(Xrange, plottemp), color='r', label='FCC')
    plt.plot(Xrange, DeltaG_mix_BCT(Xrange, plottemp), color='purple', label='BCT')
    plt.plot(Xrange, DeltaG_mix_L(Xrange, plottemp), color='b', label='L')
    plt.legend(loc='upper center')
    try:
        xleft = FCCsolvus(plottemp)
        xright = BCTsolvus(plottemp)
        yleft = DeltaG_mix_FCC(xleft, plottemp)
        yright = DeltaG_mix_BCT(xright, plottemp)
        plt.plot([xleft, xright], [yleft, yright], 'k--')  # Tangent line
        plt.plot([xleft, xleft], [yleft, ymin], 'k--')  # Left line to x-axis
        plt.plot([xright, xright], [yright, ymin], 'k--')  # Right line to x-axis
        plt.text(xleft - 2e-2, (yleft + ymin) / 2, round(xleft, 4), backgroundcolor='white', ha="right")
        plt.text(xright + 2e-2, (yright + ymin) / 2, round(xright, 4), backgroundcolor='white', ha="left")
    except:
        pass
    try:
        xleft = FCCsolidus(plottemp)
        xright = FCCliquidus(plottemp)
        yleft = DeltaG_mix_FCC(xleft, plottemp)
        yright = DeltaG_mix_L(xright, plottemp)
        plt.plot([xleft, xright], [yleft, yright], color='black', linestyle='--')  # Tangent line
        plt.plot([xleft, xleft], [yleft, ymin], color='black', linestyle='--')  # Left line to x-axis
        plt.plot([xright, xright], [yright, ymin], color='black', linestyle='--')  # Right line to x-axis
        plt.text(xleft - 2e-2, (yleft + ymin) / 2, round(xleft, 4), backgroundcolor='white',
                 horizontalalignment="right", verticalalignment="bottom")
        plt.text(xright + 2e-2, (yright + ymin) / 2, round(xright, 4), backgroundcolor='white',
                 horizontalalignment="left", verticalalignment="top")
    except:
        pass
    try:
        xleft = BCTliquidus(plottemp)
        xright = BCTsolidus(plottemp)
        yleft = DeltaG_mix_L(xleft, plottemp)
        yright = DeltaG_mix_BCT(xright, plottemp)
        plt.plot([xleft, xright], [yleft, yright], color='black', linestyle='--')  # Tangent line
        plt.plot([xleft, xleft], [yleft, ymin], color='black', linestyle='--')  # Left line to x-axis
        plt.plot([xright, xright], [yright, ymin], color='black', linestyle='--')  # Right line to x-axis
        plt.text(xleft - 2e-2, (yleft + ymin) / 2, round(xleft, 4), backgroundcolor='white',
                 horizontalalignment="right", verticalalignment="top")
        plt.text(xright + 2e-2, (yright + ymin) / 2, round(xright, 4), backgroundcolor='white',
                 horizontalalignment="left", verticalalignment="bottom")
    except:
        pass

    plt.show()

    return plt


for temp in temps:
    plotgxcurves(temp)

print(f'This cell took {time.time() - start:.2f} seconds to run.')'''