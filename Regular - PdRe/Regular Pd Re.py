import math
import os
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from scipy.misc import derivative


'''abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)'''


comp1 = 'Pd'
comp2 = 'Re'
Tmelt1 = 1555 + 273
Tmelt2 = 3186 + 273

# Eutectic is found to be 1621 C instead of 1650, which we chalk up to the ideal solution model...

model_type = 'Regular Solution Model'

model_name = f'{comp1}-{comp2} - {model_type}'

start = time.time()

print('Setting up model and plotting our first GX curves...')

R = 8.31446261815324
LN = ln = Ln = lN = np.log
resolution = 1e-6

if True:

    # This is better than lambdas with bad styling IMO :)
    def FCC_abs_Pd(T):
        if T < 298.15:
            return np.nan
        if 298.15 <= T < 900:
            return -10204.027+176.076315*T-32.211*T*LN(T)+7.120975E-3*T**2-1.919875E-6*T**3 +168687*T**(-1)
        if 900 <= T < 1828:
            return 917.062+49.659892*T-13.5708*T*LN(T)-7.17522E-3*T**2+0.191115E-6*T**3 -1112465*T**(-1)
        if 1828 <= T < 4000:
            return -67161.018+370.102147*T-54.2067086*T*LN(T)+2.091396E-3*T**2-0.062811E-6*T**3 +18683526*T**(-1)
        if 4000 <= T:
            return np.nan
    
    def HCP_abs_Re(T):
        if T < 298.15:
            return np.nan
        if 298.15 <= T < 1200:
            return -7695.279+128.421589*T-24.348*T*LN(T) -2.53505E-3*T**2+0.192818E-6*T**3+32915*T**(-1)
        if 1200 <= T < 2400:
            return -15775.998+194.667426*T-33.586*T*LN(T)+2.24565E-3*T**2-0.281835E-6*T**3 +1376270*T**(-1)
        if 2400 <= T < 3458:
            return -70882.739+462.110749*T-67.956*T*LN(T)+11.84945E-3*T**2-0.788955E-6*T**3 +18075200*T**(-1)
        if 3458 <= T < 5000:
            return 346325.888-1211.371859*T+140.8316548*T*LN(T)-33.764567E-3*T**2 +1.053726E-6*T**3-134548866*T**(-1)
        if 5000 <= T < 6000:
            return -78564.296+346.997842*T-49.519*T*LN(T)
        if 6000 <= T:
            return np.nan
    
    def FCC_hanging_Pd(T):
        return 0
    
    def FCC_hanging_Re(T):
        if T < 298.15 or T > 6000:
            return np.nan
        return 11000 - 1.5 * T
    
    def HCP_hanging_Pd(T):
        if T < 298.15 or T > 4000:
            return np.nan
        return 2000 + 0.1* T 
    
    def HCP_hanging_Re(T):
        return 0
    
    def Liq_hanging_Pd(T):
        if T < 298.15 or T > 4000:
            return np.nan
        if 298.15 <= T < 600:
            return 1302.731+170.964153*T-32.211*T*LN(T)+7.120975E-3*T**2-1.919875E-6*T**3+168687*T**(-1) - FCC_abs_Pd(T)
        if 600 <= T < 1828:
            return 23405.778-116.918419*T+10.8922031*T*LN(T)-27.266568E-3*T**2+2.430675E-6*T**3-1853674*T**(-1) - FCC_abs_Pd(T)
        if 1828 <= T < 4000:
            return -12373.637+251.416903*T-41.17*T*LN(T) - FCC_abs_Pd(T)
    
    def Liq_hanging_Re(T):
        if T < 298.15 or T > 6000:
            return np.nan
        if 298.15 <= T < 1200:
            return 16125.604+122.076209*T-24.348*T*LN(T)-2.53505E-3*T**2+0.192818E-6*T**3+32915*T**(-1) - HCP_abs_Re(T)
        if 1200 <= T < 2000:
            return 8044.885+188.322047*T-33.586*T*LN(T)+2.24565E-3*T**2-0.281835E-6*T**3+1376270*T**(-1) - HCP_abs_Re(T)
        if 2000 <= T < 3458:
            return 568842.665-2527.838455*T+314.1788975*T*LN(T)-89.39817E-3*T**2+3.92854E-6*T**3-163100987*T**(-1) - HCP_abs_Re(T)
        if 3458 <= T < 6000:
            return -39044.888+335.723691*T-49.519*T*LN(T) - HCP_abs_Re(T)
            

    S_ideal = lambda X_2: R*(X_2*ln(X_2)+(1-X_2)*ln(1-X_2))


# Values hard coded after being found by interaction_parameter_solver.py

W_FCC = 28489.25325276
W_HCP = 45879.43301668
W_L   = 31927.32242207


'''
    w * X1 * X2 
'''

DeltaG_mix_FCC = lambda X_2, T:  W_FCC * X_2 * (1 - X_2) + T*S_ideal(X_2) + (1-X_2)*FCC_hanging_Pd(T) + X_2*FCC_hanging_Re(T)
DeltaG_mix_HCP = lambda X_2, T: W_HCP * X_2 * (1 - X_2) + T*S_ideal(X_2) + (1-X_2)*HCP_hanging_Pd(T) + X_2*HCP_hanging_Re(T)
DeltaG_mix_L = lambda X_2, T: W_L * X_2 * (1 - X_2) + T*S_ideal(X_2) + (1-X_2)*Liq_hanging_Pd(T) + X_2*Liq_hanging_Re(T)

plotres = 1e-4

Xrange = np.arange(plotres, 1 - plotres, plotres)

plottemp = 1000 + 273.15

plt.figure('Random GX Plot',dpi=300)
plt.title(model_name + f' - $T$ = {plottemp - 273.15:.1f} °C\n')
plt.plot(Xrange, DeltaG_mix_FCC(Xrange, plottemp),color='r',label='FCC')
plt.plot(Xrange, DeltaG_mix_HCP(Xrange, plottemp),color='purple',label='HCP')
plt.plot(Xrange, DeltaG_mix_L(Xrange, plottemp),color='b',label='L')
plt.ylabel(r'Δ$G_\mathrm{mix}$ (J/mol)')
plt.xlabel(f'$X_\\mathrm{{{comp2}}}$ (mol/mol)')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

ymin = min(min(DeltaG_mix_FCC(Xrange, plottemp)),min(DeltaG_mix_HCP(Xrange, plottemp)),min(DeltaG_mix_L(Xrange, plottemp)))

plt.ylim(ymin*1.1,0)
plt.xlim(0,1)
plt.legend()
plt.show()

print(f'This cell took {time.time()-start:.2f} seconds to run.')

#%%
#  _____      _            _   _        ____        _                
# | ____|   _| |_ ___  ___| |_(_) ___  / ___|  ___ | |_   _____ _ __ 
# |  _|| | | | __/ _ \/ __| __| |/ __| \___ \ / _ \| \ \ / / _ \ '__|
# | |__| |_| | ||  __/ (__| |_| | (__   ___) | (_) | |\ V /  __/ |   
# |_____\__,_|\__\___|\___|\__|_|\___| |____/ \___/|_| \_/ \___|_|   

# NOTE: This cell is labeled "Eutectic Solver" since that is the three-phase
# reaction on the Pb-Sn diagram that we are modeling first. The general method
# will work for finding eutectoid, peritectic, etc. reactions as well.

start = time.time()

print('Solving for and plotting the eutectic GX curves...')

def ddx(function,dX):
    return lambda X, T: (function(X+dX,T)-function(X,T))/dX
dG_Liq = ddx(DeltaG_mix_L,10**-5)
dG_FCC = ddx(DeltaG_mix_FCC,10**-5)
dG_HCP = ddx(DeltaG_mix_HCP,10**-5)

def func(x):
    X_2_FCC, X_2_L, X_2_HCP, T = x
    
    eqns = [ dG_HCP(X_2_HCP,T) - dG_FCC(X_2_FCC,T), dG_HCP(X_2_HCP,T) - dG_Liq(X_2_L,T),
             DeltaG_mix_HCP(X_2_HCP,T) - X_2_HCP * dG_HCP(X_2_HCP,T) - (DeltaG_mix_L(X_2_L,T) - X_2_L*dG_Liq(X_2_L,T)),
             DeltaG_mix_HCP(X_2_HCP,T) - X_2_HCP * dG_HCP(X_2_HCP,T) - (DeltaG_mix_FCC(X_2_FCC,T) - X_2_FCC*dG_FCC(X_2_FCC,T))]
    
    return eqns

X_2_FCC_guess = .10
X_2_L_guess = .5 
X_2_HCP_guess = .90
T_guess = 1650

xmin = resolution
xmax = 1 - resolution
tempmin = 0
tempmax = np.inf

res = least_squares(func,[X_2_FCC_guess, X_2_L_guess, X_2_HCP_guess, T_guess],bounds = ((xmin,xmin,xmin,tempmin),(xmax,xmax,xmax,tempmax)))
print(res)


eutec_soln = res.x
print(eutec_soln)

X_2_FCC_eutectic = eutec_soln[0]
X_2_L_eutectic = eutec_soln[1]
X_2_HCP_eutectic = eutec_soln[2]
T_eutectic = eutec_soln[3]

plottemp = T_eutectic


plt.figure('Eutectic GX Plot',dpi=300)
plt.title(model_name + f' - $T$ = {plottemp - 273.15:.1f} °C\n')
plt.plot(Xrange, DeltaG_mix_FCC(Xrange, plottemp),color='r',label='FCC')
plt.plot(Xrange, DeltaG_mix_HCP(Xrange, plottemp),color='purple',label='HCP')
plt.plot(Xrange, DeltaG_mix_L(Xrange, plottemp),color='b',label='L')
plt.plot([X_2_FCC_eutectic,X_2_HCP_eutectic],[DeltaG_mix_FCC(X_2_FCC_eutectic, T_eutectic),DeltaG_mix_HCP(X_2_HCP_eutectic, T_eutectic)],color='black',linestyle='--')
plt.ylabel(r'Δ$G_\mathrm{mix}$ (J/mol)')
plt.xlabel(f'$X_\\mathrm{{{comp2}}}$ (mol/mol)')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)


ymin = min(min(DeltaG_mix_FCC(Xrange, plottemp)),min(DeltaG_mix_HCP(Xrange, plottemp)),min(DeltaG_mix_L(Xrange, plottemp)))


plt.ylim(ymin*1.1,0)
plt.xlim(0,1)
plt.legend()
plt.show()

print(f'This cell took {time.time()-start:.2f} seconds to run.')

#%%
#  _____                            _     ____        _                    
# |_   _|_ _ _ __   __ _  ___ _ __ | |_  / ___|  ___ | |_   _____ _ __ ___ 
#   | |/ _` | '_ \ / _` |/ _ \ '_ \| __| \___ \ / _ \| \ \ / / _ \ '__/ __|
#   | | (_| | | | | (_| |  __/ | | | |_   ___) | (_) | |\ V /  __/ |  \__ \
#   |_|\__,_|_| |_|\__, |\___|_| |_|\__| |____/ \___/|_| \_/ \___|_|  |___/
#                  |___/                                                   

print('Defining the tangent solving functions...')


def FCCsolvus(T):
    
    if T<=T_eutectic: #Temperature is below eutectic
            
        def func(x):
            X_2_FCC, X_2_HCP = x
            
            eqns = [ dG_HCP(X_2_HCP,T)-dG_FCC(X_2_FCC,T),
                     DeltaG_mix_HCP(X_2_HCP,T) - X_2_HCP*dG_HCP(X_2_HCP,T) - (DeltaG_mix_FCC(X_2_FCC,T) - X_2_FCC*dG_FCC(X_2_FCC,T))
                ]
            
            return eqns
         

        X_2_FCC_guess = 0.001
        X_2_HCP_guess = 0.999
        
        xmin = resolution
        xmax = 1 - resolution

        res = least_squares(func,[X_2_FCC_guess, X_2_HCP_guess], bounds = ((xmin,xmin),(xmax,xmax)))
        soln = res.x       
         
        return soln[0]

def HCPsolvus(T):
    
    if T<=T_eutectic:
    
        def func(x):
            X_2_FCC, X_2_HCP = x
            
            eqns = [dG_HCP(X_2_HCP,T)-dG_FCC(X_2_FCC,T),
                     DeltaG_mix_HCP(X_2_HCP,T) - X_2_HCP*dG_HCP(X_2_HCP,T) - (DeltaG_mix_FCC(X_2_FCC,T) - X_2_FCC*dG_FCC(X_2_FCC,T))
                ]
            
            return eqns
         
        X_2_FCC_guess = .001
        X_2_HCP_guess = .999
         
        xmin = resolution
        xmax = 1 - resolution

        res = least_squares(func,[X_2_FCC_guess, X_2_HCP_guess], bounds = ((xmin,xmin),(xmax,xmax)))
        soln = res.x       
    
        return soln[1]

def FCCsolidus(T):
    
    if T_eutectic >= T >= Tmelt1:
    
        def func(x):
            X_2_FCC, X_2_L = x
            
            eqns = [dG_Liq(X_2_L,T)-dG_FCC(X_2_FCC,T),
                     DeltaG_mix_L(X_2_L,T) - X_2_L*dG_Liq(X_2_L,T) - (DeltaG_mix_FCC(X_2_FCC,T) - X_2_FCC*dG_FCC(X_2_FCC,T))
                ]
            
            return eqns
         
        X_2_FCC_guess = .01
        X_2_L_guess = .1
         
        xmin = resolution
        xmax = 1 - resolution

        res = least_squares(func,[X_2_FCC_guess, X_2_L_guess], bounds = ((xmin,xmin),(xmax,xmax)))
        soln = res.x         
    
        return soln[0]

def FCCliquidus(T):

    if T_eutectic >= T >= Tmelt1:

        def func(x):
            X_2_FCC, X_2_L = x
            
            eqns = [dG_Liq(X_2_L,T)-dG_FCC(X_2_FCC,T),
                     DeltaG_mix_L(X_2_L,T) - X_2_L*dG_Liq(X_2_L,T) - (DeltaG_mix_FCC(X_2_FCC,T) - X_2_FCC*dG_FCC(X_2_FCC,T))
                ]
            
            return eqns
         
        X_2_FCC_guess = .01
        X_2_L_guess = .1
         
        xmin = resolution
        xmax = 1 - resolution

        res = least_squares(func,[X_2_FCC_guess, X_2_L_guess], bounds = ((xmin,xmin),(xmax,xmax)))
        soln = res.x      
    
        return soln[1]

def HCPsolidus(T):

    if T_eutectic <= T <= Tmelt2:

        def func(x):
            X_2_HCP, X_2_L = x
            
            eqns = [dG_Liq(X_2_L,T)-dG_HCP(X_2_HCP,T),
                     DeltaG_mix_L(X_2_L,T) - X_2_L*dG_Liq(X_2_L,T) - (DeltaG_mix_HCP(X_2_HCP,T) - X_2_HCP*dG_HCP(X_2_HCP,T))
                ]
            
            return eqns
         
        X_2_HCP_guess = .999
        X_2_L_guess = .5
         
        xmin = resolution
        xmax = 1 - resolution

        res = least_squares(func,[X_2_HCP_guess, X_2_L_guess], bounds = ((xmin,xmin),(xmax,xmax)))
        soln = res.x       
    
        return soln[0]

def HCPliquidus(T):

    if T_eutectic <= T <= Tmelt2:

        def func(x):
            X_2_HCP, X_2_L = x
            
            eqns = [dG_Liq(X_2_L,T)-dG_HCP(X_2_HCP,T),
                     DeltaG_mix_L(X_2_L,T) - X_2_L*dG_Liq(X_2_L,T) - (DeltaG_mix_HCP(X_2_HCP,T) - X_2_HCP*dG_HCP(X_2_HCP,T))
                ]
            
            return eqns
         
        X_2_HCP_guess = .999
        X_2_L_guess = .5
         
        xmin = resolution
        xmax = 1 - resolution

        res = least_squares(func,[X_2_HCP_guess, X_2_L_guess], bounds = ((xmin,xmin),(xmax,xmax)))
        soln = res.x      
    
        return soln[1]

#%%
 #  ____  _       _     _   _                                            
 # |  _ \| | ___ | |_  | |_| |__   ___                                   
 # | |_) | |/ _ \| __| | __| '_ \ / _ \                                  
 # |  __/| | (_) | |_  | |_| | | |  __/                                  
 # |_|__ |_|\___/ \__|  \__|_| |_|\___|_                                 
 # |  _ \| |__   __ _ ___  ___  |  _ \(_) __ _  __ _ _ __ __ _ _ __ ___  
 # | |_) | '_ \ / _` / __|/ _ \ | | | | |/ _` |/ _` | '__/ _` | '_ ` _ \ 
 # |  __/| | | | (_| \__ \  __/ | |_| | | (_| | (_| | | | (_| | | | | | |
 # |_|   |_| |_|\__,_|___/\___| |____/|_|\__,_|\__, |_|  \__,_|_| |_| |_|
 #                                             |___/                     

start = time.time()

print('Plotting the entire phase diagram... Please be patient...')

temp_start = 1000 + 273.15 # Minimum temperature to plot in K, converted from °C.
temp_end = 3200 + 273.15 # Maximum temperature to plot in K, converted from °C.

temprange = np.arange(temp_start, temp_end, 1)


FCCsolvusVALS = list(map(FCCsolvus,temprange))
FCCsolidusVALS = list(map(FCCsolidus,temprange))
FCCliquidusVALS = list(map(FCCliquidus,temprange))
HCPsolvusVALS = list(map(HCPsolvus,temprange))
HCPsolidusVALS = list(map(HCPsolidus,temprange))
HCPliquidusVALS = list(map(HCPliquidus,temprange))


plt.figure('Phase Diagram',dpi=300)


plt.title(model_name)


plt.plot(list(map(FCCsolvus,temprange)),temprange - 273.15,color='r',label='FCC')
plt.plot(list(map(FCCsolidus,temprange)),temprange - 273.15,color='r')


plt.plot(list(map(HCPsolvus,temprange)),temprange - 273.15,color='purple',label='HCP')
plt.plot(list(map(HCPsolidus,temprange)),temprange - 273.15,color='purple')


plt.plot(list(map(FCCliquidus,temprange)),temprange - 273.15,color='b',label='L')
plt.plot(list(map(HCPliquidus,temprange)),temprange - 273.15,color='b')


plt.plot([HCPsolvus(T_eutectic),HCPliquidus(T_eutectic)],[T_eutectic - 273.15,T_eutectic - 273.15],color='black')

plt.ylabel(r'$T$ (°C)')
plt.xlabel(f'$X_\\mathrm{{{comp2}}}$ (mol/mol)')

plt.xlim(0,1)
plt.ylim(temp_start - 273.15,temp_end - 273.15)


plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

plt.show()

print(f'This cell took {time.time()-start:.2f} seconds to run.')

#%%
 #  ____  _       _     _   _                                               
 # |  _ \| | ___ | |_  | |_| |__   ___                                      
 # | |_) | |/ _ \| __| | __| '_ \ / _ \                                     
 # |  __/| | (_) | |_  | |_| | | |  __/                                     
 # |_|__ |_|\___/ \__|  \__|_| |_|\___|_                                    
 # |  _ \| |__   __ _ ___  ___  |  _ \(_) __ _  __ _ _ __ __ _ _ __ ___     
 # | |_) | '_ \ / _` / __|/ _ \ | | | | |/ _` |/ _` | '__/ _` | '_ ` _ \    
 # |  __/| | | | (_| \__ \  __/ | |_| | | (_| | (_| | | | (_| | | | | | |   
 # |_|   |_| |_|\__,_|___/\___| |____/|_|\__,_|\__, |_|  \__,_|_| |_| |_|   
 #   ___      ______  __     _          _      |___/      _   _             
 #  ( _ )    / ___\ \/ /    / \   _ __ (_)_ __ ___   __ _| |_(_) ___  _ __  
 #  / _ \/\ | |  _ \  /    / _ \ | '_ \| | '_ ` _ \ / _` | __| |/ _ \| '_ \ 
 # | (_>  < | |_| |/  \   / ___ \| | | | | | | | | | (_| | |_| | (_) | | | |
 #  \___/\/  \____/_/\_\ /_/   \_\_| |_|_|_| |_| |_|\__,_|\__|_|\___/|_| |_|


start = time.time()

print('Making a super fancy animation...')


framecount = 100
temp_stepsize = math.ceil((temp_end - temp_start)/framecount)


fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(9,12))

def plotgxcurves(temp_step):
    
    plottemp = temp_start + (temp_step * temp_stepsize)
    
    ymin = 1.1 * min(min(DeltaG_mix_FCC(Xrange, plottemp)),min(DeltaG_mix_HCP(Xrange, plottemp)),min(DeltaG_mix_L(Xrange, plottemp)))
        
    ax1.clear()
    ax2.clear()

    fig.suptitle(model_name + ', $T$ = '+f"{(plottemp - 273.15):.1f} °C", fontsize=18)

    plt.xlabel(f'$X_\\mathrm{{{comp2}}}$ (mol/mol)')
    plt.xlim(0,1)

    ax1.set_ylim(ymin,0)
    ax1.set_ylabel(r'Δ$G_\mathrm{mix}$ (J/mol)')
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
            
    ax2.set_ylim(temp_start - 273.15,temp_end - 273.15)
    ax2.set_ylabel(r'$T$ (°C)')
    
    ax1.plot(Xrange, DeltaG_mix_FCC(Xrange, plottemp),color='r',label='FCC')
    ax1.plot(Xrange, DeltaG_mix_HCP(Xrange, plottemp),color='purple',label='HCP')
    ax1.plot(Xrange, DeltaG_mix_L(Xrange, plottemp),color='b',label='L')
    ax1.legend(loc='upper center')
    
    ax2.plot(FCCsolvusVALS,temprange - 273.15,color='r')
    ax2.plot(FCCsolidusVALS,temprange - 273.15,color='r')
    ax2.plot(HCPsolvusVALS,temprange - 273.15,color='purple')
    ax2.plot(HCPsolidusVALS,temprange - 273.15,color='purple')
    ax2.plot(FCCliquidusVALS,temprange - 273.15,color='b')
    ax2.plot(HCPliquidusVALS,temprange - 273.15,color='b')
    ax2.plot([X_2_L_eutectic,X_2_HCP_eutectic],[T_eutectic - 273.15,T_eutectic - 273.15],color='black')

    try:
        xleft = FCCsolvus(plottemp)
        xright = HCPsolvus(plottemp)
        yleft = DeltaG_mix_FCC(xleft, plottemp)
        yright = DeltaG_mix_HCP(xright, plottemp)
        ax1.plot([xleft,xright],[yleft,yright],'k--') # Tangent line
        ax1.plot([xleft,xleft],[yleft,ymin],'k--') # Left line to x-axis
        ax1.plot([xright,xright],[yright,ymin],'k--') # Right line to x-axis
        ax2.scatter([xleft,xright],[plottemp - 273.15,plottemp - 273.15],color='k') # Points on PD
        ax1.text(xleft - 2e-2,(yleft+ymin)/2,round(xleft,4), backgroundcolor = 'white',ha="right")
        ax1.text(xright + 2e-2,(yright+ymin)/2,round(xright,4), backgroundcolor = 'white',ha="left")
    except:
        pass

    try:
        xleft = FCCsolidus(plottemp)
        xright = FCCliquidus(plottemp)
        yleft = DeltaG_mix_FCC(xleft, plottemp)
        yright = DeltaG_mix_L(xright, plottemp)
        ax1.plot([xleft,xright],[yleft,yright],color='black',linestyle='--') # Tangent line
        ax1.plot([xleft,xleft],[yleft,ymin],color='black',linestyle='--') # Left line to x-axis
        ax1.plot([xright,xright],[yright,ymin],color='black',linestyle='--') # Right line to x-axis
        ax2.scatter([xleft,xright],[plottemp - 273.15,plottemp - 273.15],color='k') # Points on PD
        ax1.text(xleft - 2e-2,(yleft+ymin)/2,round(xleft,4), backgroundcolor = 'white',horizontalalignment="right",verticalalignment="bottom")
        ax1.text(xright + 2e-2,(yright+ymin)/2,round(xright,4), backgroundcolor = 'white',horizontalalignment="left",verticalalignment="top")
    except:
        pass

    try:
        xleft = HCPliquidus(plottemp)
        xright = HCPsolidus(plottemp)
        yleft = DeltaG_mix_L(xleft, plottemp)
        yright = DeltaG_mix_HCP(xright, plottemp)
        ax1.plot([xleft,xright],[yleft,yright],color='black',linestyle='--') # Tangent line
        ax1.plot([xleft,xleft],[yleft,ymin],color='black',linestyle='--') # Left line to x-axis
        ax1.plot([xright,xright],[yright,ymin],color='black',linestyle='--') # Right line to x-axis
        ax2.scatter([xleft,xright],[plottemp - 273.15,plottemp - 273.15],color='k') # Points on PD
        ax1.text(xleft - 2e-2,(yleft+ymin)/2,round(xleft,4), backgroundcolor = 'white',horizontalalignment="right",verticalalignment="top")
        ax1.text(xright + 2e-2,(yright+ymin)/2,round(xright,4), backgroundcolor = 'white',horizontalalignment="left",verticalalignment="bottom")
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

print(f'This cell took {time.time()-start:.2f} seconds to run.')

#%%
 #  ____  _       _      ______  __   ____                          
 # |  _ \| | ___ | |_   / ___\ \/ /  / ___|   _ _ ____   _____  ___ 
 # | |_) | |/ _ \| __| | |  _ \  /  | |  | | | | '__\ \ / / _ \/ __|
 # |  __/| | (_) | |_  | |_| |/  \  | |__| |_| | |   \ V /  __/\__ \
 # |_|   |_|\___/_\__|  \____/_/\_\  \____\__,_|_|    \_/ \___||___/
 #   __ _| |_  / ___| _ __   ___  ___(_)/ _(_) ___                  
 #  / _` | __| \___ \| '_ \ / _ \/ __| | |_| |/ __|                 
 # | (_| | |_   ___) | |_) |  __/ (__| |  _| | (__                  
 #  \__,_|\__| |____/| .__/ \___|\___|_|_| |_|\___|                 
 #  _____            |_|                    _                       
 # |_   _|__ _ __ ___  _ __   ___ _ __ __ _| |_ _   _ _ __ ___  ___ 
 #   | |/ _ \ '_ ` _ \| '_ \ / _ \ '__/ _` | __| | | | '__/ _ \/ __|
 #   | |  __/ | | | | | |_) |  __/ | | (_| | |_| |_| | | |  __/\__ \
 #   |_|\___|_| |_| |_| .__/ \___|_|  \__,_|\__|\__,_|_|  \___||___/
 #                    |_|                                           

start = time.time()

print('Plotting GX curves and tangents at several different temperatures...')

# Define plot temperatures in °C. You can add/remove as many temperatures as 
# you want from the list.
temps_in_C = [1000]

# Define plot temperature in K.
temps = [T + 273.15 for T in temps_in_C]

def plotgxcurves(plottemp):
    
    plt.figure(plottemp,dpi=300)
    
    # Find the minimum value on ALL GX curves to determine y-axis limits. This
    # is necessary to produce GX plots that are easy to read among the
    # interesting range (i.e., below DeltaG_mix = 0).
    ymin = 1.1 * min(min(DeltaG_mix_FCC(Xrange, plottemp)),min(DeltaG_mix_HCP(Xrange, plottemp)),min(DeltaG_mix_L(Xrange, plottemp)))

    # Give the overall figure a title.
    plt.title(model_name + ', $T$ = '+f"{(plottemp - 273.15):.1f} °C\n")

    # Format x-axis.
    plt.xlabel(f'$X_\\mathrm{{{comp2}}}$ (mol/mol)')
    plt.xlim(0,1)

    # Format y-axis.
    plt.ylim(ymin,0)
    plt.ylabel(r'Δ$G_\mathrm{mix}$ (J/mol)')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    
    # Plot the GX curves.
    plt.plot(Xrange, DeltaG_mix_FCC(Xrange, plottemp),color='r',label='FCC')
    plt.plot(Xrange, DeltaG_mix_HCP(Xrange, plottemp),color='purple',label='HCP')
    plt.plot(Xrange, DeltaG_mix_L(Xrange, plottemp),color='b',label='L')
    plt.legend(loc='upper center')
    
    # Plot the FCC-HCP tangent.
    try:
        xleft = FCCsolvus(plottemp)
        xright = HCPsolvus(plottemp)
        yleft = DeltaG_mix_FCC(xleft, plottemp)
        yright = DeltaG_mix_HCP(xright, plottemp)
        plt.plot([xleft,xright],[yleft,yright],'k--') # Tangent line
        plt.plot([xleft,xleft],[yleft,ymin],'k--') # Left line to x-axis
        plt.plot([xright,xright],[yright,ymin],'k--') # Right line to x-axis
        plt.text(xleft - 2e-2,(yleft+ymin)/2,round(xleft,4), backgroundcolor = 'white',ha="right")
        plt.text(xright + 2e-2,(yright+ymin)/2,round(xright,4), backgroundcolor = 'white',ha="left")
    except:
        pass

    # Plot the FCC solidus/liquidus tangent.
    try:
        xleft = FCCsolidus(plottemp)
        xright = FCCliquidus(plottemp)
        yleft = DeltaG_mix_FCC(xleft, plottemp)
        yright = DeltaG_mix_L(xright, plottemp)
        plt.plot([xleft,xright],[yleft,yright],color='black',linestyle='--') # Tangent line
        plt.plot([xleft,xleft],[yleft,ymin],color='black',linestyle='--') # Left line to x-axis
        plt.plot([xright,xright],[yright,ymin],color='black',linestyle='--') # Right line to x-axis
        plt.text(xleft - 2e-2,(yleft+ymin)/2,round(xleft,4), backgroundcolor = 'white',horizontalalignment="right",verticalalignment="bottom")
        plt.text(xright + 2e-2,(yright+ymin)/2,round(xright,4), backgroundcolor = 'white',horizontalalignment="left",verticalalignment="top")
    except:
        pass

    # Plot the HCP solidus/liquidus tangent.
    try:
        xleft = HCPliquidus(plottemp)
        xright = HCPsolidus(plottemp)
        yleft = DeltaG_mix_L(xleft, plottemp)
        yright = DeltaG_mix_HCP(xright, plottemp)
        plt.plot([xleft,xright],[yleft,yright],color='black',linestyle='--') # Tangent line
        plt.plot([xleft,xleft],[yleft,ymin],color='black',linestyle='--') # Left line to x-axis
        plt.plot([xright,xright],[yright,ymin],color='black',linestyle='--') # Right line to x-axis
        plt.text(xleft - 2e-2,(yleft+ymin)/2,round(xleft,4), backgroundcolor = 'white',horizontalalignment="right",verticalalignment="top")
        plt.text(xright + 2e-2,(yright+ymin)/2,round(xright,4), backgroundcolor = 'white',horizontalalignment="left",verticalalignment="bottom")
    except:
        pass

    plt.show()

    return plt

for temp in temps:
    plotgxcurves(temp)

print(f'This cell took {time.time()-start:.2f} seconds to run.')