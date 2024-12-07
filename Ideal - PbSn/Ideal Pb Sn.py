# This is where we import all libraries, modules, and functions we will use in
# the code below.
import math
import os
import numpy as np
from scipy.optimize import fsolve # Could be useful to test some of your code.
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from scipy.misc import derivative

# These lines change the working directory to where the .py file is saved. 
# This may not be necessary depending on how you have your environment
# configured, but I often do this so I know that any exported files will be 
# saved in the same folder as the .py file.
'''abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)'''

# Define the components in the system. I've done this so I can reuse the code
# for lots of binary systems without having to manually go through and change
# all the variable names below. I just change them here!
comp1 = 'Pb' # Obviously, this will need to be changed for different systems.
comp2 = 'Sn' # Obviously, this will need to be changed for different systems.
Tmelt1 = 327.46 + 273
Tmelt2 = 231.93 + 273

# Define what type of model you are using for this code. Again, this just helps
# keep things tidy without having to go through and manually rename a bunch of
# stuff below. Make sure to change this as necessary if you want your plots
# and exported files named appropriately.
model_type = 'Ideal Solution Model' # Again, this may need to be changed.

# Name this model, which will be used to title plots and name exported files
# below.
model_name = f'{comp1}-{comp2} - {model_type}'

# Define starting time of code to determine how long the calculations are 
# taking. I do this out of habit for every cell in my code so I can gauge if 
# any optimiziations I am implementing are having a profound effect on the 
# speed of the code.
start = time.time()

# This provides an indication of what cell we are running.
print('Setting up model and plotting our first GX curves...')

# Define gas constant in J/mol-K to an obnoxious number of sig figs (why not?).
R = 8.31446261815324

# This line assigns aliases for np.log (i.e., numpy.log) so we can use
# more conventional notation, such as LN(x) instead of np.log(x). This is 
# especially useful if we are copying and pasting expressions from TDB files
# that identify natural logs as LN().
LN = ln = Ln = lN = np.log

# Define the resolution with repsect to X_2 for all numerical calculations.
# Using a coarse resolution will compromise the precision of your calculations.
# Using an extremely fine resolution will not only make your code slow, but
# can also lead to calculation errors due to issues with machine precision. The
# resolution I've defined for you here should be good.
resolution = 1e-6

### "if True" for cleanliness
if True:
    ## The following taken from TDB File correct such that phase at STP is considered at G_0 = 0
    ## As no equations were given for below STP temps, we assume it is close enough till 273 K
    FCC_abs_Pb = lambda T: -10531.095+154.243182*T-32.4913959*T*LN(T)+.00154613*T**2+8.054E+25*T**(-9)
    
    #^^^ Only for above 600.61 as that is all that is needed. vvv similar
    
    BCT_abs_Sn = lambda T: (-5855.135+65.443315*T-15.961*T*LN(T)
                                  -.0188702*T**2+3.121167E-06*T**3
                                  -61960*T**(-1) if T<505.078
                                else 2524.724+4.005269*T-8.2590486*T*LN(T)
                                  -.016814429*T**2+2.623131E-06*T**3
                                  -1081244*T**(-1)-1.2307E+25*T**(-9))

    FCC_hanging_Pb = lambda T: 0
    FCC_hanging_Sn = lambda T: 5510-8.46*T ## THESE ARE WRONG!!! They need + BCT_abs_Sn - FCC_abs_Pb ## NO they aren't, idiot, at 100% Sn the reference is BCT Sn which is what the equation says
    BCT_hanging_Pb = lambda T: (489+3.52*T)
    BCT_hanging_Sn = lambda T: 0
    #the else term includes the FCC absolute G as it was not given in reference to it already
    Liq_hanging_Pb = lambda T: ((4672.124-7.750683*T-60.19E-20*T**7) if (T<600.61)
                                     else -5677.958+146.176046*T-32.4913959*T*LN(T)+1.54613E-3*T**2-FCC_abs_Pb(T))
    Liq_hanging_Sn = lambda T: (7103.092-14.087767*T+147.031E-20*T**7 if T<505.078
                                else 9496.31-9.809114*T-8.2590486*T*LN(T)
                                  -16.814429E-3*T**2+2.623131E-6*T**3
                                  -1081244*T**(-1)-BCT_abs_Sn(T))
    
    

    S_ideal = lambda X_2: R*(X_2*ln(X_2)+(1-X_2)*ln(1-X_2))

# Define the full mixing equations for each phase
DeltaG_mix_FCC = lambda X_2, T:  T*S_ideal(X_2) + (1-X_2)*FCC_hanging_Pb(T) + X_2*FCC_hanging_Sn(T)
DeltaG_mix_BCT = lambda X_2, T: T*S_ideal(X_2) + (1-X_2)*BCT_hanging_Pb(T) + X_2*BCT_hanging_Sn(T)
DeltaG_mix_L = lambda X_2, T: T*S_ideal(X_2) + (1-X_2)*Liq_hanging_Pb(T) + X_2*Liq_hanging_Sn(T)

# Now that we have our DeltaG_mix equations all defined, let's test our
# code by plotting the GX curves at a specific temperature. If your equations
# and code are properly defined, the code below should produce a plot showing
# a red curve for FCC, a purple curve for BCT, and a blue curve for L.

# Define plot resolution. Using the same resolution used for numerical 
# calculations is unnecessarily high and, therefore, slow for this purpose.
plotres = 1e-4

# Define the compositional range for plotting.
Xrange = np.arange(plotres, 1 - plotres, plotres)

# Define the temperature in K for plotting our first GX curves. I've chosen a 
# convenient temperature to compare you model against the project instructions.
plottemp = 170 + 273.15

# Plot the GX curves at "plottemp" as a sanity check for the equations you've
# defined above for DeltaG_mix of the various phases.
plt.figure('Random GX Plot',dpi=300)
plt.title(model_name + f' - $T$ = {plottemp - 273.15:.1f} °C\n')
plt.plot(Xrange, DeltaG_mix_FCC(Xrange, plottemp),color='r',label='FCC')
plt.plot(Xrange, DeltaG_mix_BCT(Xrange, plottemp),color='purple',label='BCT')
plt.plot(Xrange, DeltaG_mix_L(Xrange, plottemp),color='b',label='L')
plt.ylabel(r'Δ$G_\mathrm{mix}$ (J/mol)')
plt.xlabel(f'$X_\\mathrm{{{comp2}}}$ (mol/mol)')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

# Find the minimum value on all GX curves to determine optimal y-axis limits.
ymin = min(min(DeltaG_mix_FCC(Xrange, plottemp)),min(DeltaG_mix_BCT(Xrange, plottemp)),min(DeltaG_mix_L(Xrange, plottemp)))

# Set the x- and y-axis limits so our plot is reasonably zoomed in on the 
# important regions.
plt.ylim(ymin*1.1,0)
plt.xlim(0,1)
plt.legend()
plt.show() # This line isn't necessary in Spyder, but it can't hurt.

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

# This is the function that we will pass to scipy.optimize to solve for the
# eutectic temperature and compositions. REMEMBER: Each "equation" must be 
# given as an expression that defines a root (i.e., an expression equal to 
# zero). For example, if one of the equations you are attempting to solve is
# z = x + y, you would enter z - x - y to represent that equation in the 
# "eqns" list within "func".
def ddx(function,dX):
    return lambda X, T: (function(X+dX,T)-function(X,T))/dX
dG_Liq = ddx(DeltaG_mix_L,10**-5)
dG_FCC = ddx(DeltaG_mix_FCC,10**-5)
dG_BCT = ddx(DeltaG_mix_BCT,10**-5)

def func(x):
    X_2_FCC, X_2_L, X_2_BCT, T = x
    
    eqns = [ dG_BCT(X_2_BCT,T) - dG_FCC(X_2_FCC,T), dG_BCT(X_2_BCT,T) - dG_Liq(X_2_L,T),
             DeltaG_mix_BCT(X_2_BCT,T) - X_2_BCT * dG_BCT(X_2_BCT,T) - (DeltaG_mix_L(X_2_L,T) - X_2_L*dG_Liq(X_2_L,T)),
             DeltaG_mix_BCT(X_2_BCT,T) - X_2_BCT * dG_BCT(X_2_BCT,T) - (DeltaG_mix_FCC(X_2_FCC,T) - X_2_FCC*dG_FCC(X_2_FCC,T))]
    
    return eqns

# Starting guesses for each value are needed for any numerical solver to
# iterate to a solution (i.e., it needs a starting point). While you obviously
# don't need to be very precise or accurate with your initial guess, solvers
# work better with better guesses. Therefore, if you are getting convergence
# errors or garbage answers, try changing your initial guesses to something
# fairly close to what you expect the correct answer to be. IMPORTANT: Make
# sure your initial guesses fall within the domain that the solution will fall
# within (i.e., if you know your answer must be between 0 and 1, don't provide
# any initial guesses greater than 1 or less than 0). Additionally, if you are
# solving for multiple variables, make sure your guesses are in the same order
# that you expect the solutions to be in. For example, if you expect that the
# solution for x will be greater than y, make sure that your initial guess for 
# x is greater than your initial guess for y.
X_2_FCC_guess = .5
X_2_L_guess = .5 #"INSERT YOUR CODE HERE"
X_2_BCT_guess = .5
T_guess = 1000
# These variables will be used to define the bounds that we know the solutions
# must fall within to be physically meaningful. Note: We aren't setting the
# bounds of X_2 as 0 and 1 because either of those values will result in a log
# error in the ideal Gibbs free energy equation. Instead, we tell it to keep
# the solutions between a value very close to 0 and a value very close to 1.
xmin = resolution
xmax = 1 - resolution
tempmin = 0
tempmax = np.inf

# This is where the magic happens (i.e., the equations are solved). I like to
# print out the entire result in case I am getting weird answers. If you want
# to know what all the outputs in "res" are to help you troubleshoot issues
# with your solver, ask ChatGPT. :)
res = least_squares(func,[X_2_FCC_guess, X_2_L_guess, X_2_BCT_guess, T_guess],bounds = ((xmin,xmin,xmin,tempmin),(xmax,xmax,xmax,tempmax)))
print(res)

# This defines a new list that contains the outputs in "res" that we really
# care about. The solution to the system of equations!
eutec_soln = res.x
print(eutec_soln)

# Extract individual values from the solution list to be used later.
X_2_FCC_eutectic = eutec_soln[0]
X_2_L_eutectic = eutec_soln[1]
X_2_BCT_eutectic = eutec_soln[2]
T_eutectic = eutec_soln[3]

# Change "plottemp" to our newly-determined eutectic temperature to graphically
# check in our solution makes sense.
plottemp = T_eutectic

# Plot the GX curves at the eutectic temperature as a sanity check.
plt.figure('Eutectic GX Plot',dpi=300)
plt.title(model_name + f' - $T$ = {plottemp - 273.15:.1f} °C\n')
plt.plot(Xrange, DeltaG_mix_FCC(Xrange, plottemp),color='r',label='FCC')
plt.plot(Xrange, DeltaG_mix_BCT(Xrange, plottemp),color='purple',label='BCT')
plt.plot(Xrange, DeltaG_mix_L(Xrange, plottemp),color='b',label='L')
plt.plot([X_2_FCC_eutectic,X_2_BCT_eutectic],[DeltaG_mix_FCC(X_2_FCC_eutectic, T_eutectic),DeltaG_mix_BCT(X_2_BCT_eutectic, T_eutectic)],color='black',linestyle='--')
plt.ylabel(r'Δ$G_\mathrm{mix}$ (J/mol)')
plt.xlabel(f'$X_\\mathrm{{{comp2}}}$ (mol/mol)')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

# Find the minimum value on all GX curves to determine optimal y-axis limits.
ymin = min(min(DeltaG_mix_FCC(Xrange, plottemp)),min(DeltaG_mix_BCT(Xrange, plottemp)),min(DeltaG_mix_L(Xrange, plottemp)))

# Set the x- and y-axis limits so our plot is reasonably zoomed in on the 
# important regions.
plt.ylim(ymin*1.1,0)
plt.xlim(0,1)
plt.legend()
plt.show() # This line isn't necessary in Spyder, but it can't hurt.

print(f'This cell took {time.time()-start:.2f} seconds to run.')

#%%
#  _____                            _     ____        _                    
# |_   _|_ _ _ __   __ _  ___ _ __ | |_  / ___|  ___ | |_   _____ _ __ ___ 
#   | |/ _` | '_ \ / _` |/ _ \ '_ \| __| \___ \ / _ \| \ \ / / _ \ '__/ __|
#   | | (_| | | | | (_| |  __/ | | | |_   ___) | (_) | |\ V /  __/ |  \__ \
#   |_|\__,_|_| |_|\__, |\___|_| |_|\__| |____/ \___/|_| \_/ \___|_|  |___/
#                  |___/                                                   

print('Defining the tangent solving functions...')

# This function solves for and returns compositional values for the FCC solvus
# curve on the phase diagram.
def FCCsolvus(T):
    
    # If statement used to confine the solver only to temperatures at which we
    # know a common tangent must exist. This prevents errors that could stop 
    # the code prematurely.
    if T<T_eutectic: #Temperature is below eutectic
            
        def func(x):
            X_2_FCC, X_2_BCT = x
            
            eqns = [ dG_BCT(X_2_BCT,T)-dG_FCC(X_2_FCC,T),
                     DeltaG_mix_BCT(X_2_BCT,T) - X_2_BCT*dG_BCT(X_2_BCT,T) - (DeltaG_mix_FCC(X_2_FCC,T) - X_2_FCC*dG_FCC(X_2_FCC,T))
                ]
            
            return eqns
         
        # Starting guesses for each value.
        X_2_FCC_guess = 0.001 # Giving you a helpful recommendation here.
        X_2_BCT_guess = 0.999 # Giving you a helpful recommendation here.
        
        # Set the lower and upper bounds for the solver (i.e., avoid it trying 
        # non-physical values that could produce errors).
        xmin = resolution
        xmax = 1 - resolution

        # Solve the system of equations.
        res = least_squares(func,[X_2_FCC_guess, X_2_BCT_guess], bounds = ((xmin,xmin),(xmax,xmax)))
        soln = res.x       
         
        return soln[0]

def BCTsolvus(T): # <--- Solving twice is so inefficient but alas I am lazy
    
    # If statement used to confine the solver only to temperatures at which we
    # know a common tangent must exist. This prevents errors that could stop 
    # the code prematurely.  
    if T<T_eutectic:
    
        def func(x):
            X_2_FCC, X_2_BCT = x
            
            eqns = [dG_BCT(X_2_BCT,T)-dG_FCC(X_2_FCC,T),
                     DeltaG_mix_BCT(X_2_BCT,T) - X_2_BCT*dG_BCT(X_2_BCT,T) - (DeltaG_mix_FCC(X_2_FCC,T) - X_2_FCC*dG_FCC(X_2_FCC,T))
                ]
            
            return eqns
         
        # Starting guesses for each value.
        X_2_FCC_guess = .001
        X_2_BCT_guess = .999
         
        # Set the lower and upper bounds for the solver (i.e., avoid it trying 
        # non-physical values that could produce errors).
        xmin = resolution
        xmax = 1 - resolution

        # Solve the system of equations.
        res = least_squares(func,[X_2_FCC_guess, X_2_BCT_guess], bounds = ((xmin,xmin),(xmax,xmax)))
        soln = res.x       
    
        return soln[1]

def FCCsolidus(T):
    
    # If statement used to confine the solver only to temperatures at which we
    # know a common tangent must exist. This prevents errors that could stop 
    # the code prematurely.
    if T_eutectic <= T <= Tmelt1:
    
        def func(x):
            X_2_FCC, X_2_L = x
            
            eqns = [dG_Liq(X_2_L,T)-dG_FCC(X_2_FCC,T),
                     DeltaG_mix_L(X_2_L,T) - X_2_L*dG_Liq(X_2_L,T) - (DeltaG_mix_FCC(X_2_FCC,T) - X_2_FCC*dG_FCC(X_2_FCC,T))
                ]
            
            return eqns
         
        # Starting guesses for each value.
        X_2_FCC_guess = .01
        X_2_L_guess = .5
         
        # Set the lower and upper bounds for the solver (i.e., avoid it trying 
        # non-physical values that could produce errors).
        xmin = resolution
        xmax = 1 - resolution

        # Solve the system of equations.
        res = least_squares(func,[X_2_FCC_guess, X_2_L_guess], bounds = ((xmin,xmin),(xmax,xmax)))
        soln = res.x         
    
        return soln[0]

def FCCliquidus(T):

    # If statement used to confine the solver only to temperatures at which we
    # know a common tangent must exist. This prevents errors that could stop 
    # the code prematurely.
    if T_eutectic <= T <= Tmelt1:   #Might need to include melting temp too

        def func(x):
            X_2_FCC, X_2_L = x
            
            eqns = [dG_Liq(X_2_L,T)-dG_FCC(X_2_FCC,T),
                     DeltaG_mix_L(X_2_L,T) - X_2_L*dG_Liq(X_2_L,T) - (DeltaG_mix_FCC(X_2_FCC,T) - X_2_FCC*dG_FCC(X_2_FCC,T))
                ]
            
            return eqns
         
        # Starting guesses for each value.
        X_2_FCC_guess = .01
        X_2_L_guess = .5
         
        # Set the lower and upper bounds for the solver (i.e., avoid it trying 
        # non-physical values that could produce errors).
        xmin = resolution
        xmax = 1 - resolution

        # Solve the system of equations.
        res = least_squares(func,[X_2_FCC_guess, X_2_L_guess], bounds = ((xmin,xmin),(xmax,xmax)))
        soln = res.x      
    
        return soln[1]

def BCTsolidus(T):

    # If statement used to confine the solver only to temperatures at which we
    # know a common tangent must exist. This prevents errors that could stop 
    # the code prematurely.
    if T_eutectic <= T <= Tmelt2:

        def func(x):
            X_2_BCT, X_2_L = x
            
            eqns = [dG_Liq(X_2_L,T)-dG_BCT(X_2_BCT,T),
                     DeltaG_mix_L(X_2_L,T) - X_2_L*dG_Liq(X_2_L,T) - (DeltaG_mix_BCT(X_2_BCT,T) - X_2_BCT*dG_BCT(X_2_BCT,T))
                ]
            
            return eqns
         
        # Starting guesses for each value.
        X_2_BCT_guess = .999
        X_2_L_guess = .5
         
        # Set the lower and upper bounds for the solver (i.e., avoid it trying 
        # non-physical values that could produce errors).
        xmin = resolution
        xmax = 1 - resolution

        # Solve the system of equations.
        res = least_squares(func,[X_2_BCT_guess, X_2_L_guess], bounds = ((xmin,xmin),(xmax,xmax)))
        soln = res.x       
    
        return soln[0]

def BCTliquidus(T):

    # If statement used to confine the solver only to temperatures at which we
    # know a common tangent must exist. This prevents errors that could stop 
    # the code prematurely.
    if T_eutectic <= T <= Tmelt2:

        def func(x):
            X_2_BCT, X_2_L = x
            
            eqns = [dG_Liq(X_2_L,T)-dG_BCT(X_2_BCT,T),
                     DeltaG_mix_L(X_2_L,T) - X_2_L*dG_Liq(X_2_L,T) - (DeltaG_mix_BCT(X_2_BCT,T) - X_2_BCT*dG_BCT(X_2_BCT,T))
                ]
            
            return eqns
         
        # Starting guesses for each value.
        X_2_BCT_guess = .999
        X_2_L_guess = .5
         
        # Set the lower and upper bounds for the solver (i.e., avoid it trying 
        # non-physical values that could produce errors).
        xmin = resolution
        xmax = 1 - resolution

        # Solve the system of equations.
        res = least_squares(func,[X_2_BCT_guess, X_2_L_guess], bounds = ((xmin,xmin),(xmax,xmax)))
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

# Again, this isn't necessary, but I like to know how long each cell takes to 
# run so I can look for opportunities to optimize my code. I probably should've
# done that here as this cell is the slowest of the bunch in this script...
start = time.time()

print('Plotting the entire phase diagram... Please be patient...')

# Define the starting and ending temperatures for the plot. Notice how I am 
# following the bounds of the plot from ASM.
temp_start = 0 + 273.15 # Minimum temperature to plot in K, converted from °C.
temp_end = 350 + 273.15 # Maximum temperature to plot in K, converted from °C.

# Define the temperature range and resolution for your plot.
temprange = np.arange(temp_start, temp_end, 1)

# Define lists of the solvus, solidus, and liquids compositional values. This 
# is the slowest part of plotting the phase diagram. Therefore, pre-calculating
# these lists will be especially important when we want to use the phase
# diagram plot in the animations below and don't want our code to be super 
# slow.
FCCsolvusVALS = list(map(FCCsolvus,temprange))
FCCsolidusVALS = list(map(FCCsolidus,temprange))
FCCliquidusVALS = list(map(FCCliquidus,temprange))
BCTsolvusVALS = list(map(BCTsolvus,temprange))
BCTsolidusVALS = list(map(BCTsolidus,temprange))
BCTliquidusVALS = list(map(BCTliquidus,temprange))

# Now, let's actually make the plot. This first line tell's matplotlib that we
# want to initialize a new plot. This allows us to both name the figure and set
# parameters for how it is rendered and displayed. Note: If we were to use the
# plt.plot() function below without first intializing a new figure with 
# plt.figure(), matplotlib would think we just want the new plot values 
# overlaid on our previous plot.
plt.figure('Phase Diagram',dpi=300)

# Give the plot a title that is displayed above the plot area. Since we already
# gave our model a name, we'll just use that string to title our plot as well.
plt.title(model_name)

# Plot the FCC boundaries (i.e., FCC solvus and solidus in this model). I am 
# giving these boundaries a red color to help distinguish them from other
# single-phae regions. Note: only the first FCC plot is labeled to prevent 
# mutiple "FCC" entries in the legend.
plt.plot(list(map(FCCsolvus,temprange)),temprange - 273.15,color='r',label='FCC')
plt.plot(list(map(FCCsolidus,temprange)),temprange - 273.15,color='r')

# Plot the BCT boundaries (i.e., BCT solvus and solidus in this model).
plt.plot(list(map(BCTsolvus,temprange)),temprange - 273.15,color='purple',label='BCT')
plt.plot(list(map(BCTsolidus,temprange)),temprange - 273.15,color='purple')

# Plot the liquid boundaries (i.e., FCC liquidus and BCT liquidus).
plt.plot(list(map(FCCliquidus,temprange)),temprange - 273.15,color='b',label='L')
plt.plot(list(map(BCTliquidus,temprange)),temprange - 273.15,color='b')

# Plot eutectic line.
plt.plot([FCCsolvus(T_eutectic),BCTsolvus(T_eutectic)],[T_eutectic - 273.15,T_eutectic - 273.15],color='black')

# Label our plot axes. The $$ tells matplotlib that the text between should be
# rendered via LaTeX. The preceding r is not always necessary, but it is never
# a bad idea to include it if you have text you want rendered via LaTeX.
plt.ylabel(r'$T$ (°C)')
plt.xlabel(f'$X_\\mathrm{{{comp2}}}$ (mol/mol)')

# matplotlib will automatically determine boundaries on of your plot and then 
# add ugly margins. One way to deal with this is to manually set the x and y
# limits of the plot, as shown below.
plt.xlim(0,1)
plt.ylim(temp_start - 273.15,temp_end - 273.15)

# The first function here produces a legend for the plot using our labels
# defined above for each curve. I've also forced it to put the legend outside
# of the plot area. The plt.tight_layout() is used to make sure the legend is
# visible on any image files exported by matplotlib.
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

plt.show() # This line isn't necessary in Spyder, but it can't hurt.

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

# Again, this isn't necessary, but I like to know how long each cell takes to 
# run so I can look for opportunities to optimize my code.
start = time.time()

print('Making a super fancy animation...')

# These varibles are used to force the animation below to keep itself to only
# 100 frames. Otherwise, we could be waiting hours for the animation to render.
framecount = 100
temp_stepsize = math.ceil((temp_end - temp_start)/framecount)

# Initialize a figure that will contain both the current GX plot and the entire
# phase diagram at each temperature step in the animation.
fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(9,12))

def plotgxcurves(temp_step):
    
    # Define the temperature for each plot based on the step sizes.
    plottemp = temp_start + (temp_step * temp_stepsize)
    
    # Find the minimum value on ALL GX curves to determine y-axis limits. This
    # is necessary to produce GX plots that are easy to read among the
    # interesting range (i.e., below DeltaG_mix = 0).
    ymin = 1.1 * min(min(DeltaG_mix_FCC(Xrange, plottemp)),min(DeltaG_mix_BCT(Xrange, plottemp)),min(DeltaG_mix_L(Xrange, plottemp)))
        
    # Clear the current plot figure. This is necessary so each frame in our 
    # animation represents the current temperature only. Otherwise, it'll turn
    # into some very psychedelic artwork!
    ax1.clear()
    ax2.clear()

    # Give the overall figure a title.
    fig.suptitle(model_name + ', $T$ = '+f"{(plottemp - 273.15):.1f} °C", fontsize=18)

    # Format x-axis, which is shared by both the GX and phase diagram subplots.
    plt.xlabel(f'$X_\\mathrm{{{comp2}}}$ (mol/mol)')
    plt.xlim(0,1)

    # Format GX subplot y-axis.
    ax1.set_ylim(ymin,0)
    ax1.set_ylabel(r'Δ$G_\mathrm{mix}$ (J/mol)')
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
            
    # Format phase diagram subplot y-axis.
    ax2.set_ylim(temp_start - 273.15,temp_end - 273.15)
    ax2.set_ylabel(r'$T$ (°C)')
    
    # Plot the GX curves at each temperature step.
    ax1.plot(Xrange, DeltaG_mix_FCC(Xrange, plottemp),color='r',label='FCC')
    ax1.plot(Xrange, DeltaG_mix_BCT(Xrange, plottemp),color='purple',label='BCT')
    ax1.plot(Xrange, DeltaG_mix_L(Xrange, plottemp),color='b',label='L')
    ax1.legend(loc='upper center')
    
    # Plot the phase diagram at each temperaure step.
    ax2.plot(FCCsolvusVALS,temprange - 273.15,color='r')
    ax2.plot(FCCsolidusVALS,temprange - 273.15,color='r')
    ax2.plot(BCTsolvusVALS,temprange - 273.15,color='purple')
    ax2.plot(BCTsolidusVALS,temprange - 273.15,color='purple')
    ax2.plot(FCCliquidusVALS,temprange - 273.15,color='b')
    ax2.plot(BCTliquidusVALS,temprange - 273.15,color='b')
    ax2.plot([X_2_FCC_eutectic,X_2_BCT_eutectic],[T_eutectic - 273.15,T_eutectic - 273.15],color='black')

    # Plot the FCC-BCT tangent at each temperature step. The "try" statement is
    # used to prevent errors if the tangent solver returns "None" at a
    # particular temperature step.
    try:
        xleft = FCCsolvus(plottemp)
        xright = BCTsolvus(plottemp)
        yleft = DeltaG_mix_FCC(xleft, plottemp)
        yright = DeltaG_mix_BCT(xright, plottemp)
        ax1.plot([xleft,xright],[yleft,yright],'k--') # Tangent line
        ax1.plot([xleft,xleft],[yleft,ymin],'k--') # Left line to x-axis
        ax1.plot([xright,xright],[yright,ymin],'k--') # Right line to x-axis
        ax2.scatter([xleft,xright],[plottemp - 273.15,plottemp - 273.15],color='k') # Points on PD
        ax1.text(xleft - 2e-2,(yleft+ymin)/2,round(xleft,4), backgroundcolor = 'white',ha="right")
        ax1.text(xright + 2e-2,(yright+ymin)/2,round(xright,4), backgroundcolor = 'white',ha="left")
    except:
        pass

    # Plot the FCC solidus/liquidus tangent at each temperature step. The "try" statement is
    # used to prevent errors if the tangent solver returns "None" at a
    # particular temperature step.
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

    # Plot the BCT solidus/liquidus tangent at each temperature step. The "try" statement is
    # used to prevent errors if the tangent solver returns "None" at a
    # particular temperature step.
    try:
        xleft = BCTliquidus(plottemp)
        xright = BCTsolidus(plottemp)
        yleft = DeltaG_mix_L(xleft, plottemp)
        yright = DeltaG_mix_BCT(xright, plottemp)
        ax1.plot([xleft,xright],[yleft,yright],color='black',linestyle='--') # Tangent line
        ax1.plot([xleft,xleft],[yleft,ymin],color='black',linestyle='--') # Left line to x-axis
        ax1.plot([xright,xright],[yright,ymin],color='black',linestyle='--') # Right line to x-axis
        ax2.scatter([xleft,xright],[plottemp - 273.15,plottemp - 273.15],color='k') # Points on PD
        ax1.text(xleft - 2e-2,(yleft+ymin)/2,round(xleft,4), backgroundcolor = 'white',horizontalalignment="right",verticalalignment="top")
        ax1.text(xright + 2e-2,(yright+ymin)/2,round(xright,4), backgroundcolor = 'white',horizontalalignment="left",verticalalignment="bottom")
    except:
        pass

    return plt

# This defines the animation parameters as "anim" used below.
anim = animation.FuncAnimation(fig, plotgxcurves, interval=200, frames=framecount, repeat=False)

# This is a function that will allow us to print a progress indicator that will
# look nice without having to install 3rd party modules.
def print_progress(current_frame, total_frames):
    percent_done = round((current_frame / total_frames) * 100)
    # Fixed width for the progress string, padded with spaces if necessary.
    progress_str = f'Progress: {percent_done}% complete'.ljust(30, ' ')
    print(f'\r{progress_str}', end='', flush=True)

# This line actually generates the animation and saves it as a GIF. If you 
# want, you can change the file extension to determine the type of file you'd
# like to save it as (e.g., .mp4).
anim.save(model_name + " - GX and PD.gif", progress_callback=print_progress)

# This line is necessary to make sure any subsequent print() calls are on a new
# line.
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
temps_in_C = [25,100,180,200,300]

# Define plot temperature in K.
temps = [T + 273.15 for T in temps_in_C]

def plotgxcurves(plottemp):
    
    plt.figure(plottemp,dpi=300)
    
    # Find the minimum value on ALL GX curves to determine y-axis limits. This
    # is necessary to produce GX plots that are easy to read among the
    # interesting range (i.e., below DeltaG_mix = 0).
    ymin = 1.1 * min(min(DeltaG_mix_FCC(Xrange, plottemp)),min(DeltaG_mix_BCT(Xrange, plottemp)),min(DeltaG_mix_L(Xrange, plottemp)))

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
    plt.plot(Xrange, DeltaG_mix_BCT(Xrange, plottemp),color='purple',label='BCT')
    plt.plot(Xrange, DeltaG_mix_L(Xrange, plottemp),color='b',label='L')
    plt.legend(loc='upper center')
    
    # Plot the FCC-BCT tangent.
    try:
        xleft = FCCsolvus(plottemp)
        xright = BCTsolvus(plottemp)
        yleft = DeltaG_mix_FCC(xleft, plottemp)
        yright = DeltaG_mix_BCT(xright, plottemp)
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

    # Plot the BCT solidus/liquidus tangent.
    try:
        xleft = BCTliquidus(plottemp)
        xright = BCTsolidus(plottemp)
        yleft = DeltaG_mix_L(xleft, plottemp)
        yright = DeltaG_mix_BCT(xright, plottemp)
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