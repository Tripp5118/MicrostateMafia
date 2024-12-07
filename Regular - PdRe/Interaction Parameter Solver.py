import numpy as np
from scipy.optimize import least_squares

# This was my favorite part of the project bc it's so simple but its so cool

R = 8.31446261815324
LN = ln = Ln = lN = np.log
resolution = 1e-6

if True:
    # Your functions remain the same as provided.
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


# Target points
points = [(9.292763157894736, 1650 + 273.15), 
          (16.036184210526315, 1650 + 273.15), 
          (93.91447368421053, 1650 + 273.15)]

def find_eutectic(W_FCC, W_HCP, W_LIQ):

    DeltaG_mix_FCC = lambda X_2, T: W_FCC * X_2*(1-X_2) + T*S_ideal(X_2) + (1-X_2)*FCC_hanging_Pd(T) + X_2*FCC_hanging_Re(T)
    DeltaG_mix_HCP = lambda X_2, T: W_HCP * X_2*(1-X_2) + T*S_ideal(X_2) + (1 - X_2)*HCP_hanging_Pd(T) + X_2*HCP_hanging_Re(T)
    DeltaG_mix_L   = lambda X_2, T: W_LIQ * X_2*(1-X_2) + T*S_ideal(X_2) + (1 - X_2)*Liq_hanging_Pd(T) + X_2*Liq_hanging_Re(T)

    from scipy.optimize import least_squares

    def ddx(function, dX):
        return lambda X, T: (function(X + dX, T) - function(X, T)) / dX

    dG_Liq = ddx(DeltaG_mix_L, 1e-5)
    dG_FCC = ddx(DeltaG_mix_FCC, 1e-5)
    dG_HCP = ddx(DeltaG_mix_HCP, 1e-5)

    def func(x):
        X_2_FCC, X_2_L, X_2_HCP, T = x

        eqns = [
            dG_HCP(X_2_HCP,T) - dG_FCC(X_2_FCC,T),
            dG_HCP(X_2_HCP,T) - dG_Liq(X_2_L,T),
            DeltaG_mix_HCP(X_2_HCP,T) - X_2_HCP * dG_HCP(X_2_HCP,T) - (DeltaG_mix_L(X_2_L,T) - X_2_L*dG_Liq(X_2_L,T)),
            DeltaG_mix_HCP(X_2_HCP,T) - X_2_HCP * dG_HCP(X_2_HCP,T) - (DeltaG_mix_FCC(X_2_FCC,T) - X_2_FCC*dG_FCC(X_2_FCC,T))
        ]

        return eqns

    X_2_FCC_guess = 0.10
    X_2_L_guess = 0.50 
    X_2_HCP_guess = 0.90
    T_guess = 1650.0

    resolution = 1e-6
    xmin = resolution
    xmax = 1 - resolution
    tempmin = 1000
    tempmax = 3200

    res = least_squares(
        func,
        [X_2_FCC_guess, X_2_L_guess, X_2_HCP_guess, T_guess],
        bounds=((xmin, xmin, xmin, tempmin), (xmax, xmax, xmax, tempmax))
    )

    eutec_soln = res.x
    X_2_FCC_eutectic = eutec_soln[0]
    X_2_L_eutectic   = eutec_soln[1]
    X_2_HCP_eutectic = eutec_soln[2]
    T_eutectic       = eutec_soln[3]

    pred_points = sorted([(X_2_L_eutectic*100, T_eutectic),(X_2_FCC_eutectic*100, T_eutectic), (X_2_HCP_eutectic*100, T_eutectic)], key=lambda x: x[0])

    return pred_points

def residual_W(W):
    W_FCC, W_HCP, W_LIQ = W
    pred_points = find_eutectic(W_FCC, W_HCP, W_LIQ)


    resid = []
    for (px, pT), (tx, tT) in zip(pred_points, points):
        resid.append(px - tx)
        resid.append(pT - tT)
    return np.array(resid)

initial_guess = [5000, 5000, 5000]

result = least_squares(residual_W, initial_guess, bounds=(0, np.inf))
W_optimized = result.x

print("Optimized W parameters:", W_optimized)
print("Residuals:", result.fun)
print("Predicted points:", find_eutectic(*W_optimized))
