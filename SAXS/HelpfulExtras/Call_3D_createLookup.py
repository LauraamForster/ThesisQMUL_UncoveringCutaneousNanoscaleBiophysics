import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threeDXRD as t3d  # adjust import as needed

# --------------------------------------------------
# Fixed setup
# --------------------------------------------------
q1, q2, nq = 0.24, 0.45, 100
chi1, chi2, nchi = 0, 360, 180
qr = np.linspace(q1, q2, nq)
chir = np.linspace(chi1, chi2, nchi)
qrg, chirg = np.meshgrid(qr, chir)
wavelen = 0.0884

# Geometry
alpha, beta = 0, 0
qxL, qyL, qzL = t3d.calc_ewald_trace_grid_arrayinputs(wavelen, qr, chir)
qx, qy, qz = t3d.rotated_vectors_rev(qxL, qyL, qzL, alpha, beta)

# Fixed scattering setup
amp = 1
gamma0 = 0
mu = 0
wMu = 0.5
nfac = 1
integrate = True
delta = 0
peak = 'Gaussian'
angpeak = 'Gaussian'
deltaGamma0 = 0.01
wgammaInt = 1

# --------------------------------------------------
# Parameter ranges to explore
# --------------------------------------------------
q0_values = np.linspace(0.272, 0.311, 40)        # broadened reflection positions
# q0_values = [0.28]       # broadened reflection positions
deltaQ0_values = np.linspace(0.001, 0.01, 25)    # broadened widths
# deltaQ0 = 0.005   # broadened widths
wMu_values = np.linspace(0.01, 0.3, 50) # skew
wMu_values = 0.01# skew

# Experimental area range for scaling
area_min, area_max = 0.00007, 0.0098
# --------------------------------------------------
# Run simulations
# --------------------------------------------------
lookup = []

fig,ax= plt.subplots(nrows=2)

for deltaQ0 in deltaQ0_values:
    
    for q0 in q0_values:
        
        skew = []
        firstmoment = []
        secondmoment = []
        thirdmoment = []
        
        for wMu in wMu_values:
            
            # Run 3D model
            fval2D = amp * t3d.Iplanarfibril(
                qxL, qyL, qzL, gamma0, deltaGamma0, mu, q0, wMu,
                deltaQ0, nfac, alpha, beta,
                integrate=integrate, flat=1, delta=delta,
                peak=peak, angpeak=angpeak, originalFlat=True, verbose=False
            ) / wgammaInt
    
            # Average over chi to get I(q)
            fval = np.average(fval2D, axis=0)
            
            ax[0].plot(qr,fval, label=wMu)
            # ax[0].legend()
    
            # Normalise and rescale to experimental area range
            area_target = np.random.uniform(area_min, area_max)
            fval_scaled = fval / np.trapz(fval, qr) * area_target
    
            # Weighted moment analysis
            wm = t3d.WeightedMoment(qr, fval, order_no=3, ax=ax[0])
    
            # Store results
            lookup.append({
                "q0": q0,
                "deltaQ0": deltaQ0,
                "wMu": wMu,
                "firstmoment": wm["firstmoment"],
                "secondmoment": wm["secondmoment"],
                "thirdmoment": wm["thirdmoment"],
                "skewness": wm["skewness"],
                
            })
            
            skew.append(wm["skewness"])
            firstmoment.append(wm["firstmoment"])
            secondmoment.append(wm["secondmoment"])
            thirdmoment.append(wm["thirdmoment"])
            
        ax[1].plot(wMu_values, secondmoment, label=q0)     
        
        
    # ax[1].legend()


# --------------------------------------------------
# Save and visualise
# --------------------------------------------------
lookup_df = pd.DataFrame(lookup)
lookup_df.to_csv("NEWlookuptable_WM.csv", index=False)


