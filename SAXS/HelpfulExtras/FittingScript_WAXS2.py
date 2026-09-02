#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 14:10:37 2025

@author: lauraforster
"""
import h5py
from h5py import File
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.signal import savgol_filter, find_peaks
from Utils import progress_bar
from scipy.ndimage import gaussian_filter1d
from lmfit.models import GaussianModel, LinearModel, SkewedGaussianModel, ExponentialModel, ConstantModel
from matplotlib.backends.backend_pdf import PdfPages
import time
import logging
from itertools import product

logging.getLogger("pyFAI").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings(
    "ignore",
    message="Using UFloat objects with std_dev==0 may give unexpected results.",
    module="uncertainties.core"
)
import warnings

warnings.filterwarnings(
    "ignore",
    message="AffineScalarFunc\\.error_components\\(\\).*deprecated",
    category=FutureWarning,
    module=r"uncertainties\.core"
)
warnings.filterwarnings(
    "ignore",
    message="AffineScalarFunc\\.derivatives\\(\\) is deprecated.*",
    category=FutureWarning,
    module=r"uncertainties\.core"
)

# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------ IQ Fitting Functions ------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

# def FitWAXSPeak(ix, iy, I, q, pdf_pages):
#     # arrays
#     q = np.asarray(q, dtype=float)
#     I = np.asarray(I, dtype=float)

#     # cut to broad peak region (ignore sharp noise peak ~2.5–3)
#     mask = (q >= 3.0) & (q <= 10.0) & np.isfinite(q) & np.isfinite(I)
#     q_cut = q[mask]
#     I_cut = I[mask]
#     if q_cut.size < 5:
#         return None

#     # smoothing
#     win = min(11, q_cut.size if q_cut.size % 2 == 1 else q_cut.size - 1)
#     win = max(win, 5)
#     if win >= q_cut.size:
#         win = q_cut.size - (1 - q_cut.size % 2)
#     I_smooth = savgol_filter(I_cut, window_length=win, polyorder=3, mode="interp") if win >= 5 else I_cut

#     # area under peak
#     area_peak = float(np.trapz(I_smooth, q_cut))
#     if area_peak < 0:
#         area_peak = 0.0
        
#     # ---- Model: G1 + G2 + Constant
#     g1 = GaussianModel(prefix="g1_")
#     g2 = GaussianModel(prefix="g2_")
#     c  = ConstantModel(prefix="c_")
#     model = g1 + g2 + c

#     # initial baseline + sigma
#     y0 = float(np.nanmedian(I_smooth))
#     sig0 = max((q_cut.max() - q_cut.min()) / 12.0, 1e-3)

#     # find up to 2 candidate peaks on smoothed data
#     y_for_peaks = np.clip(I_smooth - y0, 0, None)
#     peaks, props = find_peaks(y_for_peaks, prominence=np.nanmax(y_for_peaks) * 0.05 if np.nanmax(y_for_peaks) > 0 else None)


#     # expected centres
#     cen1, cen2 = 4.5, 6.5
    
#     # if find_peaks found candidates, use them only if they look real
#     if peaks.size >= 1:
#         prominences = props.get("prominences", np.zeros_like(peaks, dtype=float))
    
#         # score threshold: prominence must be a decent fraction of the smoothed dynamic range
#         dyn = float(np.nanmax(I_smooth) - np.nanmedian(I_smooth))
#         prom_thresh = 0.08 * dyn  # tweak: 0.05 looser, 0.10 stricter
    
#         # keep only "realistic" peaks
#         keep = prominences >= prom_thresh
#         pk = peaks[keep]
#         pr = prominences[keep]
    
#         if pk.size >= 1:
#             # refine each expected centre using the closest kept peak
#             qpk = q_cut[pk].astype(float)
    
#             # nearest peak to cen1
#             j1 = int(np.argmin(np.abs(qpk - cen1)))
#             cen1_ref = float(qpk[j1])
    
#             # nearest peak to cen2 (avoid reusing same peak index)
#             if pk.size >= 2:
#                 # drop the one used for cen1 then choose nearest to cen2
#                 qpk2 = np.delete(qpk, j1)
#                 cen2_ref = float(qpk2[int(np.argmin(np.abs(qpk2 - cen2)))])
#             else:
#                 cen2_ref = cen2
    
#             cen1, cen2 = cen1_ref, cen2_ref
        
#     # amplitude guesses from local height
#     def amp_guess(center):
#         idx = int(np.argmin(np.abs(q_cut - center)))
#         height0 = max(float(I_smooth[idx] - y0), 0.0)
#         return max(height0 * sig0 * np.sqrt(2*np.pi), 1e-8)

#     amp1 = amp_guess(cen1)
#     amp2 = amp_guess(cen2)

#     params = model.make_params(
#         c_c=y0,
#         g1_center=cen1, g1_sigma=sig0, g1_amplitude=amp1,
#         g2_center=cen2, g2_sigma=sig0, g2_amplitude=amp2,
#     )

#     # bounds: allow either gaussian to vanish via amplitude -> 0
#     params["g1_center"].set(min=3.0, max=10.0)
#     params["g2_center"].set(min=3.0, max=10.0)
#     params["g1_sigma"].set(min=1e-6)
#     params["g2_sigma"].set(min=1e-6)
#     params["g1_amplitude"].set(min=0.0)
#     params["g2_amplitude"].set(min=0.0)

#     result = model.fit(I_smooth, params, x=q_cut)
#     y_fit = result.best_fit
    
#     # r2 on smoothed
#     ss_res = float(np.nansum((I_smooth - y_fit) ** 2))
#     ss_tot = float(np.nansum((I_smooth - np.nanmean(I_smooth)) ** 2))
#     r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

#     # extract params
#     p = result.params
#     g1_center = float(p["g1_center"].value); g1_amp = float(p["g1_amplitude"].value); g1_height = float(p["g1_height"].value)
#     g2_center = float(p["g2_center"].value); g2_amp = float(p["g2_amplitude"].value); g2_height = float(p["g2_height"].value)

#     # decide if a peak is "present" (tiny amplitudes are basically none)
#     amp_eps = 1e-6
#     g1_present = g1_amp > amp_eps
#     g2_present = g2_amp > amp_eps

#     # dominant peak (by height)
#     if g1_present and g2_present:
#         dominant = 1 if g1_height >= g2_height else 2
#     elif g1_present:
#         dominant = 1
#     elif g2_present:
#         dominant = 2
#     else:
#         dominant = 0

#     # plot: cut, smoothed, total fit + each gaussian component (baseline included in total)
#     fig, ax = plt.subplots(figsize=(7, 4))
#     # ax.plot(q_cut, I_cut, label="cut")
#     ax.plot(q_cut, I_smooth, label="smoothed")
#     ax.plot(q_cut, y_fit, label="G1+G2+const fit")

#     comps = result.eval_components(x=q_cut)
#     # comps keys: 'g1_', 'g2_', 'c_' (with your prefixes)
#     if "g1_" in comps:
#         ax.plot(q_cut, comps["g1_"] + comps.get("c_", 0), label="G1+const", alpha=0.8)
#     if "g2_" in comps:
#         ax.plot(q_cut, comps["g2_"] + comps.get("c_", 0), label="G2+const", alpha=0.8)

#     ax.legend()
#     ax.set_title(f"({ix},{iy}) dom={dominant}  R²={r2:.3f}")
#     pdf_pages.savefig(fig)
#     plt.close(fig)
#     # plt.show()


#     return  area_peak, r2,  g1_center, g1_amp, g1_height, int(g1_present), g2_center, g2_amp, g2_height, int(g2_present), dominant


def FitWAXSPeak(ix, iy, I, q, pdf_pages):
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)

    # cut region
    mask = (q >= 3.0) & (q <= 10.0) & np.isfinite(q) & np.isfinite(I)
    q_cut = q[mask]
    I_cut = I[mask]
    if q_cut.size < 5:
        return None

    # smoothing
    win = min(11, q_cut.size if q_cut.size % 2 == 1 else q_cut.size - 1)
    win = max(win, 5)
    if win >= q_cut.size:
        win = q_cut.size - (1 - q_cut.size % 2)
    I_smooth = savgol_filter(I_cut, window_length=win, polyorder=3, mode="interp") if win >= 5 else I_cut

    # area under curve (smoothed)
    area_peak = float(np.trapezoid(I_smooth, q_cut))
    if area_peak < 0:
        area_peak = 0.0

    # ---- Model: G1 + Constant
    g1 = GaussianModel(prefix="g1_")
    c  = ConstantModel(prefix="c_")
    model = g1 + c

    # initial baseline + gaussian guesses
    y0 = float(np.nanmedian(I_smooth))
    imax = int(np.nanargmax(I_smooth))
    cen0 = float(q_cut[imax])
    sig0 = max((q_cut.max() - q_cut.min()) / 12.0, 1e-3)
    amp0 = max(float(I_smooth[imax] - y0) * sig0 * np.sqrt(2*np.pi), 1e-8)

    params = model.make_params(
        c_c=y0,
        g1_center=cen0,
        g1_sigma=sig0,
        g1_amplitude=amp0
    )

    params["g1_center"].set(min=3.0, max=10.0)
    params["g1_sigma"].set(min=1e-6)
    params["g1_amplitude"].set(min=0.0)
    
    

    result = model.fit(I_smooth, params, x=q_cut)
    y_fit = result.best_fit

    # r2 on smoothed
    ss_res = float(np.nansum((I_smooth - y_fit) ** 2))
    ss_tot = float(np.nansum((I_smooth - np.nanmean(I_smooth)) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    p = result.params
    g1_center = float(p["g1_center"].value)
    g1_amp    = float(p["g1_amplitude"].value)
    g1_height = float(p["g1_height"].value)
    
    I_period = (2*np.pi ) / g1_center

    amp_eps = 1e-6
    g1_present = int(g1_amp > amp_eps)
    dominant = 1 if g1_present else 0

    

    # plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(q_cut, I_smooth, label="smoothed")
    ax.plot(q_cut, y_fit, label="fit")  # this is already G1+const
    
    comps = result.eval_components(x=q_cut)
    if "c_" in comps:
        ax.plot(q_cut, comps["c_"], label="const", alpha=0.8)
    
    ax.legend()
    ax.set_title(f"({ix},{iy}) R²={r2:.3f}")
    pdf_pages.savefig(fig)
    plt.close(fig)

    return area_peak, r2, g1_center, g1_amp, g1_height, g1_present, I_period

# ------------------------------------------------------------------------------------------------------------------------


def ProcessIQFitting(scan_no, Output_directoryCSV, Output_directorybsd, sample_loc, xcoords, ycoords):
    # # read in various file paths
    dat_nxs_file = f'i22-{scan_no}-WAXS_processed.nxs'
    nxs_file_path = os.path.join(sample_loc,dat_nxs_file)
    outputname = os.path.join(Output_directoryCSV, f"{scan_no} IQ_fitting.csv")

    # # make some directories
    os.makedirs(Output_directoryCSV, exist_ok=True)
    os.chdir(Output_directoryCSV)
    pdf_pages = PdfPages(os.path.join(Output_directoryCSV, f"{scan_no} IQ_fitting_outputplots.pdf"))

    # # storage
    ixt, iyt = [], []
    total_saxs_raw = []
    area_peaksG1, amplitudesG1, heightsG1,  presentG1,I_periods, r2s = [], [], [], [], [],[]
    
    count_total, processed = 0, 0
    start = time.time()

    with h5py.File(nxs_file_path, 'r') as nxs_file:
        r = nxs_file["processed/result"]
        iq_data = r["data"][...]          # (ny, nx, nq)
        q_values = r["q"][...]             # (nq,)
        xset  = r["p1xy_x_set"][...]    # (nx,)
        yset  = r["p1xy_y_set"][...]    # (ny,)

        ny, nx, nq = iq_data.shape
        num_frames = ny * nx

        coords = [(float(y), float(x)) for y in yset for x in xset]

        if xcoords is not None and ycoords is not None:
            req = set((round(float(y), 6), round(float(x), 6)) for y, x in product(ycoords, xcoords))
            frame_indices = [i for i, (y, x) in enumerate(coords) if (round(y, 6), round(x, 6)) in req]
        else:
            frame_indices = list(range(num_frames))

        total = len(frame_indices)
        total_raw_by_idx = np.full(num_frames, np.nan)
        
        for frame_index in frame_indices:
            iy, ix = divmod(frame_index, nx)      # map flat index -> (row, col)
        
            intensity_values = np.asarray(iq_data[iy, ix, :], dtype=float)
            if intensity_values.size == 0:
                continue
        
            area_all = float(np.trapz(intensity_values, q_values))
            if area_all < 0:
                area_all = 0.0
            total_raw_by_idx[frame_index] = area_all
        
            count_total += 1
            y_pos, x_pos = coords[frame_index]   # these are motor positions (floats)
            q_values = np.asarray(q_values, dtype=float)
            intensity_values = np.asarray(iq_data[iy, ix, :], dtype=float)
            
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(q_values, intensity_values)
            ax.set_xlim(2, 10)
            ax.set_ylim(1,10)
            ax.set_yscale('log')
            ax.set_title(f"RAW Data x={ix}, y={iy}, area={area_all:.3g}")
            pdf_pages.savefig(fig)
            plt.close(fig)
            # plt.show()
            
            processed += 1
            start = progress_bar(processed, total,
                                 prefix=f"[{scan_no}] IQ fitting",
                                 start_time=start)

            # coordinates & totals (append early to keep lengths aligned)
            ixt.append(ix)
            iyt.append(iy)
            total_saxs_raw.append(np.round(area_all, 6))
            
            area_peak, r2,  g1_center, g1_amp, g1_height, g1_present,I_period = FitWAXSPeak(ix, iy, intensity_values, q_values, pdf_pages)
           
            area_peaksG1.append(g1_center)
            amplitudesG1.append(g1_amp)
            heightsG1.append(g1_height)
            presentG1.append(g1_present)
            I_periods.append(I_period)
            
            r2s.append(r2)
            
    df = pd.DataFrame({
        "x": ixt,
        "y": iyt,
        "total_area": total_saxs_raw,
        
        "g1_center": area_peaksG1,
        "g1_amp": amplitudesG1,
        "g1_height": heightsG1,
        "g1_present": presentG1,
                
        "r2": r2s,
        "I_period": I_periods,

    })
    df.to_csv(outputname, index=False)
    
    pdf_pages.close()

    return