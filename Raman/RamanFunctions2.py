#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 08:57:53 2025

@author: lauraforster
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
from matplotlib.backends.backend_pdf import PdfPages
import ramanspy
from pathlib import Path
from sklearn.decomposition import PCA as _SKPCA
from sklearn.decomposition import TruncatedSVD as _SVD
from collections import defaultdict
from scipy.signal import savgol_filter, find_peaks
from itertools import cycle
# ---------------------------------------------------------------------------------------------------------
# -----------------------------------------------Initialisation--------------------------------------------
# ---------------------------------------------------------------------------------------------------------
"""
Raman linescan initialisation utilities.

This module prepares a working dictionary of samples and spectra, and (optionally) plots
spectra for a chosen sample. Typical flow:
1) Types(Type) -> returns Subtypes and default plot styles for that Type.
2) read_Samplemanifest(...) / read_Peakmanifest(...) -> load manifests.
3) CreateDict(sample_manifest_df, Subtypes) -> build data_dict index of samples to process.
4) readindata(DataDir, data_dict) -> read FP/EXT TXT files (X, Y, Wave, Intensity) into data_dict.
5) SplitSpectra(data_dict, Colours, Linestyles, SN, step) -> split each TXT into per-(X,Y)
   spectra, drop raw DataFrames to save memory, and optionally plot all spectra for sample SN.

Notes:
- SplitSpectra groups by the X,Y found in the TXT files; it does NOT assign skin layers from
  the Excel manifest. If layer tagging is needed, add a mapping step from (X,Y) to layer.
- EXTENDED_SUFFIX can be used to build extended file names consistently.

Functions:
- Types(Type): Return (Subtypes, Colours, Linestyles) for the given high-level Type.
- read_Samplemanifest(path): Load Excel sample manifest -> DataFrame.
- read_Peakmanifest(path): Load CSV peak/feature manifest -> DataFrame.
- CreateDict(sample_manifest_df, Subtypes): From the manifest, create a data_dict keyed by
  sample number with Type/Subtype metadata (filtered to allowed Subtypes).
- readindata(DataDir, data_dict): For each sample, read '{sample}A_linescan1.txt' and
  '{sample}A_linescan1_extended.txt' into data_dict['FP'] / data_dict['EXT'] with ffilled X,Y.
- SplitSpectra(data_dict, Colours, Linestyles, SN, step): Convert FP/EXT DataFrames into
  dicts of spectra per (X,Y) coordinate; optionally plot all spectra for sample SN, thinning
  by 'step' (plot every nth spectrum).
"""

def Types(Type):
    if Type == 'AP1':
        Subtypes = ["TS", "VH", "CL", "AC"]
        Colours = {"TS": "blue", "VH": "green",  "AC": "red", "CL": "peru"}
        Linestyles = {"TS": "-", "VH": "--",  "CL": "--", "AC": ":"}
    elif Type == 'WOUND':
        Subtypes = ["CT", "PBS", "D7", "D10" ,"D14", "D21"]
        Colours = {"CT": "blue","PBS": "blue", "D7": "green",  "D10": "red", "D14": "peru", "D21": "black"}
        Linestyles = {"CT": "-", "PBS": "-", "D7": "--",  "D10": "--", "D14": ":", "D21": "-"}
        # Subtypes = ["CT", "D7", "D14", ]
        # Colours = {"CT": "blue", "D7": "green",   "D14": "peru"}
        # Linestyles = {"CT": "-", "D7": "--", "D14": ":"}

    elif Type == 'BLEO':
        Subtypes = ["PBS", "2W", "4W_5R"]
        Colours = {"PBS": "blue", "2W": "green", "4W_5R": "peru"}
        Linestyles = {"PBS": "-", "2W": "--", "4W_5R": ":"}
    elif Type == 'BLEO_MET':
        Subtypes = ["PBS_MET", "BM_MET", "4W_MET"]
        Colours = {"PBS_MET": "blue", "BM_MET": "green", "4W_MET": "peru"}
        Linestyles = {"PBS_MET": "-", "BM_MET": "--", "4W_MET": ":"}
    elif Type == 'BLEO_OKN':
        Subtypes = ["PBS_OKN",  "4W_OKN"]
        Colours = {"PBS_OKN": "blue", "4W_OKN": "peru"}
        Linestyles = {"PBS_OKN": "-", "4W_OKN": ":"}
    
    return Subtypes, Colours, Linestyles

def count_linescan_points(data_dict):
    """
    Make a table of how many spectra (positions) each repeat has:
      - N_total: number of spectra along the processed line (prefers *Treated*, then raw)
      - N_dermis: number of spectra in the Dermis segment (after TrimRegion)
    Counts are reported per sample and per region (FP/EXT where available).

    Returns
    -------
    df : pd.DataFrame with columns:
         ["Sample", "Subtype", "Region", "N_total", "N_dermis"]
    Also stores a copy in data_dict["_counts"]["per_region"].
    """
    rows = []
    for sample_num, sdict in data_dict.items():
        if not isinstance(sdict, dict):
            continue
        subtype = sdict.get("Subtype", None)

        for reg in ("FP", "EXT"):
            # Prefer processed (analysis window) if present; otherwise fall back.
            total_dict = (
                sdict.get(f"{reg}_Spectra_Treated") or
                sdict.get(f"{reg}_Spectra_Treated_Full") or
                sdict.get(f"{reg}_Spectra")
            )
            dermis_dict = sdict.get(f"{reg}_Spectra_Treated_Dermis")

            if (total_dict is None) and (dermis_dict is None):
                # nothing for this region in this sample—skip
                continue

            n_total  = len(total_dict)  if isinstance(total_dict, dict)  else 0
            n_dermis = len(dermis_dict) if isinstance(dermis_dict, dict) else 0

            rows.append({
                "Sample": str(sample_num),
                "Subtype": subtype,
                "Region": reg,
                "N_total": int(n_total),
                "N_dermis": int(n_dermis),
            })

    df = pd.DataFrame(rows).sort_values(["Subtype", "Sample", "Region"], ignore_index=True)

    # store for later use
    data_dict.setdefault("_counts", {})["per_region"] = df
    return df, data_dict


def read_Samplemanifest(manifest_path):
    sample_df = pd.read_excel(manifest_path)
    return sample_df

def read_Peakmanifest(manifest_path):
    peak_df = pd.read_csv(manifest_path)
    return peak_df

def CreateDict(sample_manifest_df, Subtypes):
    # 1. Create an empty dictionary
    data_dict = {}
    # 2. Read manifest file and find the subtypes in the Type column
    for _, row in sample_manifest_df.iterrows():
        sample_num = str(int(row["Sample Number"])).strip()
        subtype = str(row["TYPE"]).strip()
        # 3. Find the corresponding Sample Number and create keys within the dictionary 
        # of these Sample Numbers with Type and Subtype categories wihtin the dict
        if subtype in Subtypes:
            data_dict[sample_num] = {
                "Type": str(row["TYPE"]).strip(),
                "Subtype": subtype
                # You can add more metadata fields if needed
            }

    return data_dict
    
def readindata(DataDir, data_dict):
    
    # 1. From the dictionary we know which sample numbers we have and their group (subtype)
    for sample_num in data_dict.keys():
        subtype = data_dict[sample_num]["Subtype"]
        base_path = DataDir / subtype / f"{sample_num}A"
    
    # 2. Find each .txt file 
        # Format is DataDir path +/subtype +/{samplenumber}A_linescan1.txt and +/{samplenumber}A_linescan1_extended.txt
        fp_path = base_path.with_name(f"{sample_num}A_linescan1.txt")
        ext_path = base_path.with_name(f"{sample_num}A_linescan1_extended.txt")
        
    # 3. Read in both the fingerprint and extended .txt files as 4 columns and save into the dictionary under the sample number
        for path, key in [(fp_path, "FP"), (ext_path, "EXT")]:
            try:
                # Robust read: whitespace-sep, skip comment lines
                df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=["X", "Y", "Wave", "Intensity"])
                
                # Forward-fill X and Y to replicate Raman structure
                df[["X", "Y"]] = df[["X", "Y"]].ffill()

                # Drop header rows or junk rows
                df = df[df["Wave"].apply(lambda x: isinstance(x, (float, int)))]

                data_dict[sample_num][key] = df

            except FileNotFoundError:
                print(f"{key} file not found for sample {sample_num}: {path}")

    return data_dict

def SplitSpectra(data_dict, Colours, Linestyles, SN, step):

    for sample_num in data_dict.keys():
        for region in ["FP", "EXT"]:
            key_name = f"{region}_Spectra"
            spectra = {}

            if region not in data_dict[sample_num]:
                continue

            df = data_dict[sample_num][region]

            # Group rows by (X, Y) coordinate — each group is one spectrum
            grouped = df.groupby(["X", "Y"], sort=False)
            for (x, y), group in grouped:
                wave = group["Wave"].values
                intensity = group["Intensity"].values
                spectra[(x, y)] = {"Wave": wave, "Intensity": intensity}

            data_dict[sample_num][key_name] = spectra
# Deleting Steps
        # Clean up raw DataFrames to save space
        # we can then delete the fp_df and ext_df from the dictionary and just store all the spectra under their own sample number key
        data_dict[sample_num].pop("FP", None)
        data_dict[sample_num].pop("EXT", None)

# Plotting steps

        if SN is not None and str(SN) == sample_num:
            subtype = data_dict[sample_num].get("Subtype")
            colour = Colours.get(subtype, None) if Colours else None
            linestyle = Linestyles.get(subtype, None) if Linestyles else None
            for region in ["FP_Spectra", "EXT_Spectra"]:
                if region not in data_dict[sample_num]:
                    continue

                plt.figure(figsize=(10, 6))
                spectra_dict = data_dict[sample_num][region]

                for i, ((x, y), spectrum) in enumerate(spectra_dict.items()):
                    if step is None or i % step == 0:
                        x,y = np.round(x,2), np.round(y,2)
                        label = f"x={x}, y={y}"
                        plt.plot(spectrum["Wave"], spectrum["Intensity"], label=label, color=colour, linestyle=linestyle)

                plt.title(f"{region} spectra for Sample {sample_num}")
                plt.xlabel("Wavenumber (cm⁻¹)")
                plt.ylabel("Intensity")
                plt.legend(fontsize="x-small", loc="upper right", ncol=2)
                plt.tight_layout()
                plt.show()

    return data_dict

# ---------------------------------------------------------------------------------------------------------
# -----------------------------------------------Preprocessing---------------------------------------------
# ---------------------------------------------------------------------------------------------------------
_CANONICAL_ORDER = ["despike", "smooth", "baseline", "normalise"]

def _minmax_for_plot(y):
    y = np.asarray(y, dtype=float)
    # robust min/max using percentiles to avoid spikes dominating
    lo, hi = np.percentile(y, 1), np.percentile(y, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.min(y), np.max(y)
    if hi <= lo:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo)

def _normalize_steps(preprocess_list):
    """Return a set of valid steps, lowercased, filtered to known names."""
    valid = {s.lower() for s in preprocess_list or []}
    unknown = valid.difference(_CANONICAL_ORDER)
    if unknown:
        print(f"Warning: ignoring unknown steps: {sorted(unknown)}")
    return [s for s in _CANONICAL_ORDER if s in valid]  # keep canonical order

def _build_region_steps(full_rng, crop_rng, selected_steps):
    """Build callables and titles for the region, with crop at start and end."""
    cropper_full  = ramanspy.preprocessing.misc.Cropper(region=full_rng)
    despike       = ramanspy.preprocessing.despike.WhitakerHayes()
    savgol        = ramanspy.preprocessing.denoise.SavGol(window_length=11, polyorder=3)
    baseline        = ramanspy.preprocessing.baseline.ASLS()
    auc           = ramanspy.preprocessing.normalise.AUC(pixelwise=True)
    cropper_final = ramanspy.preprocessing.misc.Cropper(region=crop_rng)

    step_map = {
        "despike":    (lambda sc: despike.apply(sc),    "After despike (WhitakerHayes)"),
        "smooth":     (lambda sc: savgol.apply(sc),     "After smoothing (SavGol)"),
        "baseline":   (lambda sc: baseline.apply(sc),     "After baseline (ASLS)"),
        "normalise":  (lambda sc: auc.apply(sc),        "After normalisation (AUC)"),
    }

    # full pipeline (no final crop) used for stored *_Treated_Full and overlay middle trace
    step_fns_full  = [lambda sc: cropper_full.apply(sc)]
    titles_full    = ["After crop (full range)"]
    for name in selected_steps:
        fn, title = step_map[name]
        step_fns_full.append(fn)
        titles_full.append(title)

    # final-cropped pipeline (adds final crop)
    step_fns_final = step_fns_full + [lambda sc: cropper_final.apply(sc)]
    titles_final   = titles_full + ["After final crop (analysis window)"]

    return step_fns_full, titles_full, step_fns_final, titles_final

def _style_for(label):
    """Consistent, visible styles by label substring."""
    lab = label.lower()
    if "raw" in lab:
        return dict(ls="-", lw=1.3, alpha=0.95, zorder=1)
    if "crop (full" in lab:
        return dict(ls="None", marker="o", ms=2.2, alpha=0.9, zorder=5, color="tab:orange")
    if "despike" in lab:
        return dict(ls="--", lw=1.5, alpha=0.95, zorder=3, color="tab:green")
    if "smoothing" in lab:
        return dict(ls="-", lw=2.0, alpha=0.95, zorder=6, color="tab:red")
    if "baseline" in lab:
        return dict(ls=":", lw=1.6, alpha=0.9, zorder=4, color="tab:purple")
    if "normalisation" in lab:
        return dict(ls="-.", lw=1.4, alpha=0.9, zorder=2, color="tab:gray")
    if "final crop" in lab:
        return dict(ls="None", marker=".", ms=2.6, alpha=0.95, zorder=7, color="saddlebrown")
    # fallback
    return dict(ls="-", lw=1.2, alpha=0.9, zorder=2)

def _pipeline_debug_fig(sc_raw, step_fns, titles, suptitle=""):
    """
    Build a single multi-panel figure showing the spectrum after each step.
    For baseline steps, insert a PREVIEW panel BEFORE the baseline-corrected panel
    that overlays the estimated baseline with the previous spectrum (previous drawn on top).

    Parameters
    ----------
    sc_raw : ramanspy.SpectralContainer
    step_fns : list[callable]   # cumulative steps; each returns a SpectralContainer
    titles : list[str]          # names per step (same length/order as step_fns)
    suptitle : str
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from math import ceil

    # Stages as list of tuples: (title, x, y, y_overlay_or_None)
    stages = []

    # step 0 (raw)
    x_prev = np.asarray(sc_raw.spectral_axis, float)
    y_prev = np.asarray(sc_raw.spectral_data, float)
    stages.append(("0 – Raw", x_prev, y_prev, None))

    sc_prev = sc_raw

    for i, (fn, name) in enumerate(zip(step_fns, titles), start=1):
        # Apply step
        sc_curr = fn(sc_prev)
        x = np.asarray(sc_curr.spectral_axis, float)
        y = np.asarray(sc_curr.spectral_data, float)

        t_clean = str(name).strip()
        is_baseline = ("baseline" in t_clean.lower())

        if is_baseline:
            # Align previous spectrum to current x (in case a crop or resample happened)
            if (len(x_prev) != len(x)) or (not np.allclose(x_prev, x, rtol=0, atol=1e-9)):
                y_prev_aligned = np.interp(x, x_prev, y_prev)
            else:
                y_prev_aligned = y_prev

            # Estimated baseline = previous - corrected
            baseline_est = y_prev_aligned - y

            # --- PREVIEW PANEL (before corrected panel) ---
            # Draw baseline first, then previous spectrum on top (solid lines)
            stages.append((f"{i} – Baseline (preview)", x, baseline_est, y_prev_aligned))

            # --- CORRECTED PANEL (actual step result) ---
            stages.append((f"{i} – {t_clean}", x, y, None))
        else:
            # Normal step panel
            stages.append((f"{i} – {t_clean}", x, y, None))

        # carry forward for next iteration
        sc_prev, x_prev, y_prev = sc_curr, x, y

    # Locate AUC panel (first title containing 'auc')
    auc_panel_index = next((k for k, (t, _, __, ___) in enumerate(stages) if "auc" in t.lower()), None)

    # Layout
    n = len(stages)
    ncols = 3 if n >= 3 else n
    nrows = int(ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 3.6 * nrows), squeeze=False)

    for idx, ax in enumerate(axes.flat):
        if idx >= n:
            ax.axis("off")
            continue

        title_i, x, y, y2 = stages[idx]
        # Plot baseline/primary first...
        ax.plot(x, y, lw=1.6)              # solid line
        # ...then overlay previous spectrum so it's on top (solid line)
        if y2 is not None:
            ax.plot(x, y2, lw=1.6)         # solid line

        ax.set_title(title_i, fontsize=11)
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.grid(True, linestyle=":", alpha=0.25)

        # Minimal legend for the preview panel
        if "baseline (preview)" in title_i.lower():
            ax.legend(["baseline", "previous"], fontsize=9, frameon=True, framealpha=0.8)

        # Shade and annotate AUC on the AUC step
        if auc_panel_index is not None and idx == auc_panel_index:
            try:
                auc_val = float(np.trapz(y, x))
            except Exception:
                auc_val = float(np.nan)
            ax.fill_between(x, 0, y, alpha=0.20, linewidth=0)
            ax.text(0.02, 0.95, f"AUC ≈ {auc_val:.3g}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="0.7", alpha=0.8))

    if suptitle:
        fig.suptitle(suptitle, y=0.995, fontsize=12)

    fig.tight_layout()
    return fig



def _overlay_fig(raw_wave, raw_int, pf_wave, pf_int, pc_wave, pc_int, title_suffix):
    """Overlay raw vs processed(no crop) vs processed+cropped with visible styles."""
    plt.figure(figsize=(10, 6))
    plt.plot(raw_wave, _minmax_for_plot(raw_int), label="Raw (scaled)", **_style_for("raw"))
    plt.plot(pf_wave,  _minmax_for_plot(pf_int),  label="Processed (no crop, scaled)", ls="--", lw=1.6, alpha=0.95, zorder=3)
    plt.plot(pc_wave,  _minmax_for_plot(pc_int),  label="Processed + Cropped (scaled)", ls="None", marker=".", ms=2.6, alpha=0.95, zorder=4)
    plt.title(f"Pipeline overlay — {title_suffix}")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Scaled intensity (0–1)")
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def _unit_normalise_container(sc):
    """Normalise a SpectralContainer to unit max-abs within its current window."""
    w = np.asarray(sc.spectral_axis, dtype=float)
    y = np.asarray(sc.spectral_data, dtype=float)
    denom = np.max(np.abs(y))
    if not (np.isfinite(denom) and denom > 0):
        denom = 1.0
    y = y / denom
    sc_out = ramanspy.SpectralContainer(y, w)
    # preserve metadata if present
    if hasattr(sc, "metadata"):
        try: sc_out.metadata = sc.metadata
        except Exception: pass
    return sc_out

def TreatSpectra(data_dict, Save_folder, Preprocess,
                 FP_full, FP_crop, EXT_full, EXT_crop,
                 colours, linestyles, plotall_treat, treatmentorder,
                 SN, step):
    """
    Apply preprocessing to every spectrum.

    Processing:
      - All samples and all spectra are ALWAYS processed and saved back into data_dict.

    Visual debugging (plotting only):
      - SN (int | list[int] | None): if provided, only PLOT spectra from these sample number(s).
        Processing still runs for every sample regardless of SN.
      - step (int | None): if >1, only PLOT every `step`-th spectrum by index (0, step, 2*step, ...).
        Processing still runs for every spectrum.

    Preprocess (order-agnostic): list like ['despike','smooth','baseline','normalise'].

    treatmentorder:
      - 'before' (default): do steps on full range -> final crop -> ROI re-normalise
      - 'after'           : crop first to ROI      -> do steps    -> ROI re-normalise

    Plotting (plotall_treat):
      'pdf' -> one PDF per sample at Save_folder/Outputs
      'screen' -> interactive
      'None' -> no plots

    Outputs per sample (unchanged keys):
      - {FP,EXT}_Spectra_Treated_Full : processed w/out final crop (for PCAorder='Whole')
      - {FP,EXT}_Spectra_Treated      : processed + cropped to analysis window, ROI-normalised
    """
    selected = _normalize_steps(Preprocess)
    order_mode = str(treatmentorder).strip().lower()
    if order_mode not in ("before", "after"):
        print(f"[TreatSpectra] Unknown treatmentorder='{treatmentorder}', using 'before'.")
        order_mode = "before"

    # Prepare output directory only if we will write PDFs
    outdir = None
    if plotall_treat == "pdf":
        outdir = Path(Save_folder) / "Outputs"
        outdir.mkdir(parents=True, exist_ok=True)

    # Normalise SN to a set for PLOTTING FILTER ONLY (do NOT filter processing)
    if SN is None:
        plot_samples = None  # means: plot for all samples
    else:
        if isinstance(SN, (list, tuple, set)):
            plot_samples = set(SN)
        else:
            plot_samples = {SN}

    for sample_num, sample_data in data_dict.items():
        # Prepare storage for this sample
        treated_full = {"FP_Spectra": {}, "EXT_Spectra": {}}
        treated_crop = {"FP_Spectra": {}, "EXT_Spectra": {}}

        # PDF context only for pdf mode (create one per sample)
        pdf_ctx = None
        if plotall_treat == "pdf":
            pdf_path = (outdir / f"Sample_{sample_num}_Treatedspectra.pdf")
            pdf_ctx = PdfPages(pdf_path)

        try:
            for region_key, region_label, full_rng, crop_rng in [
                ("FP_Spectra",  "FP",  FP_full,  FP_crop),
                ("EXT_Spectra", "EXT", EXT_full, EXT_crop),
            ]:
                if region_key not in sample_data:
                    continue

                # Build per-region pipelines from selected steps
                # step_fns_full:  initial full crop + selected steps (no final crop)
                # step_fns_final: step-by-step incl. final analysis crop as last fn
                step_fns_full, titles_full, step_fns_final, titles_final = _build_region_steps(
                    full_rng, crop_rng, selected
                )
                crop_fn = step_fns_final[-1]  # the analysis-window crop callable

                # Stable order along the line for consistent "every n-th" plotting
                items = sorted(
                    sample_data[region_key].items(),
                    key=lambda kv: (round(kv[0][0], 9), round(kv[0][1], 9))
                )

                for idx, ((x, y), spec) in enumerate(items):
                    # Decide whether to plot this spectrum:
                    # - plotting must be enabled
                    # - sample must be allowed by SN filter (if provided)
                    # - respect the step subsampling if step>1
                    plot_allowed_sample = (plot_samples is None) or (sample_num in plot_samples)
                    plot_step_ok = (step in (None, 0, 1)) or (isinstance(step, int) and step > 1 and idx % step == 0)
                    plot_this = (plotall_treat != "None") and plot_allowed_sample and plot_step_ok

                    wave_sorted, inten_sorted = map(
                        list, zip(*sorted(zip(spec["Wave"], spec["Intensity"])))
                    )
                    sc_raw = ramanspy.SpectralContainer(inten_sorted, wave_sorted)
                    sc_raw.metadata = {"x": x, "y": y, "index": idx}

                    if order_mode == "before":
                        # (A) steps on full range -> final crop -> ROI re-normalise
                        sc_full = sc_raw
                        for fn in step_fns_full:
                            sc_full = fn(sc_full)

                        sc_final = crop_fn(sc_full)
                        sc_final_norm = _unit_normalise_container(sc_final)

                        treated_full[region_key][(x, y)] = sc_full
                        treated_crop[region_key][(x, y)] = sc_final_norm

                        if plot_this:
                            xy = (f"Sample {sample_num} — {region_label} — i={idx} — "
                                  f"x={np.round(x,2)}, y={np.round(y,2)}")
                            fig_dbg = _pipeline_debug_fig(sc_raw, step_fns_final, titles_final, xy + " (before)")
                            if plotall_treat == "pdf":
                                pdf_ctx.savefig(fig_dbg); plt.close(fig_dbg)
                            else:
                                plt.show(); plt.close(fig_dbg)

                            fig_ov = _overlay_fig(
                                np.asarray(sc_raw.spectral_axis,float),  np.asarray(sc_raw.spectral_data,float),
                                np.asarray(sc_full.spectral_axis,float), np.asarray(sc_full.spectral_data,float),
                                np.asarray(sc_final_norm.spectral_axis,float), np.asarray(sc_final_norm.spectral_data,float),
                                xy + " (before)"
                            )
                            if plotall_treat == "pdf":
                                pdf_ctx.savefig(fig_ov); plt.close(fig_ov)
                            else:
                                plt.show(); plt.close(fig_ov)

                    else:  # order_mode == 'after'
                        # (B) crop first -> then steps -> ROI re-normalise
                        sc_crop_first = crop_fn(sc_raw)
                        sc_proc = sc_crop_first
                        for fn in step_fns_full:
                            sc_proc = fn(sc_proc)

                        sc_final_norm = _unit_normalise_container(sc_proc)

                        # Also provide a treated_full by applying steps on the uncropped raw spectrum
                        sc_full = sc_raw
                        for fn in step_fns_full:
                            sc_full = fn(sc_full)

                        treated_full[region_key][(x, y)] = sc_full
                        treated_crop[region_key][(x, y)] = sc_final_norm

                        if plot_this:
                            xy = (f"Sample {sample_num} — {region_label} — i={idx} — "
                                  f"x={np.round(x,2)}, y={np.round(y,2)}")
                            fig_dbg = _pipeline_debug_fig(
                                sc_raw, [crop_fn] + step_fns_full, ["Crop(ROI)"] + titles_full, xy + " (after)"
                            )
                            if plotall_treat == "pdf":
                                pdf_ctx.savefig(fig_dbg); plt.close(fig_dbg)
                            else:
                                plt.show(); plt.close(fig_dbg)

                            fig_ov = _overlay_fig(
                                np.asarray(sc_raw.spectral_axis,float),  np.asarray(sc_raw.spectral_data,float),
                                np.asarray(sc_full.spectral_axis,float), np.asarray(sc_full.spectral_data,float),
                                np.asarray(sc_final_norm.spectral_axis,float), np.asarray(sc_final_norm.spectral_data,float),
                                xy + " (after)"
                            )
                            if plotall_treat == "pdf":
                                pdf_ctx.savefig(fig_ov); plt.close(fig_ov)
                            else:
                                plt.show(); plt.close(fig_ov)

            # Optional simple cover at end of sample
            if plotall_treat == "pdf" and pdf_ctx is not None:
                cover = plt.figure(figsize=(10, 6))
                cover.text(0.5, 0.6, f"Sample {sample_num}", ha='center', va='center', fontsize=18)
                cover.text(
                    0.5, 0.5,
                    f"Order: {order_mode}  |  Regions: "
                    f"{' & '.join([k.split('_')[0] for k in ['FP_Spectra','EXT_Spectra'] if k in sample_data])}",
                    ha='center', va='center', fontsize=12
                )
                cover.tight_layout()
                pdf_ctx.savefig(cover); plt.close(cover)

        finally:
            if plotall_treat == "pdf" and pdf_ctx is not None:
                pdf_ctx.close()

        # Save back to dict for downstream use (same keys as before)
        for rk in ["FP_Spectra", "EXT_Spectra"]:
            data_dict[sample_num][f"{rk}_Treated_Full"] = treated_full[rk]  # processed, no final crop
            data_dict[sample_num][f"{rk}_Treated"]      = treated_crop[rk]  # processed, cropped, ROI-normalised

    return data_dict


def TrimRegion(data_dict, sample_manifest_df, Colours, Linestyles, SN, step):
    for sample_num, sample_data in data_dict.items():
        manifest_row = sample_manifest_df[sample_manifest_df["Sample Number"].astype(str) == sample_num]
        if manifest_row.empty:
            print(f"Sample {sample_num} not found in manifest")
            continue

        dermis_x_start = manifest_row["dermis x"].values[0]
        dermis_y_start = manifest_row["dermis y"].values[0]
        dermis_x_end   = manifest_row["epi x"].values[0]
        dermis_y_end   = manifest_row["epi y"].values[0]
        dermis_length_reported = manifest_row["leng dermis"].values[0]

        direction = manifest_row["direction"].values[0].strip().lower()

        for region_key in ["FP_Spectra_Treated", "EXT_Spectra_Treated"]:
            if region_key not in sample_data:
                continue

            spectra_dict = sample_data[region_key]
            coords = np.array(list(spectra_dict.keys()))  # shape (N,2)

            # Trim dermis
            dist_start = np.linalg.norm(coords - np.array([dermis_x_start, dermis_y_start]), axis=1)
            dist_end   = np.linalg.norm(coords - np.array([dermis_x_end, dermis_y_end]), axis=1)

            idx_start = np.argmin(dist_start)
            idx_end   = np.argmin(dist_end)

            i1, i2 = sorted([idx_start, idx_end])
            trimmed_keys = [tuple(coords[i]) for i in range(i1, i2+1)]

            if direction == "back":
                trimmed_keys = list(reversed(trimmed_keys))

            trimmed_spectra = {k: spectra_dict[k] for k in trimmed_keys}
            data_dict[sample_num][f"{region_key}_Dermis"] = trimmed_spectra

            # length check
            if abs(len(trimmed_spectra) - dermis_length_reported) > 2:
                print(f"Error! Mismatch for {sample_num} {region_key}: "
                      f"manifest={int(dermis_length_reported)} vs trimmed={len(trimmed_spectra)}")

        if SN is not None and str(SN) == sample_num:
            subtype = data_dict[sample_num].get("Subtype")
            colour = Colours.get(subtype, None) if Colours else None
            linestyle = Linestyles.get(subtype, None) if Linestyles else None
            for region_key in ["FP_Spectra_Treated_Dermis", "EXT_Spectra_Treated_Dermis"]:
                if region_key not in data_dict[sample_num]:
                    continue
                plt.figure(figsize=(10, 6))
                spectra_dict = data_dict[sample_num][region_key]
                for i, ((x, y), spectrum) in enumerate(spectra_dict.items()):
                    if step is None or i % step == 0:
                        plt.plot(spectrum.spectral_axis, spectrum.spectral_data,
                                 label=f"x={np.round(x,2)}, y={np.round(y,2)}",
                                 color=colour, linestyle=linestyle)
                plt.title(f"{region_key} spectra for Sample {sample_num}")
                plt.xlabel("Wavenumber (cm⁻¹)")
                plt.ylabel("Intensity")
                plt.legend(fontsize="x-small", loc="upper right", ncol=2)
                plt.tight_layout()
                plt.show()

    return data_dict

def BinData(data_dict, NBins, Colours, Linestyles, ST=None, region=None, Binno=None, SN=None):
    """
    Bin trimmed dermis spectra for each subtype and region into NBins along scan order.
    - Preserves the insertion order from TrimRegion (no sorting).
    - Interpolates spectra in each bin to a common wave axis before averaging.
    - Stores per-sample bins back into data_dict under keys "FP_Bins" and "EXT_Bins".
    Returns:
        binned_avg[subtype][region_short] -> list of dicts {"Wave", "Intensity"} per bin
    """
    subtype_samples = {}
    for sample_num, sample_data in data_dict.items():
        subtype = sample_data.get("Subtype")
        if subtype is None:
            continue
        subtype_samples.setdefault(subtype, []).append(sample_num)

    binned_avg = {}
    for subtype, samples in subtype_samples.items():
        print(f"Processing subtype: {subtype}")
        binned_avg[subtype] = {"FP": [], "EXT": []}

        for region_key in ["FP_Spectra_Treated_Dermis", "EXT_Spectra_Treated_Dermis"]:
            region_short = region_key.split("_")[0]  # "FP" or "EXT"
            samples_binned = []   # list of [bin0, bin1, ...] per-sample

            for sample_num in samples:
                if region_key not in data_dict[sample_num]:
                    continue

                spectra_dict = data_dict[sample_num][region_key]
                keys_order = list(spectra_dict.keys())  # preserve scan order
                n_points = len(keys_order)
                if n_points == 0:
                    continue

                N_eff = min(NBins, n_points)
                index_bins = np.array_split(np.arange(n_points), N_eff)

                ref_axis = np.asarray(spectra_dict[keys_order[0]].spectral_axis, dtype=float)

                bin_averages = []
                for idxs in index_bins:
                    if len(idxs) == 0:
                        bin_averages.append(None)
                        continue

                    bin_intensities = []
                    for idx in idxs:
                        sc = spectra_dict[keys_order[idx]]
                        wav = np.asarray(sc.spectral_axis, dtype=float)
                        inten = np.asarray(sc.spectral_data, dtype=float)
                        if wav.shape != ref_axis.shape or not np.allclose(wav, ref_axis, rtol=0, atol=1e-9):
                            inten = np.interp(ref_axis, wav, inten)
                        bin_intensities.append(inten)

                    avg_intensity = np.mean(np.vstack(bin_intensities), axis=0)
                    bin_averages.append({"Wave": ref_axis, "Intensity": avg_intensity})

                samples_binned.append(bin_averages)

                # --- Save bins back to data_dict per sample ---
                data_dict[sample_num][f"{region_short}_Bins"] = bin_averages

            # Merge across samples (per bin) for subtype-level average
            if not samples_binned:
                binned_avg[subtype][region_short] = []
                continue

            N_bins_final = len(samples_binned[0])
            bins_merged = []
            for b in range(N_bins_final):
                bin_stack = [s[b]["Intensity"] for s in samples_binned if s[b] is not None]
                if not bin_stack:
                    continue
                bin_stack = np.vstack(bin_stack)
                mean_intensity = np.mean(bin_stack, axis=0)
                wave_axis = None
                for s in samples_binned:
                    if s[b] is not None:
                        wave_axis = s[b]["Wave"]
                        break
                bins_merged.append({"Wave": wave_axis, "Intensity": mean_intensity})

            binned_avg[subtype][region_short] = bins_merged

    # --- Plotting (optional) ---
    if ST is not None or SN is not None:
        plt.figure(figsize=(8, 5))

        # Subtype-level plotting
        if ST is not None:
            if isinstance(ST, str):
                ST = [ST]
            if isinstance(Binno, int):
                Binno = [Binno]

            for st in ST:
                if st not in binned_avg:
                    print(f"Warning: subtype {st} not found in binned data.")
                    continue
                if region not in binned_avg[st]:
                    print(f"Warning: region {region} not found for subtype {st}.")
                    continue
                bins = binned_avg[st][region]
                colour = Colours.get(st, None) if Colours else None
                linestyle = Linestyles.get(st, None) if Linestyles else None

                for b in Binno:
                    if b < 0 or b >= len(bins):
                        print(f"Warning: bin number {b} out of range for subtype {st} region {region}.")
                        continue
                    spectrum = bins[b]
                    plt.plot(spectrum["Wave"], spectrum["Intensity"],
                             label=f"{st} bin {b}", color=colour, linestyle=linestyle)

        # Per-sample plotting
        if SN is not None and str(SN) in data_dict:
            for reg_short in ["FP", "EXT"]:
                key = f"{reg_short}_Bins"
                if key not in data_dict[str(SN)]:
                    continue
                bins = data_dict[str(SN)][key]
                for i, spectrum in enumerate(bins):
                    if spectrum is None:
                        continue
                    plt.plot(spectrum["Wave"], spectrum["Intensity"],
                             label=f"Sample {SN} {reg_short} bin {i}")

        plt.title(f"Binned Spectrum (Region={region})")
        plt.xlabel("Wavenumber (cm⁻¹)")
        plt.ylabel("Intensity")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return data_dict, binned_avg

def AverageSpectra(data_dict,Save_folder,PCAorder,removeoutliers, plot_mode_AverageSpectra="screen",region="FP"):
    """
    Step 7.5: Build mean ± SD spectra at two levels:
      (A) per-sample (one mean spectrum per sample),
      (B) per-subtype (mean across repeats).

    Selection (PCAorder):
      - 'Trim'  -> use {FP,EXT}_Spectra_Treated_Dermis
      - else    -> use {FP,EXT}_Spectra_Treated_Full

    Region:
      - 'FP' | 'EXT' | 'both'   (does each region independently)

    Plotting:
      - 'pdf'    -> save two PDFs per region: per-sample & per-subtype (+ overview PDFs)
      - 'screen' -> show figures interactively
      - 'None'   -> no plotting

    Outliers:
      - removeoutliers: iterable of sample numbers (str or int) to exclude.

    Stores results under:
      data_dict["_AvgSpectra"][region]["per_sample"][sample_num] = {
          "wave","mean","std","n","order"
      }
      data_dict["_AvgSpectra"][region]["per_subtype"][subtype] = {
          "wave","mean","std","n","order"
      }
    """

    # ---- helpers assumed present in your module ----
    # _region_keys(use_dermis, region) -> yields (label, dict_key)
    # Each spectra_dict: {(x,y): SpectralContainer}, preserving TrimRegion order

    def _interp_to_ref(ref_x, x, y):
        """Return y interpolated to ref_x if x differs; else original y."""
        if (len(x) != len(ref_x)) or (not np.allclose(x, ref_x, rtol=0, atol=1e-9)):
            return np.interp(ref_x, x, y)
        return y

    def _stack_X_from_spectra_dict(spectra_dict):
        """Return stacked matrix (N, W) and wave axis from a spectra_dict."""
        if not spectra_dict:
            return None, None
        keys_order = list(spectra_dict.keys())  # insertion order
        # use first as reference
        w0 = np.asarray(spectra_dict[keys_order[0]].spectral_axis, float)
        rows = []
        for k in keys_order:
            sc = spectra_dict[k]
            w = np.asarray(sc.spectral_axis, float)
            y = np.asarray(sc.spectral_data, float)
            y = _interp_to_ref(w0, w, y)
            rows.append(y)
        return np.vstack(rows), w0

    # --- normalise plotting mode once, then use throughout ---
    mode = (plot_mode_AverageSpectra if isinstance(plot_mode_AverageSpectra, str) else str(plot_mode_AverageSpectra))
    mode = mode.strip().lower()
    if mode not in ("none", "pdf", "screen"):
        print(f"[AverageSpectra] Unknown plot mode '{plot_mode_AverageSpectra}', using 'none'.")
        mode = "none"

    use_dermis = (str(PCAorder).strip().lower() == "trim")
    regions_to_do = ("FP", "EXT") if str(region).lower() == "both" else (region.upper(),)

    outdir = Path(Save_folder) / "Outputs"
    if mode == "pdf":
        outdir.mkdir(parents=True, exist_ok=True)

    # storage scaffold
    store_key = "_AvgSpectra"
    data_dict.setdefault(store_key, {})
    for reg in regions_to_do:
        data_dict[store_key].setdefault(reg, {})
        data_dict[store_key][reg].setdefault("per_sample", {})
        data_dict[store_key][reg].setdefault("per_subtype", {})

    # PDFs per region (per-sample & per-subtype)
    pdf_ps = {}   # per-sample
    pdf_pt = {}   # per-subtype
    pdf_paths_ps = {}
    pdf_paths_pt = {}
    if mode == "pdf":
        for reg in regions_to_do:
            p1 = outdir / f"AverageSpectra_perSample_{reg}.pdf"
            p2 = outdir / f"AverageSpectra_perSubtype_{reg}.pdf"
            pdf_ps[reg] = PdfPages(p1)
            pdf_pt[reg] = PdfPages(p2)
            pdf_paths_ps[reg] = str(p1)
            pdf_paths_pt[reg] = str(p2)

    excl = set(str(s) for s in (removeoutliers or []))
    title_suffix = "Dermis" if use_dermis else "Whole line"

    # ---------- PASS 1: per-sample ----------
    # Also cache per-sample means to later build per-subtype means on a common axis
    per_sample_means = {reg: {} for reg in regions_to_do}  # reg -> {sample: (wave, mean)}
    subtype_of_sample = {}

    for reg in regions_to_do:
        rlabel, rkey = next(_region_keys(use_dermis, reg))

        for sample_num, sdict in data_dict.items():
            if not isinstance(sdict, dict):
                continue
            sid = str(sample_num)
            if sid in excl:
                continue

            subtype = sdict.get("Subtype", None)
            if subtype is None:
                continue
            subtype_of_sample[sid] = subtype

            spectra_dict = sdict.get(rkey, None)
            if not spectra_dict:
                continue

            X, wave = _stack_X_from_spectra_dict(spectra_dict)
            if X is None or X.shape[0] == 0:
                continue

            mean = X.mean(axis=0)
            std  = X.std(axis=0, ddof=1) if X.shape[0] > 1 else np.zeros_like(mean)
            n    = int(X.shape[0])

            # store
            data_dict[store_key][reg]["per_sample"][sid] = {
                "wave": wave, "mean": mean, "std": std, "n": n, "order": title_suffix,
                "subtype": subtype
            }
            per_sample_means[reg][sid] = (wave, mean)

            # plot (per-sample)
            if mode != "none":
                fig = plt.figure(figsize=(8, 4))
                ax = fig.add_subplot(111)
                ax.plot(wave, mean, lw=1.6, label=f"Sample {sid}")
                ax.fill_between(wave, mean-std, mean+std, alpha=0.25, label="±1 SD")
                ax.set_xlabel("Wavenumber (cm⁻¹)")
                ax.set_ylabel("Intensity (a.u.)")
                ax.set_title(f"{reg} — Mean ± SD — Sample {sid} ({title_suffix})")
                ax.legend()
                fig.tight_layout()

                if mode == "screen":
                    plt.show(); plt.close(fig)
                else:  # pdf
                    pdf_ps[reg].savefig(fig); plt.close(fig)

    # ---------- PASS 2: per-subtype ----------
    for reg in regions_to_do:
        # group sample means by subtype
        groups = {}
        for sid, (wave, mu) in per_sample_means[reg].items():
            st = subtype_of_sample.get(sid, "Unknown")
            groups.setdefault(st, []).append((wave, mu))

        for st, items in groups.items():
            if len(items) == 0:
                continue
            # choose a reference wave (first)
            wref = np.asarray(items[0][0], float)
            means_stack = []
            for w, mu in items:
                mu_i = mu if (len(w) == len(wref) and np.allclose(w, wref, rtol=0, atol=1e-9)) else np.interp(wref, w, mu)
                means_stack.append(mu_i)
            M = np.vstack(means_stack)
            mean_st = M.mean(axis=0)
            std_st  = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(mean_st)
            n_st    = int(M.shape[0])

            # store
            data_dict[store_key][reg]["per_subtype"][st] = {
                "wave": wref, "mean": mean_st, "std": std_st, "n": n_st, "order": title_suffix
            }

            # plot (per-subtype)
            if mode != "none":
                fig = plt.figure(figsize=(8, 4))
                ax = fig.add_subplot(111)
                ax.plot(wref, mean_st, lw=1.8, label=f"{st} (n={n_st} repeats)")
                ax.fill_between(wref, mean_st - std_st, mean_st + std_st, alpha=0.25, label="±1 SD")
                ax.set_xlabel("Wavenumber (cm⁻¹)")
                ax.set_ylabel("Intensity (a.u.)")
                ax.set_title(f"{reg} — Mean ± SD — Subtype {st} ({title_suffix})")
                ax.legend()
                fig.tight_layout()

                if mode == "screen":
                    plt.show(); plt.close(fig)
                else:  # pdf
                    pdf_pt[reg].savefig(fig); plt.close(fig)

    # close PDFs
    if mode == "pdf":
        for reg in regions_to_do:
            if reg in pdf_ps and pdf_ps[reg] is not None:
                pdf_ps[reg].close()
                print(f"[AverageSpectra] Saved per-sample PDF for {reg} to {pdf_paths_ps[reg]}")
            if reg in pdf_pt and pdf_pt[reg] is not None:
                pdf_pt[reg].close()
                print(f"[AverageSpectra] Saved per-subtype PDF for {reg} to {pdf_paths_pt[reg]}")

    # ---------- FINAL OVERVIEW PLOTS ----------
    # Entire block is skipped when mode == "none"
    if mode != "none":
        for reg in regions_to_do:
            # ---- 1. Multi-panel per-subtype (all repeats overlaid, with ±SD fill) ----
            groups = {}
            for sid, d in data_dict[store_key][reg]["per_sample"].items():
                st = d.get("subtype", "Unknown")
                groups.setdefault(st, []).append((sid, d))

            if len(groups) > 0:
                n_subtypes = len(groups)
                ncols = 2
                nrows = int(np.ceil(n_subtypes / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 3.5*nrows), squeeze=False)

                for ax, (st, items) in zip(axes.flat, groups.items()):
                    for sid, d in items:
                        wave = d["wave"]
                        mu   = d["mean"]
                        sd   = d.get("std", None)
                        ax.plot(wave, mu, lw=1.2, alpha=0.9, label=f"Sample {sid}")
                        if sd is not None:
                            ax.fill_between(wave, mu - sd, mu + sd, alpha=0.20, linewidth=0)
                    ax.set_title(f"Subtype {st}")
                    ax.set_xlabel("Wavenumber (cm⁻¹)")
                    ax.set_ylabel("Intensity (a.u.)")
                    ax.grid(True, linestyle=":", alpha=0.3)
                    if len(items) <= 6:
                        ax.legend(fontsize=8)

                # hide any unused axes
                for ax in axes.flat[len(groups):]:
                    ax.axis("off")

                fig.suptitle(f"{reg} — Per-sample spectra grouped by subtype ({title_suffix})", y=0.995)

                if mode == "screen":
                    plt.tight_layout(); plt.show(); plt.close(fig)
                else:  # pdf
                    pdf_path = outdir / f"AverageSpectra_perSubtypePanels_{reg}.pdf"
                    with PdfPages(pdf_path) as pdf:
                        pdf.savefig(fig)
                    plt.close(fig)
                    print(f"[AverageSpectra] Saved subtype panel PDF for {reg} to {pdf_path}")

            # ---- 2. Single overlay of per-subtype averages ----
            if data_dict[store_key][reg]["per_subtype"]:
                fig = plt.figure(figsize=(8,5))
                ax = fig.add_subplot(111)
                for st, d in data_dict[store_key][reg]["per_subtype"].items():
                    wave, mu, sd = d["wave"], d["mean"], d["std"]
                    ax.plot(wave, mu, lw=1.8, label=f"{st} (n={d['n']})")
                    ax.fill_between(wave, mu-sd, mu+sd, alpha=0.2)
                ax.set_xlabel("Wavenumber (cm⁻¹)")
                ax.set_ylabel("Intensity (a.u.)")
                ax.set_title(f"{reg} — Averaged spectra per subtype ({title_suffix})")
                ax.legend()
                ax.grid(True, linestyle=":", alpha=0.3)
                fig.tight_layout()

                if mode == "screen":
                    plt.show(); plt.close(fig)
                else:  # pdf
                    pdf_path = outdir / f"AverageSpectra_AllSubtypeOverlay_{reg}.pdf"
                    with PdfPages(pdf_path) as pdf:
                        pdf.savefig(fig)
                    plt.close(fig)
                    print(f"[AverageSpectra] Saved subtype overlay PDF for {reg} to {pdf_path}")

    return data_dict


# ---------------------------------------------------------------------------------------------------------
# -----------------------------------------------PCA analysis----------------------------------------------
# ---------------------------------------------------------------------------------------------------------

def _stack_sample_matrix(spectra_dict):
    """Return X (n_spectra x n_waves) and a common wave axis for a single sample/region."""
    if not spectra_dict:
        return None, None
    keys_order = list(spectra_dict.keys())  # preserves TrimRegion order
    ref_axis = np.asarray(spectra_dict[keys_order[0]].spectral_axis, dtype=float)
    rows = []
    for k in keys_order:
        sc = spectra_dict[k]
        wav = np.asarray(sc.spectral_axis, dtype=float)
        inten = np.asarray(sc.spectral_data, dtype=float)
        if wav.shape != ref_axis.shape or not np.allclose(wav, ref_axis, rtol=0, atol=1e-9):
            inten = np.interp(ref_axis, wav, inten)
        rows.append(inten)
    return np.vstack(rows), ref_axis

def _region_keys(use_dermis, want_regions):
    """Yield (region_label, dict_key) for FP/EXT/both according to PCAorder."""
    if want_regions.lower() == "both":
        regions = ("FP", "EXT")
    else:
        regions = (want_regions.upper(),)
    for reg in regions:
        yield reg, (f"{reg}_Spectra_Treated_Dermis" if use_dermis else f"{reg}_Spectra_Treated_Full")

def _is_excluded(sample_id, removeoutliers):
    """Return True if sample_id is in the user-provided removeoutliers list."""
    if removeoutliers is None:
        return False
    ban = {str(x).strip() for x in removeoutliers}
    return str(sample_id).strip() in ban

def PCA(meancentre, Save_folder, PCAorder, data_dict, Colours, Linestyles,
        n_components, plotall_PCA, region="FP"):
    """
    Run PCA per sample (FP/EXT or both regions), with optional mean-centering.

    Data selection:
      - PCAorder == 'Trim'  -> use {FP,EXT}_Spectra_Treated_Dermis
      - else                -> use {FP,EXT}_Spectra_Treated_Full
    Region:
      - region = 'FP' | 'EXT' | 'both'
    Centering:
      - meancentre='with'     -> standard PCA (mean-centered)
      - meancentre='without'  -> NO centering (TruncatedSVD)

    Results stored in:
      data_dict["_PCA_per_sample"][region][sample_num] = {
          'wave', 'scores', 'loadings', 'expl',
          'centre', 'order', 'subtype'
      }

    Plotting:
      - plotall='screen' → plots inline to screen
      - plotall='pdf'    → saves ALL outputs to one PDF
      - plotall='None'   → no plotting at all
    """

    store_key = "_PCA_per_sample"
    use_dermis = (str(PCAorder).strip().lower() == "trim")
    centre = str(meancentre).strip().lower()
    if centre not in ("with", "without"):
        print(f"Warning: meancentre '{meancentre}' not recognised; using 'with'")
        centre = "with"

    # prepare storage containers
    data_dict.setdefault(store_key, {}).setdefault("FP", {})
    data_dict[store_key].setdefault("EXT", {})

    # PDF setup if required
    pdf_writer = None
    if plotall_PCA == "pdf":
        outdir = Path(Save_folder) / "Outputs"
        outdir.mkdir(parents=True, exist_ok=True)
        pdf_path = outdir / f"AllSamples_PCA.pdf"
        pdf_writer = PdfPages(pdf_path)

    for sample_num, sdict in data_dict.items():
        if not isinstance(sdict, dict):
            continue
        subtype = sdict.get("Subtype", None)
        base_colour = Colours.get(subtype, None) if Colours else None
        base_linestyle = Linestyles.get(subtype, None) if Linestyles else None

        # gather FP/EXT (or single region) for THIS sample
        per_region = []
        for reg, rkey in _region_keys(use_dermis, region):
            spectra_dict = sdict.get(rkey, None)
            if not spectra_dict:
                continue

            X, wave = _stack_sample_matrix(spectra_dict)
            if X is None or X.shape[0] < 3:
                print(f"Skipping PCA: sample {sample_num} {rkey} has too few spectra.")
                continue

            # choose estimator
            if centre == "with":
                est = _SKPCA(n_components=int(min(n_components, X.shape[0], X.shape[1])),
                             svd_solver="auto", random_state=0)
                scores = est.fit_transform(X)
                loadings = est.components_.T
                expl = est.explained_variance_ratio_
            else:
                est = _SVD(n_components=int(min(n_components, X.shape[0], X.shape[1])),
                           random_state=0)
                scores = est.fit_transform(X)
                loadings = est.components_.T
                svals = est.singular_values_
                expl = (svals**2) / np.sum((X**2))

            per_region.append(dict(name=reg, wave=wave, scores=scores,
                                   loadings=loadings, expl=expl))

            # --- STORE RESULTS ---
            data_dict[store_key][reg][str(sample_num)] = {
                "wave": wave,
                "scores": scores,
                "loadings": loadings,
                "expl": expl,
                "centre": centre,
                "order": "Dermis" if use_dermis else "Whole",
                "subtype": subtype
            }

        if not per_region or plotall_PCA == "None":
            continue  # skip plotting entirely

        title_suffix = "Dermis" if use_dermis else "Whole line"

        # --- SCREE ---
        fig = plt.figure(figsize=(8, 4)); ax = fig.add_subplot(111)
        max_k = max(len(d["expl"]) for d in per_region)
        idx = np.arange(1, max_k + 1)
        region_style = {"FP": dict(alpha=0.65), "EXT": dict(alpha=0.35)}
        for d in per_region:
            k = len(d["expl"]); offset = -0.15 if d["name"] == "FP" else 0.15
            ax.bar(idx[:k] + offset, d["expl"], width=0.3,
                   label=f"{d['name']} (EVR)", color=base_colour, **region_style.get(d["name"], {}))
            ax.plot(idx[:k] + offset, np.cumsum(d["expl"]),
                    "--o", color="black", alpha=0.6,
                    label=f"{d['name']} (cum.)")
        ax.set_xlabel("Principal Component"); ax.set_ylabel("Variance explained")
        ax.set_title(f"Scree — Sample {sample_num} ({title_suffix}, centre='{centre}')")
        ax.legend(); fig.tight_layout()

        if plotall_PCA == "screen":
            plt.show(); plt.close(fig)
        elif plotall_PCA == "pdf":
            pdf_writer.savefig(fig); plt.close(fig)

        # --- SCORES ---
        fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111)
        for d in per_region:
            sc = d["scores"]
            if sc.shape[1] < 2: continue
            marker = "o" if d["name"] == "FP" else "^"
            ax.plot(sc[:,0], sc[:,1], linestyle=base_linestyle or "-", alpha=0.35, color=base_colour)
            ax.scatter(sc[:,0], sc[:,1], s=20, alpha=0.85, color=base_colour,
                       marker=marker, label=f"{d['name']} ({subtype})")
        ax.set_xlabel("PC1 score"); ax.set_ylabel("PC2 score")
        ax.set_title(f"Scores — Sample {sample_num} ({title_suffix})")
        ax.set_xlim(-0.01, 0.01); ax.set_ylim(-0.01, 0.01)
        ax.legend(); ax.grid(True, linestyle=":", alpha=0.3)
        fig.tight_layout()

        if plotall_PCA == "screen":
            plt.show(); plt.close(fig)
        elif plotall_PCA == "pdf":
            pdf_writer.savefig(fig); plt.close(fig)

        # --- LOADINGS ---
        fig = plt.figure(figsize=(9, 4.5)); ax = fig.add_subplot(111)
        for d in per_region:
            wave = d["wave"]; load = d["loadings"]; alpha = 0.95 if d["name"] == "FP" else 0.55
            ax.plot(wave, load[:,0], label=f"{d['name']} PC1", linewidth=1.7, color=base_colour, alpha=alpha)
            if load.shape[1] >= 2:
                ax.plot(wave, load[:,1], "--", label=f"{d['name']} PC2", linewidth=1.2, color=base_colour, alpha=alpha)
        ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Loading")
        ax.set_ylim(-0.2, 0.25)
        ax.set_title(f"Loadings — Sample {sample_num} ({title_suffix})")
        ax.legend(); fig.tight_layout()

        if plotall_PCA == "screen":
            plt.show(); plt.close(fig)
        elif plotall_PCA == "pdf":
            pdf_writer.savefig(fig); plt.close(fig)

    # close one big PDF if we used it
    if pdf_writer is not None:
        pdf_writer.close()
        print(f"[PCA] Saved combined PDF to {pdf_path}")

    return data_dict

def PCA_poolloadings(data_dict, plotall_poolloadings, Save_folder, removeoutliers, region="FP",
                     norm_mode="maxabs", smooth=True, sg_window=21, sg_poly=3):
    """
    Step 9: Visualise per-sample PCA loadings grouped by subtype (from Step 8 cache).

    PART A (existing):
      - Figure A: PC1 loadings (one subplot per subtype; all repeats overlaid)
      - Figure B: PC2 loadings (one subplot per subtype; all repeats overlaid)

    PART B (new, consensus-style):
      - Figure C: PC1 *normalised + smoothed* loadings, per subtype (repeats overlaid)
      - Figure D: PC2 *normalised + smoothed* loadings, per subtype (repeats overlaid)

    Reads from: data_dict["_PCA_per_sample"][region][sample_num]

    Plot modes:
      - 'pdf'    -> save figures as PDFs in Save_folder/Outputs
      - 'screen' -> show interactively
      - 'None'   -> skip plotting entirely

    Parameters:
      norm_mode: 'maxabs' (default) or 'l2'  – how to normalise each loading curve
      smooth:    whether to apply Savitzky–Golay smoothing (light denoise for visualisation)
      sg_window: SavGol window length (odd, auto-clamped to < n_points)
      sg_poly:   SavGol polyorder (typ. 2–3)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.signal import savgol_filter

    source_key = "_PCA_per_sample"
    cache = data_dict.get(source_key, {}).get(region, {})
    if not cache:
        print(f"[PCA_poolloadings] No PCA results found for region={region}.")
        return data_dict

    # Group samples by subtype
    subtype_to_entries = {}
    for smp, entry in cache.items():
        if _is_excluded(smp, removeoutliers):   # <-- NEW: skip outlier sample
            continue
        st = entry.get("subtype", "Unknown")
        subtype_to_entries.setdefault(st, []).append((smp, entry))

    subtypes = list(subtype_to_entries.keys())
    n_subtypes = len(subtypes)
    ncols = 2
    nrows = int(np.ceil(n_subtypes / ncols))

    # ---------- utilities ----------
    def _sg_params(n, w, p):
        # Enforce valid SavGol params: odd window <= n, > poly
        w = int(w)
        if w % 2 == 0:
            w += 1
        w = max(p + 2 + (1 - (p + 2) % 2), min(w, n - (1 - n % 2)))  # be safe
        w = min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)
        if w < 3:  # too small to smooth; disable
            return None, None
        if p >= w:
            p = max(2, min(3, w - 2))
        return w, p

    def _norm_curve(y):
        y = np.asarray(y, float)
        if norm_mode == "l2":
            d = np.linalg.norm(y)
        else:  # maxabs
            d = np.max(np.abs(y))
        return y / d if d and np.isfinite(d) and d > 0 else y

    def _sign_align(ref, y):
        # Align sign by correlation to reference
        num = np.dot(ref, y)
        return (-y) if num < 0 else y

    # ---------- Helper: build one figure for a given PC index (raw) ----------
    def _plot_pc_panel_raw(pc_index: int, title_suffix: str):
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows), squeeze=False)
        ax_iter = iter(axes.flat)

        for st in subtypes:
            ax = next(ax_iter)
            entries = subtype_to_entries[st]

            has_any = False
            for smp, entry in entries:
                wave = entry["wave"]
                load = entry["loadings"]  # [n_wave, n_comp]
                if load.ndim != 2 or load.shape[1] <= pc_index:
                    continue
                has_any = True
                ls = "-" if pc_index == 0 else "--"
                lab = f"{st} {smp} PC{pc_index+1}"
                ax.plot(wave, load[:, pc_index], linestyle=ls,
                        linewidth=1.2, alpha=0.9, label=lab)

            ax.set_title(f"{st} — {region} — PC{pc_index+1} {title_suffix}")
            ax.set_xlabel("Wavenumber (cm⁻¹)")
            ax.set_ylabel("Loading")
            ax.set_ylim(-0.2, 0.25)
            if has_any:
                ax.legend(fontsize=7, loc="upper right")
            else:
                ax.text(0.5, 0.5, "No PC available", ha="center", va="center",
                        transform=ax.transAxes)

        for ax in ax_iter:
            ax.axis("off")

        fig.tight_layout()
        return fig

    # ---------- Helper: build one figure for a given PC index (normalised + smoothed + sign-aligned) ----------
    def _plot_pc_panel_consensus(pc_index: int, title_suffix: str):
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows), squeeze=False)
        ax_iter = iter(axes.flat)

        for st in subtypes:
            ax = next(ax_iter)
            entries = subtype_to_entries[st]

            # choose first valid repeat as sign reference
            ref_curve = None
            ref_wave = None
            curves = []  # list of (wave, curve, label)
            for smp, entry in entries:
                wave = entry["wave"]
                load = entry["loadings"]
                if load.ndim != 2 or load.shape[1] <= pc_index:
                    continue
                y = load[:, pc_index].astype(float)

                # normalise
                y = _norm_curve(y)

                # smooth (optional)
                if smooth:
                    w, p = _sg_params(len(y), sg_window, sg_poly)
                    if w is not None:
                        try:
                            y = savgol_filter(y, window_length=w, polyorder=p, mode="interp")
                        except Exception:
                            pass

                # establish reference for sign
                if ref_curve is None:
                    ref_curve = y.copy()
                    ref_wave = wave
                else:
                    # If waves differ, interpolate y to ref_wave
                    if (len(wave) != len(ref_wave)) or (not np.allclose(wave, ref_wave, rtol=0, atol=1e-9)):
                        y = np.interp(ref_wave, wave, y)
                    y = _sign_align(ref_curve, y)

                curves.append((ref_wave if ref_wave is not None else wave, y, f"{st} {smp} PC{pc_index+1}"))

            has_any = len(curves) > 0
            for wave, y, lab in curves:
                ls = "-" if pc_index == 0 else "--"
                ax.plot(wave, y, linestyle=ls, linewidth=1.2, alpha=0.95, label=lab)

            ax.set_title(f"{st} — {region} — PC{pc_index+1} (normalised{', smoothed' if smooth else ''})")
            ax.set_xlabel("Wavenumber (cm⁻¹)")
            ax.set_ylabel("Normalised loading")
            ax.set_ylim(-1.0, 1.0)  # fixed for consistency with raw panels
            if has_any:
                ax.legend(fontsize=7, loc="upper right")
            else:
                ax.text(0.5, 0.5, "No PC available", ha="center", va="center",
                        transform=ax.transAxes)

        for ax in ax_iter:
            ax.axis("off")

        fig.tight_layout()
        return fig

    # Skip entirely if asked
    if plotall_poolloadings == "None":
        return data_dict

    # PART A: raw loadings (your original visualisation)
    fig_pc1_raw = _plot_pc_panel_raw(pc_index=0, title_suffix="")
    fig_pc2_raw = _plot_pc_panel_raw(pc_index=1, title_suffix="")

    # PART B: consensus-style (normalised + smoothed + sign-aligned)
    fig_pc1_cons = _plot_pc_panel_consensus(pc_index=0, title_suffix="")
    fig_pc2_cons = _plot_pc_panel_consensus(pc_index=1, title_suffix="")

    # Save or show
    if plotall_poolloadings == "pdf":
        outdir = Path(Save_folder) / "Outputs"
        outdir.mkdir(parents=True, exist_ok=True)
        paths = [
            outdir / f"Subtype_Loadings_PC1_{region}.pdf",
            outdir / f"Subtype_Loadings_PC2_{region}.pdf",
            outdir / f"Subtype_Loadings_PC1_normsmooth_{region}.pdf",
            outdir / f"Subtype_Loadings_PC2_normsmooth_{region}.pdf",
        ]
        figs = [fig_pc1_raw, fig_pc2_raw, fig_pc1_cons, fig_pc2_cons]
        for pth, fg in zip(paths, figs):
            with PdfPages(pth) as pdf:
                pdf.savefig(fg)
            plt.close(fg)
        print("[PCA_poolloadings] Saved:")
        for p in paths:
            print("  ", p)
    elif plotall_poolloadings == "screen":
        for fg in [fig_pc1_raw, fig_pc2_raw, fig_pc1_cons, fig_pc2_cons]:
            plt.show(); plt.close(fg)

    return data_dict

def PCA_poolloadings_quant(
    data_dict,
    Save_folder,
    removeoutliers, 
    region="FP",
    norm_mode="maxabs",    # 'maxabs' or 'l2'
    smooth=True,
    sg_window=21,
    sg_poly=3,
    n_pcs=2,               # how many PCs to quantify (usually 2)
    peak_prom=0.10,        # min prominence on the NORMALISED curve (0..1)
    peak_distance=5,       # min distance (in index points) between peaks
    peak_tol=10.0          # cm^-1 tolerance for consensus clustering
):
    """
    Quantify per-subtype PC loadings (from Step 8 cache) and export to Excel.

    Reads from: data_dict["_PCA_per_sample"][region][<sample_num>]
      (expects keys: 'wave', 'loadings', 'subtype')

    Writes: {Save_folder}/Outputs/PCA_Loadings_Quant_{region}.xlsx with sheets:
      - 'Peaks'         : each repeat's peaks (wavenumber, height, sign)
      - 'Correlations'  : pairwise Pearson r between repeats (per subtype, per PC)
      - 'PeakFrequency' : consensus peaks per subtype (clustered within ±peak_tol cm⁻¹)
    """

    source_key = "_PCA_per_sample"
    cache = data_dict.get(source_key, {}).get(region, {})
    if not cache:
        print(f"[PCA_poolloadings_quant] No PCA results found for region={region}.")
        return data_dict

    # ---------- helpers ----------
    def _sg_params(n, w, p):
        w = int(w)
        if w % 2 == 0: w += 1
        if w >= n: w = n-1 if (n-1) % 2 == 1 else n-2
        if w < 3: return None, None
        if p >= w: p = min(3, w-2)
        return w, max(2, p)

    def _norm_curve(y):
        y = np.asarray(y, float)
        if norm_mode.lower() == "l2":
            d = np.linalg.norm(y)
        else:
            d = np.max(np.abs(y))
        return y / d if (d and np.isfinite(d) and d > 0) else y

    def _sign_align(ref, y):
        return -y if float(np.dot(ref, y)) < 0 else y

    # Group by subtype
    subtype_to_entries = {}
    for smp, entry in cache.items():
        if _is_excluded(smp, removeoutliers):   # <-- NEW
            continue
        st = entry.get("subtype", "Unknown")
        subtype_to_entries.setdefault(st, []).append((str(smp), entry))


    peaks_rows = []
    corr_rows  = []
    freq_rows  = []

    # ---------- main loop: per subtype ----------
    for st, entries in subtype_to_entries.items():
        # establish a reference wave for this subtype (first valid)
        ref_wave = None
        for _, e in entries:
            if "wave" in e:
                ref_wave = np.asarray(e["wave"], float)
                break
        if ref_wave is None: 
            continue

        # for each PC separately
        for pc in range(n_pcs):
            # build aligned, normalised, (smoothed) loadings per repeat
            aligned = {}   # sample -> y (same wave grid)
            order = []     # keep repeat order for correlation matrix output

            ref_curve = None
            for smp, e in entries:
                L = np.asarray(e.get("loadings", None))
                if L is None or L.ndim != 2 or L.shape[1] <= pc:
                    continue
                wave = np.asarray(e["wave"], float)
                y = L[:, pc].astype(float)

                # normalise
                y = _norm_curve(y)
                # smooth (optional)
                if smooth:
                    w, p = _sg_params(len(y), sg_window, sg_poly)
                    if w is not None:
                        try:
                            y = savgol_filter(y, w, p, mode="interp")
                        except Exception:
                            pass

                # interpolate to ref_wave if needed
                if wave.shape != ref_wave.shape or not np.allclose(wave, ref_wave, rtol=0, atol=1e-9):
                    y = np.interp(ref_wave, wave, y)

                # sign-align to first valid curve
                if ref_curve is None:
                    ref_curve = y.copy()
                else:
                    y = _sign_align(ref_curve, y)

                aligned[smp] = y
                order.append(smp)

                # ---- peak picking on normalised curve (both polarities) ----
                # positive peaks
                idx_pos, props_pos = find_peaks(y, prominence=peak_prom, distance=peak_distance)
                # negative peaks (flip the signal)
                idx_neg, props_neg = find_peaks(-y, prominence=peak_prom, distance=peak_distance)

                for idx in idx_pos:
                    peaks_rows.append({
                        "Subtype": st, "PC": f"PC{pc+1}", "Repeat": smp,
                        "Peak_Pos_cm-1": float(ref_wave[idx]),
                        "Peak_Height": float(y[idx]),
                        "Sign": "pos"
                    })
                for idx in idx_neg:
                    peaks_rows.append({
                        "Subtype": st, "PC": f"PC{pc+1}", "Repeat": smp,
                        "Peak_Pos_cm-1": float(ref_wave[idx]),
                        "Peak_Height": float(y[idx]),
                        "Sign": "neg"
                    })

            # ---- correlations (pairwise between repeats) ----
            for i in range(len(order)):
                for j in range(i+1, len(order)):
                    a, b = aligned[order[i]], aligned[order[j]]
                    if a.size and b.size:
                        r = np.corrcoef(a, b)[0,1]
                        corr_rows.append({
                            "Subtype": st, "PC": f"PC{pc+1}",
                            "Repeat_i": order[i], "Repeat_j": order[j],
                            "Pearson_r": float(r)
                        })

            # ---- consensus via clustering peaks within ±peak_tol cm-1 ----
            # gather all peaks for this subtype+pc
            st_pc_peaks = [r for r in peaks_rows if r["Subtype"]==st and r["PC"]==f"PC{pc+1}"]
            if st_pc_peaks:
                # sort by position
                pts = sorted([(p["Peak_Pos_cm-1"], p["Repeat"], p["Sign"]) for p in st_pc_peaks], key=lambda x: x[0])
                # greedy clustering by tolerance
                clusters = []
                cur = [pts[0]]
                for pos, rep, sgn in pts[1:]:
                    if abs(pos - cur[-1][0]) <= peak_tol:
                        cur.append((pos, rep, sgn))
                    else:
                        clusters.append(cur); cur = [(pos, rep, sgn)]
                clusters.append(cur)

                # summarise clusters
                repeats_total = len(set([r for r in order])) if order else 0
                for cl in clusters:
                    center = float(np.mean([x[0] for x in cl]))
                    reps   = set([x[1] for x in cl])
                    count  = len(reps)
                    frac   = (count / repeats_total) if repeats_total else np.nan
                    pos_count = sum(1 for x in cl if x[2] == "pos")
                    neg_count = sum(1 for x in cl if x[2] == "neg")
                    freq_rows.append({
                        "Subtype": st, "PC": f"PC{pc+1}",
                        "Consensus_Pos_cm-1": center,
                        "Repeats_with_peak": count,
                        "Total_repeats": repeats_total,
                        "Fraction": frac,
                        "Pos_votes": pos_count,
                        "Neg_votes": neg_count
                    })

    # ---------- to DataFrames ----------
    df_peaks = pd.DataFrame(peaks_rows, columns=[
        "Subtype","PC","Repeat","Peak_Pos_cm-1","Peak_Height","Sign"
    ])
    df_corr  = pd.DataFrame(corr_rows, columns=[
        "Subtype","PC","Repeat_i","Repeat_j","Pearson_r"
    ])
    df_freq  = pd.DataFrame(freq_rows, columns=[
        "Subtype","PC","Consensus_Pos_cm-1","Repeats_with_peak","Total_repeats","Fraction","Pos_votes","Neg_votes"
    ])

    # ---------- write workbook ----------
    outdir = Path(Save_folder) / "Outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    xlsx_path = outdir / f"PCA_Loadings_Quant_{region}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
        df_peaks.to_excel(xw, index=False, sheet_name="Peaks")
        df_corr.to_excel(xw,  index=False, sheet_name="Correlations")
        df_freq.to_excel(xw,  index=False, sheet_name="PeakFrequency")
    print(f"[PCA_poolloadings_quant] Wrote: {xlsx_path}")

    # ---------- quick console highlights ----------
    if not df_freq.empty:
        print("\nTop consensus peaks per subtype (by Fraction):")
        for st in sorted(df_freq["Subtype"].unique()):
            sub = df_freq[df_freq["Subtype"]==st].sort_values(["PC","Fraction"], ascending=[True, False])
            top = sub.groupby("PC").head(3)  # top 3 per PC
            for _, r in top.iterrows():
                print(f"  {st} {r['PC']}: {r['Consensus_Pos_cm-1']:.1f} cm^-1  "
                      f"({int(r['Repeats_with_peak'])}/{int(r['Total_repeats'])} repeats)")
    else:
        print("[PCA_poolloadings_quant] No peaks detected with current parameters.")

    return data_dict

def PCApersubtype(data_dict, Save_folder,removeoutliers,  plotall_persubtype, PCAorder, meancentre, n_components, region="FP"):
    """
    Step 10: Pool PCA across repeats within each subtype to obtain a common PC basis.

    Selection:
      - PCAorder == 'Trim'  -> use {FP,EXT}_Spectra_Treated_Dermis
      - else                -> use {FP,EXT}_Spectra_Treated_Full

    Region:
      - 'FP' | 'EXT' | 'both'   (runs independently per region)

    Centering:
      - meancentre='with'     -> sklearn PCA (centers data)
      - meancentre='without'  -> TruncatedSVD (does NOT center)

    Plots:
      - plot_mode='pdf'    -> one combined PDF per region (pages per subtype)
      - plot_mode='screen' -> show plots interactively
      - plot_mode='None'   -> no plotting

    Results stored at:
      data_dict["_PCA_per_subtype"][<region>][<subtype>] = {
        "wave", "loadings", "expl",
        "scores_all", "scores_mean_by_sample",
        "centre", "order", "n_samples", "n_spectra"
      }
    """

    # --- helpers already in your module ---
    # _region_keys(use_dermis, region)   -> yields (label, dict_key)
    # _stack_sample_matrix(spectra_dict) -> (X, wave)

    store_key = "_PCA_per_subtype"
    data_dict.setdefault(store_key, {}).setdefault("FP", {})
    data_dict[store_key].setdefault("EXT", {})

    use_dermis = (str(PCAorder).strip().lower() == "trim")
    centre = str(meancentre).strip().lower()
    if centre not in ("with", "without"):
        print(f"[PCApersubtype] Warning: meancentre='{meancentre}' not recognised; using 'with'")
        centre = "with"

    regions_to_do = ("FP", "EXT") if str(region).lower() == "both" else (region.upper(),)
    title_suffix = "Dermis" if use_dermis else "Whole line"

    # Prepare PDFs if requested (one per region)
    pdf_writers = {}
    pdf_paths   = {}   # <-- keep the paths here
    if plotall_persubtype == "pdf":
        outdir = Path(Save_folder) / "Outputs"
        outdir.mkdir(parents=True, exist_ok=True)
        for reg in regions_to_do:
            path = outdir / f"PCA_per_subtype_{reg}.pdf"
            pdf_writers[reg] = PdfPages(path)
            pdf_paths[reg]   = str(path)

    # --- group samples by subtype once ---
    subtype_to_samples = defaultdict(list)
    for sample_num, sdict in data_dict.items():
        if not isinstance(sdict, dict):
            continue
        if _is_excluded(sample_num, removeoutliers):  # <-- NEW
            continue
        st = sdict.get("Subtype", None)
        if st is None:
            continue
        subtype_to_samples[st].append(str(sample_num))


    # --- per region workflow ---
    for reg in regions_to_do:
        rkey_name, rkey_full = next(_region_keys(use_dermis, reg))  # ('FP', 'FP_Spectra_Treated_*')
        # build pooled matrices per subtype
        for subtype, sample_list in subtype_to_samples.items():

            rows = []
            sample_of_row = []
            waves_ref = None

            # collect spectra across all repeats within this subtype
            for sample_num in sample_list:
                if _is_excluded(sample_num, removeoutliers):  # <-- NEW (extra safety)
                    continue
                sdict = data_dict.get(sample_num, {})
                spectra_dict = sdict.get(rkey_full, None)

                if not spectra_dict:
                    continue

                X, wave = _stack_sample_matrix(spectra_dict)
                if X is None or X.shape[0] < 2:
                    continue

                # align to a single reference wave
                if waves_ref is None:
                    waves_ref = wave
                else:
                    if wave.shape != waves_ref.shape or not np.allclose(wave, waves_ref, rtol=0, atol=1e-9):
                        X = np.vstack([np.interp(waves_ref, wave, xi) for xi in X])

                rows.append(X)
                sample_of_row.extend([sample_num] * X.shape[0])

            if not rows:
                # nothing for this subtype/region
                continue

            X_all = np.vstack(rows)  # (total_spectra, n_waves)

            # choose estimator
            ncomp_eff = int(min(n_components, X_all.shape[0], X_all.shape[1]))
            if centre == "with":
                est = _SKPCA(n_components=ncomp_eff, svd_solver="auto", random_state=0)
                scores_all = est.fit_transform(X_all)
                loadings   = est.components_.T
                expl       = est.explained_variance_ratio_
            else:
                est = _SVD(n_components=ncomp_eff, random_state=0)
                scores_all = est.fit_transform(X_all)
                loadings   = est.components_.T
                svals      = est.singular_values_
                expl       = (svals**2) / np.sum((X_all**2))

            # per-sample means in PC space (one point per repeat)
            scores_by_sample = defaultdict(list)
            for i, sname in enumerate(sample_of_row):
                scores_by_sample[sname].append(scores_all[i])
            scores_mean_by_sample = {s: np.mean(np.vstack(v), axis=0) for s, v in scores_by_sample.items()}

            # --- store results ---
            data_dict[store_key][reg][subtype] = {
                "wave":       waves_ref,
                "loadings":   loadings,
                "expl":       expl,
                "scores_all": scores_all,
                "scores_mean_by_sample": scores_mean_by_sample,
                "centre":     centre,
                "order":      ("Dermis" if use_dermis else "Whole"),
                "n_samples":  len(scores_by_sample),
                "n_spectra":  X_all.shape[0],
            }

            # --- plotting ---
            if plotall_persubtype != "None":
                # 1) Scree
                fig = plt.figure(figsize=(7.5, 4)); ax = fig.add_subplot(111)
                idx = np.arange(1, len(expl) + 1)
                ax.bar(idx, expl, width=0.6, color="tab:gray", alpha=0.6, label="Explained variance")
                ax.plot(idx, np.cumsum(expl), "--o", color="black", label="Cumulative")
                ax.set_xlabel("Principal Component"); ax.set_ylabel("Variance explained")
                ax.set_title(f"{reg} — Subtype {subtype} — Scree ({title_suffix}, centre='{centre}')")
                ax.legend(); fig.tight_layout()

                if plotall_persubtype == "screen":
                    plt.show(); plt.close(fig)
                else:
                    pdf_writers[reg].savefig(fig); plt.close(fig)

                # 2) Scores (per-sample means, PC1 vs PC2)
                fig = plt.figure(figsize=(6.5, 6)); ax = fig.add_subplot(111)
                for sname, mu in scores_mean_by_sample.items():
                    if mu.shape[0] < 2: continue
                    ax.scatter(mu[0], mu[1], s=40, label=f"repeat {sname}")
                ax.set_xlabel("PC1 score (mean per repeat)")
                ax.set_ylabel("PC2 score (mean per repeat)")
                ax.set_title(f"{reg} — Subtype {subtype} — Scores (repeat means)")
                ax.grid(True, linestyle=":", alpha=0.3); ax.legend(fontsize=8)
                fig.tight_layout()

                if plotall_persubtype == "screen":
                    plt.show(); plt.close(fig)
                else:
                    pdf_writers[reg].savefig(fig); plt.close(fig)

                # 3) Loadings (PC1 & PC2)
                fig = plt.figure(figsize=(9, 4.5)); ax = fig.add_subplot(111)
                ax.plot(waves_ref, loadings[:, 0], label="PC1", linewidth=1.7)
                if loadings.shape[1] >= 2:
                    ax.plot(waves_ref, loadings[:, 1], "--", label="PC2", linewidth=1.2)
                ax.set_xlabel("Wavenumber (cm⁻¹)")
                ax.set_ylabel("Loading")
                ax.set_title(f"{reg} — Subtype {subtype} — Loadings")
                ax.legend(); fig.tight_layout()

                if plotall_persubtype == "screen":
                    plt.show(); plt.close(fig)
                else:
                    pdf_writers[reg].savefig(fig); plt.close(fig)

    # close PDFs
    if plotall_persubtype == "pdf":
        for reg, writer in pdf_writers.items():
            writer.close()
            print(f"[PCApersubtype] Saved combined PDF for {reg} to {pdf_paths.get(reg, '(unknown path)')}")
    return data_dict

def singlePCA(data_dict, Save_folder, removeoutliers, PCAorder, meancentre, n_components, plotall_singlePCA, region="FP", save_xlsx=True):

    """
    Step 11: One PCA across ALL samples & subtypes to obtain a single PC basis
    for the whole experiment (per region). Plots scores grouped by subtype/repeat.

    Selection:
      - PCAorder == 'Trim'  -> use {FP,EXT}_Spectra_Treated_Dermis
      - else                -> use {FP,EXT}_Spectra_Treated_Full

    Region:
      - 'FP' | 'EXT' | 'both' (runs per region independently)

    Centering:
      - meancentre='with'     -> sklearn PCA (centers)
      - meancentre='without'  -> TruncatedSVD (does NOT center)

    Plots per region (if plot_mode ≠ 'None'):
      1) Scree (EVR + cumulative)
      2) Scores: PC1 vs PC2 (per-sample mean point), colored by subtype
      3) Subtype means ± SEM in PC1–PC2 space
      4) Loadings: PC1 & PC2 vs wavenumber

    Results stored:
      data_dict["_PCA_all"][region] = {
        "wave", "loadings", "expl",
        "scores_all"           : (N_spectra, n_comp),
        "scores_sample_mean"   : {sample_id: vector},
        "scores_subtype_stats" : {subtype: {"mean": v, "sem": v}},
        "labels"               : {
            "sample_of_row": [sample_id] * N_spectra,
            "subtype_of_row": [subtype]  * N_spectra
        },
        "centre", "order", "n_samples", "n_spectra"
      }

    Excel export per region (if save_xlsx=True):
      - Scores_all
      - Scores_sample_means
      - Subtype_means_SEM
      - Loadings_PC1_PC2
      - ExplainedVariance
    """


    store_key = "_PCA_all"
    data_dict.setdefault(store_key, {}).setdefault("FP", {})
    data_dict[store_key].setdefault("EXT", {})

    use_dermis = (str(PCAorder).strip().lower() == "trim")
    centre = str(meancentre).strip().lower()
    if centre not in ("with", "without"):
        print(f"[singlePCA] Warning: meancentre='{meancentre}' not recognised; using 'with'")
        centre = "with"

    regions_to_do = ("FP", "EXT") if str(region).lower() == "both" else (region.upper(),)
    title_suffix = "Dermis" if use_dermis else "Whole line"

    # Prepare PDFs if requested (one per region)
    pdf_writers = {}
    pdf_paths   = {}   # <-- keep track of paths
    if plotall_singlePCA == "pdf":
        outdir = Path(Save_folder) / "Outputs"
        outdir.mkdir(parents=True, exist_ok=True)
        for reg in regions_to_do:
            path = outdir / f"PCA_all_{reg}.pdf"
            pdf_writers[reg] = PdfPages(path)
            pdf_paths[reg]   = str(path)
    # --- iterate regions ---
    for reg in regions_to_do:
        # gather all spectra across all samples for this region
        rows = []
        waves_ref = None
        sample_of_row = []
        subtype_of_row = []

        # (label, dict_key)
        rlabel, rkey = next(_region_keys(use_dermis, reg))

        # collect
        for sample_num, sdict in data_dict.items():
            if not isinstance(sdict, dict):
                continue
            if _is_excluded(sample_num, removeoutliers):   # <-- NEW
                continue
            st = sdict.get("Subtype", None)
            if st is None:
                continue

            spectra_dict = sdict.get(rkey, None)
            if not spectra_dict:
                continue

            X, wave = _stack_sample_matrix(spectra_dict)
            if X is None or X.shape[0] < 2:
                continue

            # fix a common wave axis
            if waves_ref is None:
                waves_ref = wave
            else:
                if wave.shape != waves_ref.shape or not np.allclose(wave, waves_ref, rtol=0, atol=1e-9):
                    X = np.vstack([np.interp(waves_ref, wave, xi) for xi in X])

            rows.append(X)
            sample_of_row.extend([str(sample_num)] * X.shape[0])
            subtype_of_row.extend([st] * X.shape[0])

        if not rows:
            print(f"[singlePCA] No data for region={reg}. Skipping.")
            continue

        X_all = np.vstack(rows)  # (N_spectra, n_waves)

        # estimator
        ncomp_eff = int(min(n_components, X_all.shape[0], X_all.shape[1]))
        if centre == "with":
            est = _SKPCA(n_components=ncomp_eff, svd_solver="auto", random_state=0)
            scores_all = est.fit_transform(X_all)
            loadings   = est.components_.T
            expl       = est.explained_variance_ratio_
        else:
            est = _SVD(n_components=ncomp_eff, random_state=0)
            scores_all = est.fit_transform(X_all)
            loadings   = est.components_.T
            svals      = est.singular_values_
            expl       = (svals**2) / np.sum((X_all**2))

        # per-sample means (one point per repeat)
        scores_by_sample = defaultdict(list)
        for i, sname in enumerate(sample_of_row):
            scores_by_sample[sname].append(scores_all[i])
        scores_sample_mean = {s: np.mean(np.vstack(v), axis=0) for s, v in scores_by_sample.items()}

        # per-subtype mean ± SEM (over per-sample means)
        subtype_map = {}
        for smp, vec in scores_sample_mean.items():
            # fetch subtype for this sample: scan one row to find subtype
            # Build a quick lookup once:
            pass
        # build sample->subtype lookup from any row label (first occurrence)
        sample_to_subtype = {}
        for i, smp in enumerate(sample_of_row):
            if smp not in sample_to_subtype:
                sample_to_subtype[smp] = subtype_of_row[i]

        subtype_agg = defaultdict(list)
        for smp, vec in scores_sample_mean.items():
            subtype_agg[sample_to_subtype.get(smp, "Unknown")].append(vec)

        scores_subtype_stats = {}
        for st, arr in subtype_agg.items():
            A = np.vstack(arr)
            scores_subtype_stats[st] = dict(mean=A.mean(axis=0),
                                            sem=A.std(axis=0, ddof=1)/np.sqrt(A.shape[0]))

        # ---- store results ----
        data_dict[store_key][reg] = {
            "wave": waves_ref,
            "loadings": loadings,
            "expl": expl,
            "scores_all": scores_all,
            "scores_sample_mean": scores_sample_mean,
            "scores_subtype_stats": scores_subtype_stats,
            "labels": {
                "sample_of_row": sample_of_row,
                "subtype_of_row": subtype_of_row
            },
            "centre": centre,
            "order": ("Dermis" if use_dermis else "Whole"),
            "n_samples": len(scores_sample_mean),
            "n_spectra": X_all.shape[0],
        }

        # ---- plotting ----
        if plotall_singlePCA != "None":
            # 1) Scree
            fig = plt.figure(figsize=(8, 4)); ax = fig.add_subplot(111)
            idx = np.arange(1, len(expl) + 1)
            ax.bar(idx, expl, width=0.6, color="tab:gray", alpha=0.6, label="Explained variance")
            ax.plot(idx, np.cumsum(expl), "--o", color="black", label="Cumulative")
            ax.set_xlabel("Principal Component")
            ax.set_ylabel("Variance explained")
            ax.set_title(f"{reg} — Scree (all samples; {title_suffix}, centre='{centre}')")
            ax.legend(); fig.tight_layout()

            if plotall_singlePCA == "screen":
                plt.show(); plt.close(fig)
            else:
                pdf_writers[reg].savefig(fig); plt.close(fig)

            # 2) Scores: per-sample means colored by subtype
            fig = plt.figure(figsize=(7.5, 6)); ax = fig.add_subplot(111)
            # inside plotting: Scores and Subtype means
            for smp, vec in scores_sample_mean.items():
                st = sample_to_subtype.get(smp, "Unknown")
                # pull a colour/linestyle from any sample belonging to this subtype
                st_col, st_ls = None, None
                for sname, sdict in data_dict.items():
                    if isinstance(sdict, dict) and sdict.get("Subtype") == st:
                        st_col = sdict.get("Colour", None)
                        st_ls  = sdict.get("Linestyle", None)
                        break
                if vec.shape[0] >= 2:
                    ax.scatter(vec[0], vec[1], s=45, color=st_col, label=st)

            # avoid repeated legend entries
            handles, labels = ax.get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            ax.legend(uniq.values(), uniq.keys(), title="Subtype", fontsize=9)
            ax.set_xlabel("PC1 score (sample mean)")
            ax.set_ylabel("PC2 score (sample mean)")
            ax.set_title(f"{reg} — Sample mean scores (one point per repeat)")
            ax.grid(True, linestyle=":", alpha=0.3)
            fig.tight_layout()

            if plotall_singlePCA == "screen":
                plt.show(); plt.close(fig)
            else:
                pdf_writers[reg].savefig(fig); plt.close(fig)

            # 3) Subtype means ± SEM
            fig = plt.figure(figsize=(7.5, 6)); ax = fig.add_subplot(111)
            for st, stats in scores_subtype_stats.items():
                # get stored colour for this subtype
                st_col = None
                for sname, sdict in data_dict.items():
                    if isinstance(sdict, dict) and sdict.get("Subtype") == st:
                        st_col = sdict.get("Colour", None)
                        break
                mu, se = stats["mean"], stats["sem"]
                if mu.shape[0] >= 2:
                    ax.errorbar(mu[0], mu[1], xerr=se[0], yerr=se[1],
                                fmt="o", color=st_col, ecolor=st_col,
                                elinewidth=1.2, capsize=3,
                                label=f"{st} mean±SEM")

            ax.set_xlabel("PC1 score (subtype mean)")
            ax.set_ylabel("PC2 score (subtype mean)")
            ax.set_title(f"{reg} — Subtype centroids (mean ± SEM)")
            ax.legend(fontsize=9); ax.grid(True, linestyle=":", alpha=0.3)
            fig.tight_layout()

            if plotall_singlePCA == "screen":
                plt.show(); plt.close(fig)
            else:
                pdf_writers[reg].savefig(fig); plt.close(fig)

            # 4) Loadings
            fig = plt.figure(figsize=(9, 4.5)); ax = fig.add_subplot(111)
            ax.plot(waves_ref, loadings[:, 0], label="PC1", linewidth=1.7)
            if loadings.shape[1] >= 2:
                ax.plot(waves_ref, loadings[:, 1], "--", label="PC2", linewidth=1.2)
            ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Loading")
            ax.set_title(f"{reg} — Loadings (PC1 & PC2)")
            ax.legend(); fig.tight_layout()

            if plotall_singlePCA == "screen":
                plt.show(); plt.close(fig)
            else:
                pdf_writers[reg].savefig(fig); plt.close(fig)

        # ---- Excel export (optional) ----
        if save_xlsx:
            outdir = Path(Save_folder) / "Outputs"
            outdir.mkdir(parents=True, exist_ok=True)
            xlsx_path = outdir / f"PCA_all_{reg}.xlsx"

            # Scores_all
            df_scores_all = pd.DataFrame(scores_all[:, :min(5, scores_all.shape[1])],
                                         columns=[f"PC{i+1}" for i in range(min(5, scores_all.shape[1]))])
            df_scores_all["Sample"] = sample_of_row
            df_scores_all["Subtype"] = subtype_of_row

            # Sample means
            df_sample_means = pd.DataFrame([
                {"Sample": s, **{f"PC{i+1}": v[i] for i in range(len(v))}}
                for s, v in scores_sample_mean.items()
            ])

            # Subtype means ± SEM (PC1..k)
            rows = []
            for st, stats in scores_subtype_stats.items():
                mu = stats["mean"]; se = stats["sem"]
                row = {"Subtype": st}
                for i in range(len(mu)):
                    row[f"PC{i+1}_mean"] = mu[i]
                    row[f"PC{i+1}_sem"]  = se[i]
                rows.append(row)
            df_subtype_stats = pd.DataFrame(rows)

            # Loadings (PC1 & PC2)
            df_load = pd.DataFrame({
                "Wavenumber_cm-1": waves_ref,
                "PC1": loadings[:, 0]
            })
            if loadings.shape[1] >= 2:
                df_load["PC2"] = loadings[:, 1]

            # Explained variance
            df_evr = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(expl))],
                                   "ExplainedVarianceRatio": expl})

            with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
                df_scores_all.to_excel(xw, index=False, sheet_name="Scores_all")
                df_sample_means.to_excel(xw, index=False, sheet_name="Scores_sample_means")
                df_subtype_stats.to_excel(xw, index=False, sheet_name="Subtype_means_SEM")
                df_load.to_excel(xw, index=False, sheet_name="Loadings_PC1_PC2")
                df_evr.to_excel(xw, index=False, sheet_name="ExplainedVariance")
            print(f"[singlePCA] Saved Excel for {reg} to {xlsx_path}")

    # close PDFs
    if plotall_singlePCA == "pdf":
        for reg, writer in pdf_writers.items():
            writer.close()
            print(f"[singlePCA] Saved combined PDF for {reg} to {pdf_paths.get(reg,'(unknown path)')}")

    return data_dict

def BinnedPCA_whole(data_dict,Save_folder,PCAorder,meancentre,n_components,n_bins,plot_mode_BinPCA,region="FP"):
    """
    Bin each sample's line-scan into n_bins contiguous segments and run PCA per bin.

    Selection:
      - PCAorder == 'Trim'  -> use {FP,EXT}_Spectra_Treated_Dermis
      - else                -> use {FP,EXT}_Spectra_Treated_Full

    Region:
      - 'FP' | 'EXT' | 'both' (runs independently per region)

    Centering:
      - meancentre='with'     -> sklearn PCA (centers)
      - meancentre='without'  -> TruncatedSVD (does NOT center)

    Plots per sample & region (if plot_mode != 'None'):
      1) Scree overlay: EVR curves for all bins (lines/markers)
      2) Scores overlay: PC1 vs PC2, points coloured by bin
      3) Loadings overlay: PC1 (solid) & PC2 (dashed), colour per bin

    Results stored at:
      data_dict["_PCA_binned"][region][sample_num] = {
          "meta": {"centre": 'with'|'without', "order": "Dermis"|"Whole", "n_bins": int},
          "bins": {
              0: {"wave", "scores", "loadings", "expl", "idx_range": (start,end)},
              1: {...},
              ...
          }
      }
    """

    # --- Helpers you already have in your module ---
    # _region_keys(use_dermis, region)   -> yields (label, dict_key)
    # _stack_sample_matrix(spectra_dict) -> (X, wave)   (not used directly here; we need sub-ranges)

    def _stack_subset(spectra_dict, key_list):
        """Stack only the spectra in key_list, aligning to the first spectrum's wave."""
        if not key_list:
            return None, None
        ref_wav = np.asarray(spectra_dict[key_list[0]].spectral_axis, dtype=float)
        rows = []
        for k in key_list:
            sc = spectra_dict[k]
            wav = np.asarray(sc.spectral_axis, dtype=float)
            inten = np.asarray(sc.spectral_data, dtype=float)
            if wav.shape != ref_wav.shape or not np.allclose(wav, ref_wav, rtol=0, atol=1e-9):
                inten = np.interp(ref_wav, wav, inten)
            rows.append(inten)
        return np.vstack(rows), ref_wav

    def _choose_estimator(X, centre, n_components):
        n_eff = int(min(n_components, X.shape[0], X.shape[1]))
        if centre == "with":
            est = _SKPCA(n_components=n_eff, svd_solver="auto", random_state=0)
            scores = est.fit_transform(X)
            load = est.components_.T
            evr = est.explained_variance_ratio_
        else:
            est = _SVD(n_components=n_eff, random_state=0)
            scores = est.fit_transform(X)
            load = est.components_.T
            svals = est.singular_values_
            evr = (svals**2) / np.sum((X**2))
        return scores, load, evr

    def _bin_indices(n_points, n_bins):
        """Return a list of index arrays (contiguous) that partition 0..n_points-1."""
        n_eff = max(1, min(int(n_bins), int(n_points)))
        return list(np.array_split(np.arange(n_points), n_eff))

    # ------------------------------------------------------------------
    # config
    # ------------------------------------------------------------------
    use_dermis = (str(PCAorder).strip().lower() == "trim")
    centre = str(meancentre).strip().lower()
    if centre not in ("with", "without"):
        print(f"[BinnedPCA_whole] Warning: meancentre='{meancentre}' not recognised; using 'with'")
        centre = "with"
    regions_to_do = ("FP", "EXT") if str(region).lower() == "both" else (region.upper(),)
    title_suffix = "Dermis" if use_dermis else "Whole line"

    # storage root
    store_key = "_PCA_binned"
    data_dict.setdefault(store_key, {}).setdefault("FP", {})
    data_dict[store_key].setdefault("EXT", {})

    # output dir for PDFs
    if plot_mode_BinPCA == "pdf":
        outdir = Path(Save_folder) / "Outputs"
        outdir.mkdir(parents=True, exist_ok=True)

    # colour cycle for bins (per sample)
    
    bin_colors_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]))

    # ------------------------------------------------------------------
    # per sample, per region
    # ------------------------------------------------------------------
    for sample_num, sdict in data_dict.items():
        if not isinstance(sdict, dict):
            continue

        for reg, rkey in _region_keys(use_dermis, region):
            if reg not in regions_to_do:
                continue

            spectra_dict = sdict.get(rkey, None)
            if not spectra_dict:
                continue

            # maintain scan order
            keys_order = list(spectra_dict.keys())
            n_points = len(keys_order)
            if n_points < 2:
                continue

            idx_bins = _bin_indices(n_points, n_bins)

            # --- compute PCA per bin ---
            # prepare storage
            data_dict[store_key][reg].setdefault(str(sample_num), {})
            sample_bin_store = {
                "meta": {"centre": centre, "order": ("Dermis" if use_dermis else "Whole"), "n_bins": len(idx_bins)},
                "bins": {}
            }

            # Collect plotting payloads
            scree_evrs = []      # list of (bin_id, evr)
            scores_list = []     # list of (bin_id, scores)
            loads_list = []      # list of (bin_id, wave, loadings)

            for b_id, idxs in enumerate(idx_bins):
                idxs = np.asarray(idxs, dtype=int)
                if idxs.size < 2:
                    # need at least 2 spectra to get meaningful PC1/PC2 scores
                    continue

                key_subset = [keys_order[i] for i in idxs]
                Xb, wave = _stack_subset(spectra_dict, key_subset)
                if Xb is None or Xb.shape[0] < 2:
                    continue

                scores, load, evr = _choose_estimator(Xb, centre, n_components)

                # store
                sample_bin_store["bins"][b_id] = {
                    "wave": wave,
                    "scores": scores,
                    "loadings": load,
                    "expl": evr,
                    "idx_range": (int(idxs[0]), int(idxs[-1])),
                }
                scree_evrs.append((b_id, evr))
                scores_list.append((b_id, scores))
                loads_list.append((b_id, wave, load))

            # save to data_dict
            data_dict[store_key][reg][str(sample_num)] = sample_bin_store

            # nothing to plot if we have no valid bins
            if not scree_evrs or plot_mode_BinPCA == "None":
                continue

            # --- plotting: one PDF per sample/region or on-screen ---
            if plot_mode_BinPCA == "pdf":
                pdf_path = Path(Save_folder) / "Outputs" / f"{sample_num}_BinnedPCA_{reg}.pdf"
                pdf = PdfPages(pdf_path)
            else:
                pdf = None

            try:
                # 1) Scree overlay
                fig = plt.figure(figsize=(8, 4)); ax = fig.add_subplot(111)
                color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key().get("color", []))
                for b_id, evr in scree_evrs:
                    c = next(color_cycle)
                    idx = np.arange(1, len(evr) + 1)
                    # ax.plot(idx, evr, "-o", label=f"Bin {b_id+1}", color=c, alpha=0.9)
                    ax.bar(idx, evr,  label=f"Bin {b_id+1}", color=c, alpha=0.9)
                    ax.plot(idx, np.cumsum(evr), "--", color=c, alpha=0.5)
                ax.set_xlabel("Principal Component")
                ax.set_ylabel("Variance explained")
                ax.set_title(f"{reg} — {sample_num} — Scree (bins overlaid; {title_suffix}, centre='{centre}')")
                ax.legend(ncol=2, fontsize=8); fig.tight_layout()
                if pdf: pdf.savefig(fig); plt.close(fig)
                else: plt.show(); plt.close(fig)

                # 2) Scores overlay (PC1 vs PC2)
                fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111)
                color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key().get("color", []))
                for b_id, sc in scores_list:
                    if sc.shape[1] < 2:  # need PC1, PC2
                        continue
                    c = next(color_cycle)
                    ax.plot(sc[:, 0], sc[:, 1], linestyle="-", alpha=0.25, color=c)
                    ax.scatter(sc[:, 0], sc[:, 1], s=20, alpha=0.9, color=c, label=f"Bin {b_id+1}")
                ax.set_xlabel("PC1 score"); ax.set_ylabel("PC2 score")
                ax.set_title(f"{reg} — {sample_num} — Scores (bins overlaid)")
                ax.grid(True, linestyle=":", alpha=0.3); ax.legend(ncol=2, fontsize=8)
                fig.tight_layout()
                if pdf: pdf.savefig(fig); plt.close(fig)
                else: plt.show(); plt.close(fig)

                # 3) Loadings overlay (PC1 & PC2)
                fig = plt.figure(figsize=(9, 4.8)); ax = fig.add_subplot(111)
                color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key().get("color", []))
                for b_id, wave, load in loads_list:
                    c = next(color_cycle)
                    ax.plot(wave, load[:, 0], color=c, linewidth=1.6, alpha=0.95, label=f"Bin {b_id+1} PC1")
                    if load.shape[1] >= 2:
                        ax.plot(wave, load[:, 1], "--", color=c, linewidth=1.2, alpha=0.9, label=f"Bin {b_id+1} PC2")
                ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Loading")
                ax.set_title(f"{reg} — {sample_num} — Loadings (bins overlaid)")
                ax.legend(ncol=2, fontsize=8)
                ax.set_ylim(-0.2, 0.25)
                fig.tight_layout()
                if pdf: pdf.savefig(fig); plt.close(fig)
                else: plt.show(); plt.close(fig)

            finally:
                if pdf:
                    pdf.close()
                    # print path so users can find it easily
                    print(f"[BinnedPCA_whole] Saved: {pdf_path}")

    return data_dict

def BinnedPCA_average(data_dict,Save_folder,PCAorder,meancentre,n_components,n_bins,plot_mode_BinPCA_av, removeoutliers, region="FP", pos_grid_n=101, pos_window=0.5):
    """
    Bin-wise PCA on a *subtype-averaged* line (keeps spatial structure).

    Pipeline per REGION and per SUBTYPE:
      1) Gather all repeats (samples) for that subtype (optionally excluding 'removeoutliers').
      2) For each repeat: use scan order from trimmed dict; map index 0..(N-1) -> 0..100.
      3) Align all spectra to a *common wave axis* (per region).
      4) Build a uniform position grid (0..100, 'pos_grid_n' points).
         For each grid position, average spectra from repeats that have sample
         positions within ±pos_window around that grid position.
      5) Split the averaged stack (pos_grid_n x n_waves) into 'n_bins' contiguous bins.
         Run PCA per bin (meancentre='with' -> sklearn PCA; 'without' -> TruncatedSVD).
      6) Store & plot (overlay Scree/ Scores / Loadings for all bins).

    Stores results at:
      data_dict["_PCA_binned_avg"][region][subtype] = {
          "meta": {"centre", "order", "n_bins", "pos_grid"},
          "bins": {
              0: {"wave","scores","loadings","expl","pos_idx_range":(i0,i1)},
              1: {...},
              ...
          }
      }
    """

    # Helpers you already have:
    # - _region_keys(use_dermis, region) -> yields (label, dict_key)
    # - spectra dict keys are (x,y) tuples; dict insertion order preserves scan order after TrimRegion

    def _sample_ok(sname, remove):
        if remove is None:
            return True
        s = str(sname)
        return (s not in set(map(str, remove)))

    def _ensure_common_wave(ref_wave, wave, inten):
        """Interpolate intensity onto ref_wave if needed."""
        if ref_wave is None:
            return None, inten, wave
        if (wave.shape != ref_wave.shape) or (not np.allclose(wave, ref_wave, rtol=0, atol=1e-9)):
            inten = np.interp(ref_wave, wave, inten)
            wave = ref_wave
        return ref_wave, inten, wave

    def _stack_position_averaged(subtype_samples, rkey_full, pos_grid, half_window, ref_wave):
        """
        Build a (len(pos_grid) x n_waves) averaged stack across repeats for one subtype & region.
        Uses TrimRegion-preserved scan order -> normalised to 0..100, and local window averaging.
        """
        # Collect all (pos_norm, spectrum) pairs across repeats
        all_pos = []
        all_int = []

        for sample_num in subtype_samples:
            sdict = data_dict.get(sample_num, {})
            spectra_dict = sdict.get(rkey_full, None)
            if not spectra_dict:
                continue
            # maintain scan order:
            keys_order = list(spectra_dict.keys())
            n_points = len(keys_order)
            if n_points == 0:
                continue
            # normalised positions for this repeat
            if n_points == 1:
                pos_norm = np.array([50.0])  # degenerate, centre it
            else:
                pos_norm = np.linspace(0.0, 100.0, n_points)

            # loop through positions & add spectra (aligned to ref_wave)
            for i, k in enumerate(keys_order):
                sc = spectra_dict[k]
                wav = np.asarray(sc.spectral_axis, dtype=float)
                y   = np.asarray(sc.spectral_data, dtype=float)
                if ref_wave is None:
                    # first time, set ref_wave from this spectrum
                    ref_wave = wav.copy()
                else:
                    # interpolate onto ref_wave if needed
                    if (wav.shape != ref_wave.shape) or (not np.allclose(wav, ref_wave, rtol=0, atol=1e-9)):
                        y = np.interp(ref_wave, wav, y)
                all_pos.append(pos_norm[i])
                all_int.append(y)

        if ref_wave is None or len(all_int) == 0:
            return None, None  # nothing collected

        all_pos = np.asarray(all_pos, dtype=float)
        all_int = np.vstack(all_int)  # (#spectra_total, n_waves)

        # For each position on the grid, average spectra within +- half_window
        stack = []
        for p in pos_grid:
            mask = np.abs(all_pos - p) <= half_window
            if not np.any(mask):
                # fallback: use the nearest spectrum if window empty
                j = int(np.argmin(np.abs(all_pos - p)))
                stack.append(all_int[j])
            else:
                stack.append(all_int[mask].mean(axis=0))
        return np.vstack(stack), ref_wave  # (len(pos_grid), n_waves)

    def _choose_estimator(X, centre, n_components):
        n_eff = int(min(n_components, X.shape[0], X.shape[1]))
        if centre == "with":
            est = _SKPCA(n_components=n_eff, svd_solver="auto", random_state=0)
            scores = est.fit_transform(X)
            load = est.components_.T
            evr = est.explained_variance_ratio_
        else:
            est = _SVD(n_components=n_eff, random_state=0)
            scores = est.fit_transform(X)
            load = est.components_.T
            svals = est.singular_values_
            evr = (svals**2) / np.sum((X**2))
        return scores, load, evr

    def _bin_indices(n_points, n_bins):
        n_eff = max(1, min(int(n_bins), int(n_points)))
        return list(np.array_split(np.arange(n_points), n_eff))

    # -------------------- config --------------------
    use_dermis = (str(PCAorder).strip().lower() == "trim")
    centre = str(meancentre).strip().lower()
    if centre not in ("with", "without"):
        print(f"[BinnedPCA_average] Warning: meancentre='{meancentre}' not recognised; using 'with'")
        centre = "with"

    regions_to_do = ("FP", "EXT") if str(region).lower() == "both" else (region.upper(),)
    title_suffix = "Dermis" if use_dermis else "Whole line"
    pos_grid = np.linspace(0.0, 100.0, int(pos_grid_n))
    half_w = float(pos_window)

    # storage root
    store_key = "_PCA_binned_avg"
    data_dict.setdefault(store_key, {}).setdefault("FP", {})
    data_dict[store_key].setdefault("EXT", {})

    # output dir for PDFs
    if plot_mode_BinPCA_av == "pdf":
        outdir = Path(Save_folder) / "Outputs"
        outdir.mkdir(parents=True, exist_ok=True)

    # group samples by subtype once (exclude outliers now)
    subtype_to_samples = defaultdict(list)
    for sample_num, sdict in data_dict.items():
        if not isinstance(sdict, dict):
            continue
        st = sdict.get("Subtype", None)
        if st is None:
            continue
        if not _sample_ok(sample_num, removeoutliers):
            continue
        subtype_to_samples[st].append(str(sample_num))

    # -------------------- per region --------------------
    for reg in regions_to_do:
        # dict key for spectra in this selection
        _, rkey_full = next(_region_keys(use_dermis, reg))

        # One PDF per subtype (optional)
        pdf = None

        for subtype, sample_list in subtype_to_samples.items():
            # Build averaged stack across repeats at uniform position grid
            ref_wave = None
            Xpos, ref_wave = _stack_position_averaged(sample_list, rkey_full, pos_grid, half_w, ref_wave)
            if Xpos is None or Xpos.shape[0] < 2:
                # nothing to do
                continue

            # Split positions into bins
            pos_bins = _bin_indices(Xpos.shape[0], n_bins)

            # Prepare storage
            data_dict[store_key][reg].setdefault(subtype, {})
            subtype_store = {
                "meta": {"centre": centre, "order": ("Dermis" if use_dermis else "Whole"),
                         "n_bins": len(pos_bins), "pos_grid": pos_grid},
                "bins": {}
            }

            # Accumulators for plotting
            scree_evrs = []
            scores_list = []
            loads_list  = []

            for b_id, idxs in enumerate(pos_bins):
                idxs = np.asarray(idxs, dtype=int)
                if idxs.size < 2:
                    continue
                Xb = Xpos[idxs, :]  # (n_pos_in_bin, n_waves)

                scores, load, evr = _choose_estimator(Xb, centre, n_components)

                # store
                subtype_store["bins"][b_id] = {
                    "wave": ref_wave,
                    "scores": scores,
                    "loadings": load,
                    "expl": evr,
                    "pos_idx_range": (int(idxs[0]), int(idxs[-1])),
                }
                scree_evrs.append((b_id, evr))
                scores_list.append((b_id, scores))
                loads_list.append((b_id, ref_wave, load))

            data_dict[store_key][reg][subtype] = subtype_store

            # --- plotting (skip if requested) ---
            if plot_mode_BinPCA_av == "None" or not scree_evrs:
                continue

            # Open/rotate a PDF per subtype & region
            if plot_mode_BinPCA_av == "pdf":
                pdf_path = Path(Save_folder) / "Outputs" / f"{subtype}_BinnedPCA_avg_{reg}.pdf"
                pdf = PdfPages(pdf_path)

            try:
                # 1) Scree overlay
                fig = plt.figure(figsize=(8, 4)); ax = fig.add_subplot(111)
                colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
                for i, (b_id, evr) in enumerate(scree_evrs):
                    c = colors[i % len(colors)] if colors else None
                    idx = np.arange(1, len(evr) + 1)
                    # ax.plot(idx, evr, "-o", color=c, alpha=0.95, label=f"Bin {b_id+1}")
                    ax.bar(idx, evr,  color=c, alpha=0.95, label=f"Bin {b_id+1}")
                    ax.plot(idx, np.cumsum(evr), "--", color=c, alpha=0.5)
                ax.set_xlabel("Principal Component"); ax.set_ylabel("Variance explained")
                ax.set_title(f"{reg} — {subtype} — Scree (bins overlaid; {title_suffix}, centre='{centre}')")
                ax.legend(ncol=2, fontsize=8); fig.tight_layout()
                if plot_mode_BinPCA_av == "pdf":
                    pdf.savefig(fig); plt.close(fig)
                else:
                    plt.show(); plt.close(fig)

                # 2) Scores overlay
                fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111)
                for i, (b_id, sc) in enumerate(scores_list):
                    if sc.shape[1] < 2: 
                        continue
                    c = colors[i % len(colors)] if colors else None
                    ax.plot(sc[:, 0], sc[:, 1], "-", alpha=0.25, color=c)
                    ax.scatter(sc[:, 0], sc[:, 1], s=22, alpha=0.9, color=c, label=f"Bin {b_id+1}")
                ax.set_xlabel("PC1 score"); ax.set_ylabel("PC2 score")
                ax.set_title(f"{reg} — {subtype} — Scores (bins overlaid)")
                ax.grid(True, linestyle=":", alpha=0.3); ax.legend(ncol=2, fontsize=8)
                fig.tight_layout()
                if plot_mode_BinPCA_av == "pdf":
                    pdf.savefig(fig); plt.close(fig)
                else:
                    plt.show(); plt.close(fig)

                # 3) Loadings overlay
                fig = plt.figure(figsize=(9, 4.8)); ax = fig.add_subplot(111)
                for i, (b_id, wave, load) in enumerate(loads_list):
                    c = colors[i % len(colors)] if colors else None
                    ax.plot(wave, load[:, 0], color=c, linewidth=1.6, alpha=0.95, label=f"Bin {b_id+1} PC1")
                    if load.shape[1] >= 2:
                        ax.plot(wave, load[:, 1], "--", color=c, linewidth=1.2, alpha=0.9, label=f"Bin {b_id+1} PC2")
                ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Loading")
                ax.set_title(f"{reg} — {subtype} — Loadings (bins overlaid)")
                ax.legend(ncol=2, fontsize=8)
                ax.set_ylim(-0.2, 0.25)
                fig.tight_layout()
                if plot_mode_BinPCA_av == "pdf":
                    pdf.savefig(fig); plt.close(fig)
                else:
                    plt.show(); plt.close(fig)

            finally:
                if plot_mode_BinPCA_av == "pdf" and pdf is not None:
                    pdf.close()
                    print(f"[BinnedPCA_average] Saved: {pdf_path}")

    return data_dict

def PCA_MPCA(data_dict, Save_folder, PCAorder, meancentre, plot_mode_MPCA, n_components, removeoutliers, pos_grid, pos_components,sample_components=None,tensorly_backend="numpy", region="FP"):

    """
    MPCA (Tucker/HOSVD) run **per subtype**.
    Builds a tensor for each subtype: [repeats × position × wavenumber].

    Selection via PCAorder:
      - 'Trim'  -> {FP,EXT}_Spectra_Treated_Dermis
      - else    -> {FP,EXT}_Spectra_Treated_Full

    Centering:
      - meancentre='with'     -> subtract per-(pos,wave) mean across repeats
      - meancentre='without'  -> no centering

    Ranks:
      - spectral rank  = n_components
      - spatial rank   = pos_components
      - sample  rank   = sample_components (default min(3, n_repeats))

    Plotting:
      - 'pdf'    -> one PDF per subtype: Outputs/MPCA_{region}_{subtype}.pdf
      - 'screen' -> show interactively
      - 'None'   -> no plotting

    Results saved in:
      data_dict["_MPCA"][region][subtype] = {
         "wave": wave_axis,
         "pos":  pos_axis (0..100),
         "core": core,                  # ndarray [Rs, Rp, Rw]
         "factors": {
             "sample":  U_s,            # [n_repeats, Rs]
             "position":U_p,            # [pos_grid,  Rp]
             "spectral":U_w             # [n_waves,   Rw]
         },
         "ranks": (Rs, Rp, Rw),
         "centre": 'with'/'without',
         "order": 'Dermis'/'Whole',
         "samples": [list of repeat ids used],
      }
    """
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # tensorly (with correct arg name 'rank')
    try:
        import tensorly as tl
        from tensorly.decomposition import tucker
        tl.set_backend(tensorly_backend)
    except Exception as e:
        raise ImportError(
            "tensorly is required for MPCA. Install with `pip install tensorly`.\n"
            f"Import error: {e}"
        )

    use_dermis = (str(PCAorder).strip().lower() == "trim")
    centre     = str(meancentre).strip().lower()
    if centre not in ("with", "without"):
        print(f"[PCA_MPCA] Warning: meancentre='{meancentre}' not recognised; using 'with'")
        centre = "with"

    # which regions to do
    regions = ("FP", "EXT") if str(region).lower() == "both" else (region.upper(),)
    outdir  = Path(Save_folder) / "Outputs"
    if plot_mode_MPCA == "pdf":
        outdir.mkdir(parents=True, exist_ok=True)

    # Prepare storage
    store_key = "_MPCA"
    data_dict.setdefault(store_key, {})
    for reg in regions:
        data_dict[store_key].setdefault(reg, {})

    # helpers already in your module:
    #  - _region_keys(use_dermis, region) -> yields (label, dict_key)
    #  - _stack_sample_matrix(spectra_dict) -> (X, wave) with TrimRegion order

    def _resample_positions_to_grid(X_raw, pos_grid):
        """Map raw scan rows to %depth [0..100] and resample to `pos_grid` points."""
        npos = X_raw.shape[0]
        pos_raw = np.linspace(0.0, 100.0, npos)
        pos_ref = np.linspace(0.0, 100.0, int(pos_grid))
        X_res = np.empty((pos_ref.size, X_raw.shape[1]), dtype=float)
        for j in range(X_raw.shape[1]):
            X_res[:, j] = np.interp(pos_ref, pos_raw, X_raw[:, j])
        return pos_ref, X_res

    # ---- run per region ----
    for reg in regions:
        _, rkey = next(_region_keys(use_dermis, reg))
        excl = set(str(s) for s in (removeoutliers or []))

        # Build: subtype -> list of (sample_id, matrix[pos_grid × n_waves])
        by_subtype = {}
        wave_ref = None

        # Gather all repeats, grouped by subtype
        for sample_num, sdict in data_dict.items():
            if not isinstance(sdict, dict):
                continue
            sid = str(sample_num)
            if sid in excl:
                continue
            subtype = sdict.get("Subtype", None)
            if subtype is None:
                continue

            spectra_dict = sdict.get(rkey, None)
            if not spectra_dict:
                continue

            # Raw stack [n_positions_raw × n_waves], wave alignment to a common ref
            X_raw, wave_axis = _stack_sample_matrix(spectra_dict)
            if X_raw is None or X_raw.shape[0] < 3:
                continue

            if wave_ref is None:
                wave_ref = np.asarray(wave_axis, float)
            else:
                if (len(wave_axis) != len(wave_ref)) or (not np.allclose(wave_axis, wave_ref, rtol=0, atol=1e-9)):
                    X_raw = np.vstack([np.interp(wave_ref, wave_axis, row) for row in X_raw])

            # optional mean-centering along the wave dimension (per repeat)
            if centre == "with":
                X_raw = X_raw - X_raw.mean(axis=0, keepdims=True)

            # Resample along position to a fixed grid
            pos_ref, X_res = _resample_positions_to_grid(X_raw, pos_grid)

            by_subtype.setdefault(subtype, []).append((sid, X_res))

        if not by_subtype:
            print(f"[PCA_MPCA] No usable repeats for region={reg}. Skipping.")
            continue

        # ---- run Tucker per subtype ----
        for subtype, items in by_subtype.items():
            # items: list of (sample_id, pos×wave)
            sample_ids = [sid for sid, _ in items]
            mats       = [M for _, M in items]

            if len(mats) < 2:
                # Tucker is defined, but not very informative with 1 sample — still handle gracefully
                # We’ll create a 3D tensor with S=1.
                pass

            T = np.stack(mats, axis=0)   # [S × P × W]
            S, P, W = T.shape

            # choose ranks
            Rs = sample_components if (sample_components is not None) else max(1, min(3, S))
            Rp = max(1, min(pos_components, P))
            Rw = max(1, min(n_components, W))
            ranks = (Rs, Rp, Rw)

            # mean-centering across samples (repeats) per cell, if requested
            if centre == "with" and S > 1:
                mean_pw = T.mean(axis=0, keepdims=True)
                Tc = T - mean_pw
            else:
                Tc = T

            # Tucker decomposition (HOSVD)
            core, factors = tucker(Tc, rank=ranks, init='svd')
            U_s, U_p, U_w = factors  # [S,Rs], [P,Rp], [W,Rw]

            # store results under subtype
            data_dict[store_key][reg][subtype] = {
                "wave":    wave_ref,
                "pos":     pos_ref,
                "core":    tl.to_numpy(core),
                "factors": {
                    "sample":   tl.to_numpy(U_s),
                    "position": tl.to_numpy(U_p),
                    "spectral": tl.to_numpy(U_w),
                },
                "ranks":   ranks,
                "centre":  centre,
                "order":   ("Dermis" if use_dermis else "Whole"),
                "samples": sample_ids
            }

            # -------------------
            # Visualisation (top pos×spec pairs by energy)
            # -------------------
            if plot_mode_MPCA != "None":
                G = tl.to_numpy(core)               # [Rs, Rp, Rw]
                energy_pw = (G**2).sum(axis=0)      # [Rp, Rw]
                flat_idx = np.argsort(energy_pw.ravel())[::-1]
                K = min(4, energy_pw.size)
                chosen = [np.unravel_index(i, energy_pw.shape) for i in flat_idx[:K]]  # (p_idx, w_idx)

                maps = []
                for (pi, wi) in chosen:
                    M = np.outer(U_p[:, pi], U_w[:, wi])  # [P × W]
                    scale = np.sqrt((G[:, pi, wi]**2).sum())  # energy over sample mode
                    maps.append((pi, wi, M * scale))

                def _panel(maps, wave, pos, title):
                    ncols = 2
                    nrows = int(np.ceil(len(maps)/ncols))
                    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 3.8*nrows), squeeze=False)
                    for ax, (pi, wi, M) in zip(axes.flat, maps):
                        im = ax.imshow(M, aspect='auto', origin='lower',
                                       extent=[wave[0], wave[-1], pos[0], pos[-1]],
                                       cmap="viridis")
                        ax.set_xlabel("Wavenumber (cm⁻¹)")
                        ax.set_ylabel("Position (% depth)")
                        ax.set_title(f"pos#{pi+1} × spec#{wi+1}")
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    for ax in axes.flat[len(maps):]:
                        ax.axis("off")
                    fig.suptitle(title, y=0.99)
                    fig.tight_layout(rect=[0,0,1,0.95])
                    return fig

                title = f"{reg} — MPCA ({'Dermis' if use_dermis else 'Whole'}, centre='{centre}') — subtype={subtype}"
                fig = _panel(maps, wave_ref, pos_ref, title)

                if plot_mode_MPCA == "pdf":
                    pdf_path = outdir / f"MPCA_{reg}_{subtype}.pdf"
                    with PdfPages(pdf_path) as pdf:
                        pdf.savefig(fig)
                    plt.close(fig)
                    print(f"[PCA_MPCA] Saved {pdf_path}")
                else:
                    plt.show(); plt.close(fig)

    return data_dict

