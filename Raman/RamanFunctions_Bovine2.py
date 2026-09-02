#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 14:30:07 2026

@author: lauraforster
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
from matplotlib.backends.backend_pdf import PdfPages
import ramanspy
from pathlib import Path
from math import ceil
from scipy.signal import savgol_filter

def Types(Type):
    if Type == 'Bovine':
        Subtypes = ["fasicleWET", "fasicleDRY", "glassDRY", "glassONLY", "glassPBS", "sample"]
        Colours = {"fasicleWET": "blue", "fasicleDRY": "green", "glassDRY": "red",
                   "glassONLY": "peru", "glassPBS": "orange", "sample": "black"}
        Linestyles = {"fasicleWET": "-", "fasicleDRY": "-", "glassDRY": "--",
                      "glassONLY": "--", "glassPBS": "--", "sample": "-"}
    elif Type == 'WOUND':
        Subtypes = ["CT", "D7", "D14", ]
        Colours = {"CT": "grey", "D7": "tomato",  "D14": "royalblue", }
        Linestyles = {"CT": "-",  "D7": "--",   "D14": ":", }
    return Subtypes, Colours, Linestyles

def read_Samplemanifest(manifest_path):
    sample_df = pd.read_excel(manifest_path)
    return sample_df

def read_Peakmanifest(manifest_path):
    peak_df = pd.read_csv(manifest_path)
    return peak_df

def CreateDict(sample_manifest_df, Subtypes):
    """
    Combined manifest reader.

    Handles:
      1) Bovine-style manifest:
            TYPE | SAMPLE NAME | Sample Number

      2) Wound-style manifest:
            TYPE | FOLDER NAME | Sample Number
            files stored as DataDir / TYPE / f"{SampleNumber}A_linescan1.txt"

    Returns:
        data_dict[subtype] -> list[scan_entry]
    """

    data_dict = {}
    canon = {s.lower(): s for s in Subtypes}

    for _, row in sample_manifest_df.iterrows():
        sample_num_raw = row.get("Sample Number", np.nan)
        subtype_raw = row.get("TYPE", np.nan)

        if pd.isna(sample_num_raw) or pd.isna(subtype_raw):
            continue

        try:
            sample_num = str(int(sample_num_raw)).strip()
        except Exception:
            sample_num = str(sample_num_raw).strip()

        subtype_manifest = str(subtype_raw).strip()
        subtype_key = canon.get(subtype_manifest.lower())

        if subtype_key is None:
            continue

        # --------------------------------------------------
        # BOVINE STYLE: explicit SAMPLE NAME column
        # --------------------------------------------------
        sample_name_raw = row.get("SAMPLE NAME", np.nan)

        if not pd.isna(sample_name_raw) and str(sample_name_raw).strip().lower() not in ("", "nan"):
            sample_name = str(sample_name_raw).strip()
            fname = sample_name if sample_name.lower().endswith(".txt") else sample_name + ".txt"
            stem = fname[:-4]

            parts = stem.split("_")

            sampletype = parts[0] if len(parts) > 0 else ""
            substrate = parts[1] if len(parts) > 1 else None
            scantype = parts[2] if len(parts) > 2 else None
            wavelength = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None

            acc, exp = None, None
            if len(parts) > 4 and "a" in parts[4]:
                a, b = parts[4].split("a", 1)
                acc = int(a) if a.isdigit() else None
                try:
                    exp = float(b.split("-", 1)[0])
                except Exception:
                    exp = None

            otherinfo = "_".join(parts[5:]) if len(parts) > 5 else None
            folder_name = None

        # --------------------------------------------------
        # WOUND STYLE: no SAMPLE NAME, use Sample Number
        # --------------------------------------------------
        else:
            folder_name_raw = row.get("FOLDER NAME", "")
            folder_name = str(folder_name_raw).strip() if not pd.isna(folder_name_raw) else ""

            fname = f"{sample_num}A_linescan1.txt"
            stem = fname[:-4]

            sampletype = sample_num
            substrate = None
            scantype = "linescan"
            wavelength = 442
            acc = None
            exp = None
            otherinfo = folder_name

        scan_entry = {
            "SampleNumber": sample_num,
            "Subtype": subtype_key,
            "Type": subtype_key,
            "SampleName": stem,
            "Filename": fname,
            "SampleType": sampletype,
            "Substrate": substrate,
            "ScanType": scantype,
            "Wavelength": wavelength,
            "Accumulations": acc,
            "Exposure": exp,
            "OtherInfo": otherinfo,
            "FolderName": folder_name,
        }

        data_dict.setdefault(subtype_key, []).append(scan_entry)

    return data_dict
    
def readindata(DataDir, data_dict):
    DataDir = Path(DataDir)

    for subtype, scans in data_dict.items():
        for scan in scans:
            fname = scan["Filename"]

            possible_paths = [
                DataDir / fname,
                DataDir / subtype / fname,
            ]

            path = next((p for p in possible_paths if p.exists()), None)

            if path is None:
                print("File not found. Tried:")
                for p in possible_paths:
                    print(f"  - {p}")
                continue

            try:
                df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, engine="python")

                name = fname.lower()

                if "linescan" in name:
                    df = df.iloc[:, :4]
                    df.columns = ["X", "Y", "Wave", "Intensity"]
                    df[["X", "Y"]] = df[["X", "Y"]].ffill()

                elif "depthscan" in name:
                    df = df.iloc[:, :3]
                    df.columns = ["Z", "Wave", "Intensity"]
                    df["Z"] = df["Z"].ffill()

                else:
                    print(f"Unknown scan type: {fname}")
                    continue

                df = df[pd.to_numeric(df["Wave"], errors="coerce").notna()].copy()
                df["Wave"] = pd.to_numeric(df["Wave"], errors="coerce")
                df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")

                scan["Data"] = df

            except Exception as e:
                print(f"Could not read {path}: {e}")

    return data_dict

def SplitSpectra(data_dict, Colours, Linestyles, SN, step):

    for subtype, scans in data_dict.items():
        colour = Colours.get(subtype, None) if Colours else None
        linestyle = Linestyles.get(subtype, None) if Linestyles else None

        for scan in scans:
            if "Data" not in scan:
                continue

            df = scan["Data"]
            scantype = str(scan.get("ScanType", "")).lower()

            spectra = {}
            point_i = 0

            # -----------------------------
            # LINESCAN: split by (X, Y) blocks in order
            # -----------------------------
            if "line" in scantype and {"X", "Y", "Wave", "Intensity"}.issubset(df.columns):
                # make sure XY are filled if your reader didn't already
                df[["X", "Y"]] = df[["X", "Y"]].ffill()

                cur_xy = None
                w, I = [], []

                for x, y, wave, inten in df[["X", "Y", "Wave", "Intensity"]].itertuples(index=False, name=None):
                    xy = (x, y)

                    if cur_xy is None:
                        cur_xy = xy

                    # coordinate changed -> save previous spectrum
                    if xy != cur_xy:
                        point_i += 1
                        spectra[f"point{point_i}"] = {
                            "Coord": cur_xy,
                            "Wave": np.asarray(w),
                            "Intensity": np.asarray(I),
                        }
                        cur_xy = xy
                        w, I = [], []

                    w.append(wave)
                    I.append(inten)

                # flush last spectrum
                if w:
                    point_i += 1
                    spectra[f"point{point_i}"] = {
                        "Coord": cur_xy,
                        "Wave": np.asarray(w),
                        "Intensity": np.asarray(I),
                    }

            # -----------------------------
            # DEPTHSCAN: split by Z blocks in order
            # -----------------------------
            elif "depth" in scantype and {"Z", "Wave", "Intensity"}.issubset(df.columns):
                df["Z"] = df["Z"].ffill()

                cur_z = None
                w, I = [], []

                for z, wave, inten in df[["Z", "Wave", "Intensity"]].itertuples(index=False, name=None):
                    if cur_z is None:
                        cur_z = z

                    if z != cur_z:
                        point_i += 1
                        spectra[f"point{point_i}"] = {
                            "Coord": cur_z,
                            "Wave": np.asarray(w),
                            "Intensity": np.asarray(I),
                        }
                        cur_z = z
                        w, I = [], []

                    w.append(wave)
                    I.append(inten)

                if w:
                    point_i += 1
                    spectra[f"point{point_i}"] = {
                        "Coord": cur_z,
                        "Wave": np.asarray(w),
                        "Intensity": np.asarray(I),
                    }

            else:
                # fallback: treat as one spectrum
                point_i = 1
                spectra["point1"] = {
                    "Coord": None,
                    "Wave": df["Wave"].to_numpy(),
                    "Intensity": df["Intensity"].to_numpy(),
                }

            scan["Spectra"] = spectra
            scan.pop("Data", None)  # keep your memory-saving behaviour

            # -----------------------------
            # Optional plotting for checks
            # SN = sample number (tendon ID), step = plot every nth point
            # -----------------------------
            if SN is not None and str(scan.get("SampleNumber")) == str(SN):
                plt.figure(figsize=(10, 6))
                for i, (pname, spec) in enumerate(spectra.items(), start=1):
                    if step is None or i % step == 0:
                        coord = spec["Coord"]
                        if isinstance(coord, tuple):
                            lab = f"{pname} x={np.round(coord[0],2)}, y={np.round(coord[1],2)}"
                        elif coord is not None:
                            lab = f"{pname} z={np.round(coord,2)}"
                        else:
                            lab = pname

                        plt.plot(spec["Wave"], spec["Intensity"], label=lab,
                                 color=colour, linestyle=linestyle)

                # plt.title(f"{subtype} | {scan.get('SampleName', '')}")
                plt.xlabel("Wavenumber (cm$^{-1}$)")
                plt.ylabel("Intensity")
                plt.legend(fontsize="x-small", ncol=2)
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
    baseline = ramanspy.preprocessing.baseline.ASLS(p=0.01,lam=1e4)
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

def _pipeline_debug_fig(sc_raw, step_fns, titles, suptitle="", norm_band=None):
    """
    Multi-panel debug figure.

    Keeps 9 panels total by:
      - still APPLYING all steps in step_fns (so processing is faithful)
      - but HIDING the old:
          * "After normalisation (AUC)"
          * "After final crop (analysis window)"
      - and inserting the new three:
          5) After AUC normalisation (FP_crop)
          6) After normalisation to band (e.g. 1590–1720)
          7) Overlay of both
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from math import ceil

    def _auc(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if x.size < 2 or y.size < 2:
            return np.nan
        return float(np.trapezoid(y, x))

    def _crop_xy(x, y, region):
        if region is None:
            return x, y
        lo, hi = region
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = (x >= lo) & (x <= hi)
        return x[m], y[m]

    def _is_old_auc_panel(title):
        t = str(title).lower()
        return ("normalisation" in t and "auc" in t)

    def _is_old_final_crop_panel(title):
        t = str(title).lower()
        return ("final crop" in t) or ("analysis window" in t)
    
    def _negative_area_fraction(y):
        y = np.asarray(y, float)
        if y.size == 0 or not np.all(np.isfinite(y)):
            return np.nan
        denom = np.sum(np.abs(y)) + 1e-12
        return float(np.sum(np.abs(y[y < 0])) / denom)

    # Stages as list of tuples: (title, x, y, y_overlay_or_None)
    stages = []

    # step 0 (raw)
    x_prev = np.asarray(sc_raw.spectral_axis, float)
    y_prev = np.asarray(sc_raw.spectral_data, float)
    stages.append(("0 – Raw", x_prev, y_prev, None))

    sc_prev = sc_raw

    # capture baseline-corrected and final-crop output (even if we don't plot them)
    x_basecorr = y_basecorr = None
    x_final = y_final = None

    for i, (fn, name) in enumerate(zip(step_fns, titles), start=1):
        sc_curr = fn(sc_prev)
        x = np.asarray(sc_curr.spectral_axis, float)
        y = np.asarray(sc_curr.spectral_data, float)

        t_clean = str(name).strip()
        t_lc = t_clean.lower()
        is_baseline = ("baseline" in t_lc)

        # Always track final crop output (even if we hide its panel)
        if _is_old_final_crop_panel(t_clean):
            x_final, y_final = x, y

        # ---- HIDE the two old panels, but do NOT skip running the step ----
        hide_this_panel = _is_old_auc_panel(t_clean) or _is_old_final_crop_panel(t_clean)

        if is_baseline:
            # Align previous to current x
            if (len(x_prev) != len(x)) or (not np.allclose(x_prev, x, rtol=0, atol=1e-9)):
                y_prev_aligned = np.interp(x, x_prev, y_prev)
            else:
                y_prev_aligned = y_prev

            baseline_est = y_prev_aligned - y

            # preview + corrected baseline panels are kept
            stages.append((f"{i} – Baseline (preview)", x, baseline_est, y_prev_aligned))
            stages.append((f"{i} – {t_clean}", x, y, None))

            x_basecorr, y_basecorr = x, y

        else:
            if not hide_this_panel:
                stages.append((f"{i} – {t_clean}", x, y, None))

        sc_prev, x_prev, y_prev = sc_curr, x, y

    # Insert new post-baseline panels AFTER "After baseline" if possible
    insert_at = None
    for k, (t, *_rest) in enumerate(stages):
        if "after baseline" in t.lower():
            insert_at = k + 1
            break

    if (x_basecorr is not None) and (y_basecorr is not None) and (x_final is not None) and (y_final is not None):
        # Align baseline-corrected signal onto final crop axis
        if (len(x_basecorr) != len(x_final)) or (not np.allclose(x_basecorr, x_final, rtol=0, atol=1e-9)):
            y_base_on_final = np.interp(x_final, x_basecorr, y_basecorr)
        else:
            y_base_on_final = y_basecorr

        # Option 1: AUC normalise on FP_crop (i.e. final crop window)
        auc_crop = _auc(x_final, y_base_on_final)
        denom1 = auc_crop if (np.isfinite(auc_crop) and abs(auc_crop) > 0) else 1.0
        y_opt1 = y_base_on_final / denom1

        # Option 2: normalise to AUC in norm_band (within final window)
        if norm_band is not None:
            xb, yb = _crop_xy(x_final, y_base_on_final, norm_band)
            auc_band = _auc(xb, yb)
        else:
            auc_band = np.nan
        denom2 = auc_band if (np.isfinite(auc_band) and abs(auc_band) > 0) else 1.0
        y_opt2 = y_base_on_final / denom2

        new_panels = [
            ("5 – After AUC normalisation (FP_crop)", x_final, y_opt1, None),
            (f"6 – After normalisation to band {norm_band}", x_final, y_opt2, None),
            ("7 – Overlay: FP_crop AUC vs band-normalised", x_final, y_opt1, y_opt2),
        ]

        if insert_at is None:
            stages.extend(new_panels)
        else:
            stages[insert_at:insert_at] = new_panels

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
        ax.plot(x, y, lw=1.6)
        if y2 is not None:
            ax.plot(x, y2, lw=1.6)
        
        # ---- Baseline-corrected diagnostic panel ----
        if "after baseline" in title_i.lower():
            # horizontal zero line
            ax.axhline(0, color="k", lw=1.0, alpha=0.7)
        
            # negative area fraction
            naf = _negative_area_fraction(y)
        
            ax.text(
                0.02, 0.95,
                f"Neg. area frac ≈ {naf:.3f}",
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="0.7", alpha=0.85)
            )

        # ax.set_title(title_i, fontsize=11)
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.grid(True, linestyle=":", alpha=0.25)

        if "baseline (preview)" in title_i.lower():
            ax.legend(["baseline", "previous"], fontsize=9, frameon=True, framealpha=0.8)

        if "overlay: fp_crop auc vs band-normalised" in title_i.lower():
            ax.legend(["FP_crop AUC", "Band-normalised"], fontsize=9, frameon=True, framealpha=0.8)

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

def _normalise_sc(sc_in, mode, band):
    """
    mode:
      - "crop": divide by AUC over the current window
      - "band": divide by AUC over `band` within current window; falls back to crop if band invalid
    """
    w = np.asarray(sc_in.spectral_axis, float)
    y = np.asarray(sc_in.spectral_data, float)

    if w.size < 2 or y.size < 2 or not np.all(np.isfinite(y)):
        return sc_in

    denom = np.nan

    if mode == "band" and band is not None:
        lo, hi = float(band[0]), float(band[1])
        m = (w >= lo) & (w <= hi) & np.isfinite(w) & np.isfinite(y)
        if np.any(m):
            denom = float(np.trapezoid(y[m], w[m]))

    if not (np.isfinite(denom) and denom != 0):
        denom = float(np.trapezoid(y, w))

    if not (np.isfinite(denom) and denom != 0):
        denom = 1.0

    sc_out = ramanspy.SpectralContainer(y / denom, w)

    if hasattr(sc_in, "metadata"):
        try:
            sc_out.metadata = sc_in.metadata
        except Exception:
            pass

    return sc_out

def TreatSpectra(data_dict, Save_folder, Preprocess,
                 FP_full_785, FP_full_633, FP_full_532, FP_full_442,
                 FP_crop_785, FP_crop_633, FP_crop_532, FP_crop_442,
                 FP_band_785, FP_band_633, FP_band_532, FP_band_442,
                 colours, linestyles, plotall_treat, treatmentorder, normalisation,
                 SN, step):
    """
    FP-only preprocessing for the Bovine dict layout:
      data_dict[subtype] -> list[scan]
      scan["Spectra"]["pointN"] -> {"Coord": (x,y) or z, "Wave": array, "Intensity": array}

    plotall_treat: "screen" | "pdf" | "false"
      - "screen": show ONLY the pipeline debug figure (no overlay figure)
      - "pdf": save debug + overlay figures to ONE PDF PER TENDON ID (SampleNumber)
      - "false": no plots

    SN, step: plotting filters ONLY (processing always runs for all spectra)
      - SN: tendon ID (int/str) or list/set/tuple of tendon IDs
      - step: plot every nth spectrum by point index (0, step, 2*step, ...)
    """

    # -------------------------
    # Parse plot mode
    # -------------------------
    mode = str(plotall_treat).strip().lower()
    if mode in ("false", "none", "0", ""):
        plot_mode = "false"
    elif mode in ("screen", "pdf"):
        plot_mode = mode
    else:
        plot_mode = "screen" if bool(plotall_treat) else "false"

    # -------------------------
    # Treatment order
    # -------------------------
    order_mode = str(treatmentorder).strip().lower()
    if order_mode not in ("before", "after"):
        print(f"[TreatSpectra] Unknown treatmentorder='{treatmentorder}', using 'before'.")
        order_mode = "before"
    norm_mode = str(normalisation).strip().lower()
    if norm_mode not in ("crop", "band"):
        print(f"[TreatSpectra] Unknown normalisation='{normalisation}', using 'crop'.")
        norm_mode = "crop"

    # -------------------------
    # Plot sample filter (PLOTTING ONLY)
    # -------------------------
    if SN is None:
        plot_samples = None
    else:
        plot_samples = set(SN) if isinstance(SN, (list, tuple, set)) else {SN}

    # -------------------------
    # Output dir + PDF handles (ONE PER TENDON ID)
    # -------------------------
    outdir = None
    pdf_by_tendon = {}
    if plot_mode == "pdf":
        outdir = Path(Save_folder) / "Outputs"
        outdir.mkdir(parents=True, exist_ok=True)

    def _get_pdf(sample_num):
        if plot_mode != "pdf":
            return None
        if sample_num not in pdf_by_tendon:
            pdf_path = outdir / f"Tendon_{sample_num}_Treatedspectra.pdf"
            pdf_by_tendon[sample_num] = PdfPages(pdf_path)
        return pdf_by_tendon[sample_num]

    # -------------------------
    # Safety wrapper: avoid NaNs / divide-by-zero outputs from Ramanspy steps
    # -------------------------
    def _safe_apply(sc_prev, fn):
        try:
            y_prev = np.asarray(sc_prev.spectral_data, dtype=float)
        except Exception:
            return sc_prev

        if (y_prev.size == 0) or (not np.all(np.isfinite(y_prev))) or (np.nanstd(y_prev) == 0):
            return sc_prev

        try:
            sc_out = fn(sc_prev)
        except Exception:
            return sc_prev

        try:
            y_out = np.asarray(sc_out.spectral_data, dtype=float)
        except Exception:
            return sc_prev

        if (y_out.size == 0) or (not np.all(np.isfinite(y_out))):
            return sc_prev

        return sc_out

    def _wrap_steps(step_fns, titles):
        wrapped = []
        for fn, title in zip(step_fns, titles):
            t = str(title).lower()
            if "crop" in t:
                wrapped.append(fn)
            else:
                wrapped.append(lambda sc, fn=fn: _safe_apply(sc, fn))
        return wrapped

    # -------------------------
    # Select FP ranges per wavelength
    # -------------------------
    fp_full_by_wl = {
        785: FP_full_785,
        633: FP_full_633,
        532: FP_full_532,
        442: FP_full_442,
    }
    fp_crop_by_wl = {
        785: FP_crop_785,
        633: FP_crop_633,
        532: FP_crop_532,
        442: FP_crop_442,
    }
    
    # Paper-style baseline anchors (cm^-1)
    LINEAR_ANCHORS = [700, 800, 834, 903, 991, 1018, 1145, 1216, 1362, 1510, 1590, 1720]

    # Precompute selected steps once
    selected = _normalize_steps(Preprocess)

    # Cache pipelines per wavelength (so we don't rebuild them per spectrum)
    pipeline_cache = {}  # wl -> (full_safe, titles_full, final_safe, titles_final, crop_fn_safe)

    def _get_pipelines_for_wavelength(wl):
        if wl not in pipeline_cache:
            full_rng = fp_full_by_wl.get(wl, FP_full_785)
            crop_rng = fp_crop_by_wl.get(wl, FP_crop_785)

            step_fns_full, titles_full, step_fns_final, titles_final = _build_region_steps(full_rng, crop_rng, selected)
            step_fns_full_safe  = _wrap_steps(step_fns_full,  titles_full)
            step_fns_final_safe = _wrap_steps(step_fns_final, titles_final)
            crop_fn_safe = step_fns_final_safe[-1]

            pipeline_cache[wl] = (step_fns_full_safe, titles_full, step_fns_final_safe, titles_final, crop_fn_safe)

        return pipeline_cache[wl]

    # Stable point ordering
    def _point_sort_key(k):
        try:
            return int(str(k).replace("point", ""))
        except Exception:
            return 10**9

    try:
        for subtype, scans in data_dict.items():
            for scan in scans:
                if "Spectra" not in scan:
                    continue

                sample_num = scan.get("SampleNumber", None)
                sample_name = scan.get("SampleName", scan.get("Filename", ""))
                wl = scan.get("Wavelength", None)

                step_fns_full_safe, titles_full, step_fns_final_safe, titles_final, crop_fn_safe = _get_pipelines_for_wavelength(wl)

                treated_full = {}
                treated_crop = {}

                point_keys = sorted(scan["Spectra"].keys(), key=_point_sort_key)

                for idx, pkey in enumerate(point_keys):
                    spec = scan["Spectra"][pkey]
                    wave = np.asarray(spec["Wave"], float)
                    inten = np.asarray(spec["Intensity"], float)
                    coord = spec.get("Coord", None)

                    # Plot filters (PLOTTING ONLY)
                    plot_allowed_sample = (plot_samples is None) or (sample_num in plot_samples)
                    plot_step_ok = (step in (None, 0, 1)) or (isinstance(step, int) and step > 1 and idx % step == 0)
                    plot_this = (plot_mode != "false") and plot_allowed_sample and plot_step_ok

                    # Ensure axis order
                    if wave.size and np.any(np.diff(wave) < 0):
                        order = np.argsort(wave)
                        wave = wave[order]
                        inten = inten[order]

                    sc_raw = ramanspy.SpectralContainer(inten, wave)
                    sc_raw.metadata = {
                        "coord": coord, "index": idx, "point": pkey,
                        "sample_num": sample_num, "subtype": subtype,
                        "scan": sample_name, "wavelength": wl
                    }

                    if order_mode == "before":
                        sc_full = sc_raw
                        for fn in step_fns_full_safe:
                            sc_full = fn(sc_full)

                        sc_final = crop_fn_safe(sc_full)

                        fp_band_by_wl = {785: FP_band_785, 633: FP_band_633, 532: FP_band_532, 442: FP_band_442}
                        norm_band = fp_band_by_wl.get(wl, (1590, 1720))
                        
                        sc_final_norm = _normalise_sc(sc_final, norm_mode, norm_band)
                        
                        treated_full[pkey] = sc_full
                        treated_crop[pkey] = sc_final_norm

                        if plot_this:
                            if isinstance(coord, tuple):
                                coord_txt = f"x={np.round(coord[0],2)}, y={np.round(coord[1],2)}"
                            elif coord is not None:
                                coord_txt = f"z={np.round(coord,2)}"
                            else:
                                coord_txt = "coord=?"

                            title = f"{subtype} | tendon {sample_num} | {sample_name} | {wl}nm | {pkey} | {coord_txt}"
                            fp_band_by_wl = {785: FP_band_785, 633: FP_band_633, 532: FP_band_532, 442: FP_band_442}
                            norm_band = fp_band_by_wl.get(wl, (1590, 1720))


                            fig_dbg = _pipeline_debug_fig(sc_raw, step_fns_final_safe, titles_final, title + " (before)", norm_band=norm_band)

                            if plot_mode == "pdf":
                                fig_ov = _overlay_fig(
                                    np.asarray(sc_raw.spectral_axis, float),  np.asarray(sc_raw.spectral_data, float),
                                    np.asarray(sc_full.spectral_axis, float), np.asarray(sc_full.spectral_data, float),
                                    np.asarray(sc_final_norm.spectral_axis, float), np.asarray(sc_final_norm.spectral_data, float),
                                    title + " (before)"
                                )
                                pdf = _get_pdf(sample_num)
                                pdf.savefig(fig_dbg); plt.close(fig_dbg)
                                pdf.savefig(fig_ov);  plt.close(fig_ov)
                            else:  # screen: ONLY debug fig
                                plt.show(); plt.close(fig_dbg)

                    else:
                        sc_crop_first = crop_fn_safe(sc_raw)
                        sc_proc = sc_crop_first
                        for fn in step_fns_full_safe:
                            sc_proc = fn(sc_proc)
                            
                        fp_band_by_wl = {785: FP_band_785, 633: FP_band_633, 532: FP_band_532, 442: FP_band_442}
                        norm_band = fp_band_by_wl.get(wl, (1590, 1720))
                        
                        sc_final_norm = _normalise_sc(sc_proc, norm_mode, norm_band)

                        sc_full = sc_raw
                        for fn in step_fns_full_safe:
                            sc_full = fn(sc_full)

                        treated_full[pkey] = sc_full
                        treated_crop[pkey] = sc_final_norm

                        if plot_this:
                            if isinstance(coord, tuple):
                                coord_txt = f"x={np.round(coord[0],2)}, y={np.round(coord[1],2)}"
                            elif coord is not None:
                                coord_txt = f"z={np.round(coord,2)}"
                            else:
                                coord_txt = "coord=?"

                            title = f"{subtype} | tendon {sample_num} | {sample_name} | {wl}nm | {pkey} | {coord_txt}"
                            fig_dbg = _pipeline_debug_fig(
                                sc_raw,
                                [crop_fn_safe] + step_fns_full_safe,
                                ["Crop(ROI)"] + titles_full,
                                title + " (after)"
                            )

                            if plot_mode == "pdf":
                                fig_ov = _overlay_fig(
                                    np.asarray(sc_raw.spectral_axis, float),  np.asarray(sc_raw.spectral_data, float),
                                    np.asarray(sc_full.spectral_axis, float), np.asarray(sc_full.spectral_data, float),
                                    np.asarray(sc_final_norm.spectral_axis, float), np.asarray(sc_final_norm.spectral_data, float),
                                    title + " (after)"
                                )
                                pdf = _get_pdf(sample_num)
                                pdf.savefig(fig_dbg); plt.close(fig_dbg)
                                pdf.savefig(fig_ov);  plt.close(fig_ov)
                            else:  # screen: ONLY debug fig
                                plt.show(); plt.close(fig_dbg)

                scan["Spectra_Treated_Full"] = treated_full
                scan["Spectra_Treated"] = treated_crop

    finally:
        for pdf in pdf_by_tendon.values():
            try:
                pdf.close()
            except Exception:
                pass

    return data_dict




def PlotMiddleAverageSpectra(data_dict, sample_names,
                             middle_fraction=0.20,
                             smoothing=0,
                             use_cropped=True,
                             axvlines=None,
                             xlim=None,
                             verbose=False):
    """
    For each scan whose SampleName matches sample_names (case-insensitive, .txt optional):
      - take treated spectra (cropped or full)
      - average the middle `middle_fraction` of points
      - optionally smooth the averaged spectrum
      - overlay all averaged spectra

    axvlines : list[float] or None
        Raman shifts (cm^-1) to draw as vertical reference lines

    xlim : tuple(float, float) or None
        X-axis limits (cm^-1), e.g. (800, 1200)
    """


    def _point_sort_key(k):
        try:
            return int(str(k).replace("point", ""))
        except Exception:
            return 10**9

    def _smooth(y, pct):
        if pct <= 0:
            return y
        n = len(y)
        frac = np.clip(pct / 100.0, 0.0, 1.0)
        win = int(round(n * (0.02 + 0.23 * frac)))
        win = max(5, win)
        if win % 2 == 0:
            win += 1
        win = min(win, n - (1 - n % 2))
        try:
            return savgol_filter(y, window_length=win, polyorder=3)
        except Exception:
            return y

    wanted = {str(s).strip().lower().replace(".txt", "") for s in sample_names}
    averaged = []
    matched = set()

    key = "Spectra_Treated" if use_cropped else "Spectra_Treated_Full"

    for subtype, scans in data_dict.items():
        for scan in scans:
            sname = str(scan.get("SampleName", "")).strip().lower()
            if sname not in wanted:
                continue

            matched.add(sname)

            if key not in scan or not scan[key]:
                continue

            treated = scan[key]
            pkeys = sorted(treated.keys(), key=_point_sort_key)
            n = len(pkeys)
            if n == 0:
                continue

            frac = np.clip(float(middle_fraction), 0.0, 1.0)
            k = max(1, int(round(n * frac)))
            start = (n - k) // 2
            sel_keys = pkeys[start:start + k]

            sc0 = treated[sel_keys[0]]
            x = np.asarray(sc0.spectral_axis, float)

            Ys = []
            for pk in sel_keys:
                sc = treated[pk]
                y = np.asarray(sc.spectral_data, float)
                x_sc = np.asarray(sc.spectral_axis, float)

                if len(x_sc) != len(x) or not np.allclose(x_sc, x, atol=1e-9):
                    y = np.interp(x, x_sc, y)

                Ys.append(y)

            y_mean = np.nanmean(np.vstack(Ys), axis=0)
            y_mean = _smooth(y_mean, smoothing)

            wl = scan.get("Wavelength", "")
            sub = scan.get("Subtype", subtype)
            label = f"{scan.get('SampleName','')} | {sub} | {wl}nm | mid {k}/{n}"

            averaged.append((label, x, y_mean))

    if verbose:
        missing = sorted(wanted - matched)
        if missing:
            print("[PlotMiddleAverageSpectra] Requested but not found:")
            for m in missing:
                print("  -", m)

    if not averaged:
        print("[PlotMiddleAverageSpectra] No matching samples found.")
        return None

    # ---------------- plot ----------------
    plt.figure(figsize=(10, 6))

    for label, x, y in averaged:
        plt.plot(x, y, label=label)

    # Vertical reference lines
    if axvlines:
        for i, xv in enumerate(axvlines):
            plt.axvline(
                xv,
                color="k",
                linestyle="--",
                alpha=0.4,
                linewidth=1,
                label="Reference peaks" if i == 0 else None
            )

    if xlim is not None:
        plt.xlim(xlim)
    plt.axhline(0, color='k')
    plt.ylim(0,0.04)
    plt.title(
        f"Middle-average spectra overlay "
        f"(middle={middle_fraction:.2f}, smoothing={smoothing}, "
        f"{'cropped' if use_cropped else 'full'})"
    )
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity (a.u.)")
    plt.legend(fontsize="x-small")
    plt.tight_layout()
    plt.show()

    return averaged



def DebugFindScans(data_dict, needle="sample1", show=50):
    needle = str(needle).lower()
    hits = []

    for subtype, scans in data_dict.items():
        for scan in scans:
            sname = str(scan.get("SampleName", "")).lower()
            fname = str(scan.get("Filename", "")).lower()
            if needle in sname or needle in fname:
                hits.append((subtype, scan))

    print(f"\n[DEBUG] Find scans containing '{needle}': {len(hits)} hit(s)\n")

    for i, (subtype, scan) in enumerate(hits[:show], start=1):
        sample_num = scan.get("SampleNumber", None)
        sname = scan.get("SampleName", "")
        fname = scan.get("Filename", "")
        wl = scan.get("Wavelength", None)
        st = scan.get("ScanType", None)
        n_spec = len(scan.get("Spectra", {}) or {})
        n_full = len(scan.get("Spectra_Treated_Full", {}) or {})
        n_crop = len(scan.get("Spectra_Treated", {}) or {})

        print(f"{i:02d}. subtype={subtype} | tendon={sample_num} | wl={wl} | scantype={st}")
        print(f"    SampleName: {sname}")
        print(f"    Filename  : {fname}")
        print(f"    Spectra points           : {n_spec}")
        print(f"    Spectra_Treated_Full pts : {n_full}")
        print(f"    Spectra_Treated pts      : {n_crop}")
        print("")

    if len(hits) > show:
        print(f"... showing first {show} of {len(hits)} hits\n")

    return hits










def ExportWeightedMoments_AveragedAndAll(
    data_dict,
    sample_names,
    peak_regions,
    out_xlsx,
    middle_fraction=0.20,
    smoothing=0,
    use_cropped=True,
    verbose=True,
):
    """
    Export weighted-moment peak metrics to one Excel workbook:
      - Sheet 'Averaged': one row per scan (middle-average spectrum)
      - Sheet per scan: one row per point spectrum with Coord (x,y) or z

    Parameters
    ----------
    data_dict : dict
        data_dict[subtype] -> list[scan]
        scan has keys like SampleName, Subtype, Substrate, ScanType, Wavelength, etc.
        scan["Spectra_Treated"] or ["Spectra_Treated_Full"] -> dict(pointN -> ramanspy.SpectralContainer)
    sample_names : list[str]
        Names like "fasicleDRY_glass_linescan_442_10a0.5" ('.txt' optional)
    peak_regions : list[tuple]
        e.g. [("Amide I", (1550,1800)), ("CH2/CH3", (1150,1450))]
    out_xlsx : str | Path
        Output excel path
    middle_fraction : float
        fraction of points to average from the middle of the scan
    smoothing : float (0..100)
        SavGol smoothing strength for averaged spectra only
    use_cropped : bool
        True: use scan["Spectra_Treated"] (your chosen normalisation output)
        False: use scan["Spectra_Treated_Full"]
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from scipy.signal import savgol_filter

    # -------------------------
    # small helpers
    # -------------------------
    def _norm_name(s):
        return str(s).strip().lower().replace(".txt", "")

    wanted = {_norm_name(s) for s in sample_names}

    key = "Spectra_Treated" if use_cropped else "Spectra_Treated_Full"

    def _point_sort_key(k):
        try:
            return int(str(k).replace("point", ""))
        except Exception:
            return 10**9

    def _smooth_avg(y, pct):
        pct = float(pct)
        if pct <= 0:
            return y
        n = len(y)
        frac = np.clip(pct / 100.0, 0.0, 1.0)
        win = int(round(n * (0.02 + 0.23 * frac)))  # ~2% → ~25% of length
        win = max(5, win)
        if win % 2 == 0:
            win += 1
        win = min(win, n - (1 - n % 2))
        try:
            return savgol_filter(y, window_length=win, polyorder=3)
        except Exception:
            return y

    def _weighted_moments(x, y, xlim, max_order=3):
        """
        Weighted moments on a region.
        Uses nonnegative weights: w = y - min(y) if min(y)<0 else y, then clip at 0.
        Returns:
          m1 (mean), mu2 (variance), mu3 (3rd central), sigma, skewness, area, neg_area_frac
        """
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        lo, hi = float(xlim[0]), float(xlim[1])
        m = (x >= lo) & (x <= hi) & np.isfinite(x) & np.isfinite(y)
        if not np.any(m):
            return None

        xr = x[m]
        yr = y[m]

        # negative area fraction on the *signal* (before weight-fixing)
        area_total = float(np.trapezoid(np.abs(yr), xr)) if xr.size > 1 else float("nan")
        area_neg = float(np.trapezoid(np.clip(-yr, 0, None), xr)) if xr.size > 1 else float("nan")
        neg_area_frac = (area_neg / area_total) if (np.isfinite(area_total) and area_total > 0) else float("nan")

        # ensure nonnegative weights for moments
        offset = -min(0.0, float(np.nanmin(yr)))
        w = np.clip(yr + offset, 0, None)
        wsum = float(np.nansum(w))
        if not (np.isfinite(wsum) and wsum > 0):
            return {
                "m1": float("nan"),
                "mu2": float("nan"),
                "mu3": float("nan"),
                "sigma": float("nan"),
                "skewness": float("nan"),
                "area_w": float("nan"),
                "neg_area_frac": neg_area_frac,
                "n_points": int(xr.size),
            }

        m1 = float(np.nansum(w * xr) / wsum)
        mu2 = float(np.nansum(w * (xr - m1) ** 2) / wsum)
        mu3 = float(np.nansum(w * (xr - m1) ** 3) / wsum)
        sigma = float(np.sqrt(mu2)) if (np.isfinite(mu2) and mu2 >= 0) else float("nan")
        skew = float(mu3 / (mu2 ** 1.5)) if (np.isfinite(mu2) and mu2 > 0 and np.isfinite(mu3)) else float("nan")
        area_w = float(np.trapezoid(w, xr)) if xr.size > 1 else float("nan")

        return {
            "m1": m1,
            "mu2": mu2,
            "mu3": mu3,
            "sigma": sigma,
            "skewness": skew,
            "area_w": area_w,
            "neg_area_frac": neg_area_frac,
            "n_points": int(xr.size),
        }

    def _iter_matching_scans():
        matched = set()
        for subtype, scans in data_dict.items():
            for scan in scans:
                sname = _norm_name(scan.get("SampleName", ""))
                if sname in wanted:
                    matched.add(sname)
                    yield scan, sname
        if verbose:
            missing = sorted(wanted - matched)
            if missing:
                print("[ExportWeightedMoments] Requested but not found in data_dict:")
                for m in missing:
                    print("  -", m)

    def _sheet_safe(name):
        # Excel sheet max = 31
        s = str(name)
        s = s.replace("/", "_").replace("\\", "_").replace("[", "").replace("]", "").replace("*", "").replace("?", "")
        s = s.replace(":", "_")
        return s[:31] if len(s) > 31 else s

    # -------------------------
    # build results
    # -------------------------
    rows_avg = []
    per_scan_tables = {}  # sheet_name -> DataFrame

    for scan, sname in _iter_matching_scans():
        treated = scan.get(key, {})
        if not treated:
            if verbose:
                print(f"[ExportWeightedMoments] Missing {key} for {scan.get('SampleName','')}")
            continue

        pkeys = sorted(treated.keys(), key=_point_sort_key)
        n = len(pkeys)
        if n == 0:
            continue

        # ---- averaged spectrum from middle fraction ----
        frac = float(np.clip(middle_fraction, 0.0, 1.0))
        k = max(1, int(round(n * frac)))
        start = (n - k) // 2
        sel = pkeys[start:start + k]

        sc0 = treated[sel[0]]
        x0 = np.asarray(sc0.spectral_axis, float)

        Ys = []
        for pk in sel:
            sc = treated[pk]
            x = np.asarray(sc.spectral_axis, float)
            y = np.asarray(sc.spectral_data, float)
            if (len(x) != len(x0)) or (not np.allclose(x, x0, atol=1e-9, rtol=0)):
                y = np.interp(x0, x, y)
            Ys.append(y)

        y_mean = np.nanmean(np.vstack(Ys), axis=0)
        y_mean = _smooth_avg(y_mean, smoothing)

        base_meta = {
            "SampleName": scan.get("SampleName", ""),
            "Subtype": scan.get("Subtype", scan.get("Type", "")),
            "SampleNumber": scan.get("SampleNumber", ""),
            "Substrate": scan.get("Substrate", ""),
            "ScanType": scan.get("ScanType", ""),
            "Wavelength": scan.get("Wavelength", ""),
            "Accumulations": scan.get("Accumulations", ""),
            "Exposure": scan.get("Exposure", ""),
            "OtherInfo": scan.get("OtherInfo", ""),
            "MiddleFraction": frac,
            "MiddleK": k,
            "NpointsTotal": n,
            "AvgSmoothingPct": float(smoothing),
            "SourceKey": key,
        }

        row = dict(base_meta)
        for region_name, xlim in peak_regions:
            moms = _weighted_moments(x0, y_mean, xlim)
            if moms is None:
                for col in ("m1", "sigma", "skewness", "area_w", "neg_area_frac", "mu2", "mu3", "n_points"):
                    row[f"{region_name}_{col}"] = np.nan
            else:
                for col, val in moms.items():
                    row[f"{region_name}_{col}"] = val
        rows_avg.append(row)

        # ---- per-point spectra moments ----
        # map point key -> original 1-based index along the scan
        orig_index = {pk: (i + 1) for i, pk in enumerate(pkeys)}
        
        rows_pt = []
        for pk in pkeys:
            sc = treated[pk]
            x = np.asarray(sc.spectral_axis, float)
            y = np.asarray(sc.spectral_data, float)
        
            coord = None
            try:
                coord = getattr(sc, "metadata", {}).get("coord", None)
            except Exception:
                coord = None
        
            pt_row = dict(base_meta)
            pt_row.update({
                "PointKey": pk,
                "PointIndex": orig_index.get(pk, np.nan),  # <-- original position along line (1..N)
                "Coord": coord,
                "X": coord[0] if isinstance(coord, tuple) and len(coord) == 2 else np.nan,
                "Y": coord[1] if isinstance(coord, tuple) and len(coord) == 2 else np.nan,
                "Z": coord if (not isinstance(coord, tuple)) else np.nan,
            })
        
            for region_name, xlim in peak_regions:
                moms = _weighted_moments(x, y, xlim)
                if moms is None:
                    for col in ("m1", "sigma", "skewness", "area_w", "neg_area_frac", "mu2", "mu3", "n_points"):
                        pt_row[f"{region_name}_{col}"] = np.nan
                else:
                    for col, val in moms.items():
                        pt_row[f"{region_name}_{col}"] = val
        
            rows_pt.append(pt_row)

        sheet = _sheet_safe(scan.get("SampleName", sname))
        per_scan_tables[sheet] = pd.DataFrame(rows_pt)

    df_avg = pd.DataFrame(rows_avg)

    # -------------------------
    # write Excel
    # -------------------------
    out_xlsx = Path(out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_avg.to_excel(writer, sheet_name="Averaged", index=False)
        for sheet, df in per_scan_tables.items():
            df.to_excel(writer, sheet_name=sheet, index=False)

    if verbose:
        print(f"[ExportWeightedMoments] Wrote: {out_xlsx}")
        print(f"  - Averaged rows: {len(df_avg)}")
        print(f"  - Per-scan sheets: {len(per_scan_tables)}")

    return df_avg, per_scan_tables




def _align_spectralcontainers(sc_list):
    sc_list = [sc for sc in sc_list if sc is not None]
    if not sc_list:
        return None, None

    x0 = np.asarray(sc_list[0].spectral_axis, float)
    Ys = []

    for sc in sc_list:
        x = np.asarray(sc.spectral_axis, float)
        y = np.asarray(sc.spectral_data, float)

        if (x.shape != x0.shape) or (not np.allclose(x, x0, rtol=0, atol=1e-9)):
            y = np.interp(x0, x, y)

        Ys.append(y)

    return x0, np.vstack(Ys)


def GetWoundAverageSpectraForOverlay(
    data_dict,
    TypestoPlot=("CT", "D7", "D14"),
    region="dermis",
    use_FP=True,
    use_treated=True,
):
    """
    Returns one averaged spectrum per wound subtype.
    Uses the wound-code data_dict:
        data_dict[sample_num]["FP_Spectra_Treated_Dermis"]
    """

    reg_short = "FP" if use_FP else "EXT"
    treated_tag = "Treated" if use_treated else "Treated_Full"
    region_key = f"{reg_short}_Spectra_{treated_tag}_{region.capitalize()}"

    curves = {}

    for subtype in TypestoPlot:
        sc_all = []

        for sample_num, sample_data in data_dict.items():
            if not isinstance(sample_data, dict):
                continue

            if str(sample_data.get("Subtype", "")).strip() != subtype:
                continue

            spectra_dict = sample_data.get(region_key, None)

            if not isinstance(spectra_dict, dict) or not spectra_dict:
                continue

            sc_all.extend(list(spectra_dict.values()))

        x, Y = _align_spectralcontainers(sc_all)

        if x is None:
            print(f"[GetWoundAverageSpectraForOverlay] No spectra found for {subtype} using {region_key}")
            continue

        curves[subtype] = {
            "label": subtype,
            "x": x,
            "y": np.nanmean(Y, axis=0),
            "n_spectra": Y.shape[0],
        }

    return curves


def GetBovineMiddleAverageSpectraForOverlay(
    data_dict,
    sample_names,
    middle_fraction=0.8,
    use_cropped=True,
):
    """
    Returns one middle-average spectrum per selected bovine scan.
    Uses the bovine-code data_dict:
        data_dict[subtype] -> list[scan]
    """

    def _norm_name(s):
        return str(s).strip().lower().replace(".txt", "")

    def _point_sort_key(k):
        try:
            return int(str(k).replace("point", ""))
        except Exception:
            return 10**9

    wanted = {_norm_name(s) for s in sample_names}
    key = "Spectra_Treated" if use_cropped else "Spectra_Treated_Full"

    curves = {}
    matched = set()

    for subtype, scans in data_dict.items():
        for scan in scans:
            sname = _norm_name(scan.get("SampleName", ""))

            if sname not in wanted:
                continue

            matched.add(sname)

            treated = scan.get(key, {})
            if not treated:
                print(f"[GetBovineMiddleAverageSpectraForOverlay] Missing {key} for {scan.get('SampleName', '')}")
                continue

            pkeys = sorted(treated.keys(), key=_point_sort_key)
            n = len(pkeys)

            frac = float(np.clip(middle_fraction, 0.0, 1.0))
            k = max(1, int(round(n * frac)))
            start = (n - k) // 2
            sel_keys = pkeys[start:start + k]

            sc_list = [treated[pk] for pk in sel_keys]
            x, Y = _align_spectralcontainers(sc_list)

            if x is None:
                continue

            label = scan.get("SampleName", sname)

            curves[label] = {
                "label": label,
                "x": x,
                "y": np.nanmean(Y, axis=0),
                "n_spectra": Y.shape[0],
            }

    missing = sorted(wanted - matched)
    if missing:
        print("[GetBovineMiddleAverageSpectraForOverlay] Requested but not found:")
        for m in missing:
            print("  -", m)

    return curves


def PlotWoundBovineOverlay(
    wound_curves,
    bovine_curves,
    axvlines=None,
    xlim=None,
    ylim=None,
    title="Wound skin averages vs bovine fascicle spectra",
):
    plt.figure(figsize=(10, 6))

    wound_styles = {
        "CT":  {"color": "grey", "linestyle": "-",  "linewidth": 2.5},
        "D7":  {"color": "tomato", "linestyle": "-", "linewidth": 2.5},
        "D14": {"color": "royalblue", "linestyle": "-", "linewidth": 2.5},
    }

    bovine_styles = {
        "fasicleWET": {"color": "blue", "linestyle": "-", "linewidth": 2.0},
        "fasicleDRY": {"color": "green", "linestyle": "-", "linewidth": 2.0},
    }

    for subtype, curve in wound_curves.items():
        style = wound_styles.get(subtype, {"linewidth": 2.5})
        label = f"{subtype} skin average | n={curve['n_spectra']}"
        plt.plot(curve["x"], curve["y"], label=label, **style)

    for name, curve in bovine_curves.items():
        style_key = "fasicleWET" if "fasiclewet" in name.lower() else "fasicleDRY"
        style = bovine_styles.get(style_key, {"linestyle": "--", "linewidth": 2.0})
        label = f"{name} | middle n={curve['n_spectra']}"
        plt.plot(curve["x"], curve["y"], label=label, **style)

    if axvlines:
        for i, xv in enumerate(axvlines):
            plt.axvline(
                xv,
                color="k",
                linestyle="--",
                alpha=0.35,
                linewidth=1,
                label="Peak markers" if i == 0 else None,
            )

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.axhline(0, color="k", linewidth=1, alpha=0.5)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity (a.u.)")
    # plt.title(title)
    # plt.legend(fontsize="x-small")
    plt.tight_layout()
    plt.show()
    




def GetSubtypeAverageSpectraForOverlay(
    data_dict,
    TypestoPlot=("CT", "D7", "D14"),
    use_cropped=True,
    middle_fraction=None,
):
    """
    For current combined RF structure:
        data_dict[subtype] -> list[scan]

    Returns one averaged spectrum per subtype.
    If middle_fraction is None, averages all treated points in each scan.
    If middle_fraction is given, averages only the middle fraction of each scan.
    """

    key = "Spectra_Treated" if use_cropped else "Spectra_Treated_Full"

    def _point_sort_key(k):
        try:
            return int(str(k).replace("point", ""))
        except Exception:
            return 10**9

    def _align_stack(sc_list):
        sc_list = [sc for sc in sc_list if sc is not None]
        if not sc_list:
            return None, None

        x0 = np.asarray(sc_list[0].spectral_axis, float)
        Ys = []

        for sc in sc_list:
            x = np.asarray(sc.spectral_axis, float)
            y = np.asarray(sc.spectral_data, float)

            if (x.shape != x0.shape) or (not np.allclose(x, x0, rtol=0, atol=1e-9)):
                y = np.interp(x0, x, y)

            Ys.append(y)

        return x0, np.vstack(Ys)

    curves = {}

    for subtype in TypestoPlot:
        if subtype not in data_dict:
            print(f"[GetSubtypeAverageSpectraForOverlay] Subtype not found: {subtype}")
            continue

        sc_all = []

        for scan in data_dict[subtype]:
            treated = scan.get(key, {})
            if not treated:
                continue

            pkeys = sorted(treated.keys(), key=_point_sort_key)

            if middle_fraction is not None:
                n = len(pkeys)
                frac = float(np.clip(middle_fraction, 0.0, 1.0))
                k = max(1, int(round(n * frac)))
                start = (n - k) // 2
                pkeys = pkeys[start:start + k]

            sc_all.extend([treated[pk] for pk in pkeys])

        x, Y = _align_stack(sc_all)

        if x is None:
            print(f"[GetSubtypeAverageSpectraForOverlay] No treated spectra found for {subtype}")
            continue

        curves[subtype] = {
            "label": subtype,
            "x": x,
            "y": np.nanmean(Y, axis=0),
            "n_spectra": Y.shape[0],
        }

    return curves