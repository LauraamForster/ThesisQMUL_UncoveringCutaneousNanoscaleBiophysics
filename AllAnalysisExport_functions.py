#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 13:08:18 2026

@author: lauraforster
"""
from pathlib import Path
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import re
import ramanspy
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
    
Type = 'WOUND'
_CANONICAL_ORDER = ["despike", "smooth", "baseline", "normalise"]
# =================================================================     RAMAN     ================================================================

# =============================================================================
# Raman manifest functions
# =============================================================================
def RAMAN_read_Samplemanifest(manifest_path):
    sample_df = pd.read_excel(manifest_path)
    sample_df.columns = [str(c).strip() for c in sample_df.columns]
    return sample_df

def RAMAN_read_Peakmanifest(manifest_path, sheet_name="Paper"):
    manifest_path = Path(manifest_path)

    if manifest_path.suffix.lower() in [".xlsx", ".xls"]:
        peak_df = pd.read_excel(manifest_path, sheet_name=sheet_name)
    elif manifest_path.suffix.lower() == ".csv":
        peak_df = pd.read_csv(manifest_path)
    else:
        raise ValueError(f"Unsupported peak manifest format: {manifest_path.suffix}")

    peak_df.columns = [str(c).strip() for c in peak_df.columns]
    return peak_df

def RAMAN_BuildAssignmentsFromManifest(peak_manifest_df, component_colours=None):
    """
    Build peak assignment dictionaries from the peak manifest.

    Expected columns:
        Component
        Position

    Optional:
        Region
    """

    required = {"Component", "Position"}
    missing = required.difference(peak_manifest_df.columns)

    if missing:
        raise ValueError(f"Peak manifest missing required columns: {sorted(missing)}")

    df = peak_manifest_df.copy()
    df["Component"] = df["Component"].astype(str).str.strip()
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df = df.dropna(subset=["Component", "Position"])

    assignments = {
        comp: sorted(group["Position"].astype(float).unique().tolist())
        for comp, group in df.groupby("Component", sort=False)
    }

    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    if not default_cycle:
        default_cycle = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
        ]

    assign_colours = {}

    for i, comp in enumerate(assignments):
        assign_colours[comp] = (
            component_colours[comp]
            if component_colours and comp in component_colours
            else default_cycle[i % len(default_cycle)]
        )

    return assignments, assign_colours

def RAMAN_CreateDict(sample_manifest_df, Type, Subtypes):
    """
    Build base Raman data dictionary from the sample manifest.

    Expected manifest columns:
        Sample Number
        TYPE
        direction
    """

    required = {"Sample Number", "TYPE", "direction"}
    missing = required.difference(sample_manifest_df.columns)

    if missing:
        raise ValueError(f"Sample manifest missing required columns: {sorted(missing)}")

    data_dict = {}

    for _, row in sample_manifest_df.iterrows():
        subtype = str(row["TYPE"]).strip()

        if subtype not in Subtypes:
            continue

        sample_num = str(int(row["Sample Number"])).strip()
        
        data_dict[sample_num] = {
            "Type": Type,
            "Subtype": subtype,
            "direction": str(row["direction"]).strip().lower(),
        }

    return data_dict

# =============================================================================
# Raman read-in functions
# =============================================================================
def RAMAN_readindata(DataDir, data_dict):
    """
    Read Raman fingerprint and extended line scan text files into data_dict.

    Expected file structure:
        DataDir / subtype / sampleA_linescan1.txt
        DataDir / subtype / sampleA_linescan1_extended.txt
    """

    for sample_num, sample_data in data_dict.items():
        subtype = sample_data["Subtype"]
        base_path = DataDir / subtype / f"{sample_num}A"

        paths = {
            "FP": base_path.with_name(f"{sample_num}A_linescan1.txt"),
            "EXT": base_path.with_name(f"{sample_num}A_linescan1_extended.txt"),
        }

        for key, path in paths.items():
            try:
                df = pd.read_csv(
                    path,
                    sep=r"\s+",
                    comment="#",
                    header=None,
                    names=["X", "Y", "Wave", "Intensity"],
                )

                df[["X", "Y"]] = df[["X", "Y"]].ffill()

                for col in ["X", "Y", "Wave", "Intensity"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                df = df.dropna(subset=["X", "Y", "Wave", "Intensity"])

                data_dict[sample_num][key] = df

            except FileNotFoundError:
                print(f"{key} file not found for sample {sample_num}: {path}")

    return data_dict

def RAMAN_SplitSpectra(data_dict, Colours=None, Linestyles=None, SN=None, step=None):
    """
    Split Raman line scan dataframe into individual spectra grouped by X/Y coordinate.
    """

    for sample_num, sample_data in data_dict.items():
        for region in ["FP", "EXT"]:
            if region not in sample_data:
                continue

            df = sample_data[region]
            spectra = {}

            for (x, y), group in df.groupby(["X", "Y"], sort=False):
                spectra[(x, y)] = {
                    "Wave": group["Wave"].to_numpy(),
                    "Intensity": group["Intensity"].to_numpy(),
                }

            sample_data[f"{region}_Spectra"] = spectra

        sample_data.pop("FP", None)
        sample_data.pop("EXT", None)

        if SN is not None and str(SN) == sample_num:
            RAMAN_plot_split_spectra(
                sample_num=sample_num,
                sample_data=sample_data,
                Colours=Colours,
                Linestyles=Linestyles,
                step=step,
            )

    return data_dict

def RAMAN_plot_split_spectra(sample_num, sample_data, Colours=None, Linestyles=None, step=None):
    """
    Optional diagnostic plot for one sample.
    """

    subtype = sample_data.get("Subtype")
    colour = Colours.get(subtype, None) if Colours else None
    linestyle = Linestyles.get(subtype, None) if Linestyles else None

    for region in ["FP_Spectra", "EXT_Spectra"]:
        if region not in sample_data:
            continue

        plt.figure(figsize=(10, 6))

        for i, ((x, y), spectrum) in enumerate(sample_data[region].items()):
            if step is None or i % step == 0:
                label = f"x={np.round(x, 2)}, y={np.round(y, 2)}"
                plt.plot(
                    spectrum["Wave"],
                    spectrum["Intensity"],
                    label=label,
                    color=colour,
                    linestyle=linestyle,
                )

        plt.title(f"{region} spectra for Sample {sample_num}")
        plt.xlabel("Wavenumber (cm⁻¹)")
        plt.ylabel("Intensity")
        plt.legend(fontsize="x-small", loc="upper right", ncol=2)
        plt.tight_layout()
        plt.show()
           
# =============================================================================
# Raman checking functions
# =============================================================================
def RAMAN_count_linescan_points(data_dict):
    """
    Count total and dermis spectra per sample and spectral region.

    Returns
    -------
    df : pd.DataFrame
        Columns:
        Sample, Subtype, Region, N_total, N_dermis
    """

    rows = []

    for sample_num, sample_data in data_dict.items():
        if not isinstance(sample_data, dict):
            continue

        subtype = sample_data.get("Subtype")

        for reg in ["FP", "EXT"]:
            total_dict = (
                sample_data.get(f"{reg}_Spectra_Treated")
                or sample_data.get(f"{reg}_Spectra_Treated_Full")
                or sample_data.get(f"{reg}_Spectra")
            )

            dermis_dict = sample_data.get(f"{reg}_Spectra_Treated_Dermis")

            if total_dict is None and dermis_dict is None:
                continue

            rows.append({
                "Sample": str(sample_num),
                "Subtype": subtype,
                "Region": reg,
                "N_total": len(total_dict) if isinstance(total_dict, dict) else 0,
                "N_dermis": len(dermis_dict) if isinstance(dermis_dict, dict) else 0,
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            ["Subtype", "Sample", "Region"],
            ignore_index=True,
        )

    data_dict.setdefault("_counts", {})["per_region"] = df

    return df, data_dict

# =============================================================================
# Raman preprocessing
# =============================================================================
def RAMAN_normalize_steps(preprocess_list):
    """
    Return valid preprocessing steps in canonical order.
    """

    valid = {s.lower() for s in preprocess_list or []}
    unknown = valid.difference(_CANONICAL_ORDER)

    if unknown:
        print(f"Warning: ignoring unknown preprocessing steps: {sorted(unknown)}")

    return [s for s in _CANONICAL_ORDER if s in valid]

def RAMAN_build_region_steps(full_rng, crop_rng, selected_steps):
    """
    Build preprocessing callables for full-range and final cropped spectra.
    """

    cropper_full = ramanspy.preprocessing.misc.Cropper(region=full_rng)
    cropper_final = ramanspy.preprocessing.misc.Cropper(region=crop_rng)

    despike = ramanspy.preprocessing.despike.WhitakerHayes()
    savgol = ramanspy.preprocessing.denoise.SavGol(window_length=11, polyorder=3)
    baseline = ramanspy.preprocessing.baseline.ASLS(p=0.01, lam=1e4)
    auc = ramanspy.preprocessing.normalise.AUC(pixelwise=True)

    step_map = {
        "despike": lambda sc: despike.apply(sc),
        "smooth": lambda sc: savgol.apply(sc),
        "baseline": lambda sc: baseline.apply(sc),
        "normalise": lambda sc: auc.apply(sc),
    }

    step_fns_full = [lambda sc: cropper_full.apply(sc)]
    step_fns_full += [step_map[name] for name in selected_steps]

    step_fns_final = step_fns_full + [lambda sc: cropper_final.apply(sc)]

    return step_fns_full, step_fns_final

def RAMAN_normalise_sc(sc_in, mode="crop", band=None):
    """
    Manually normalise a Raman spectrum.

    mode:
        crop = divide by AUC over current window
        band = divide by AUC over chosen band
    """

    w = np.asarray(sc_in.spectral_axis, float)
    y = np.asarray(sc_in.spectral_data, float)

    if w.size < 2 or y.size < 2 or not np.all(np.isfinite(y)):
        return sc_in

    denom = np.nan

    if mode == "band" and band is not None:
        lo, hi = map(float, band)
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

def RAMAN_TreatSpectra(
    data_dict,
    Preprocess,
    FP_full,
    FP_crop,
    EXT_full,
    EXT_crop,
    treatmentorder="before",
    normalisation="crop",
    FP_band=None,
    EXT_band=None,
):
    """
    Preprocess Raman spectra and store:

        FP_Spectra_Treated_Full
        EXT_Spectra_Treated_Full
            processed full spectral range

        FP_Spectra_Treated
        EXT_Spectra_Treated
            processed, cropped, normalised analysis window

    No plots or PDFs are produced in the mega preprocessing script.
    """

    order_mode = str(treatmentorder).strip().lower()
    norm_mode = str(normalisation).strip().lower()

    if order_mode not in {"before", "after"}:
        print(f"[TreatSpectra] Unknown treatmentorder='{treatmentorder}', using 'before'.")
        order_mode = "before"

    if norm_mode not in {"crop", "band"}:
        print(f"[TreatSpectra] Unknown normalisation='{normalisation}', using 'crop'.")
        norm_mode = "crop"

    selected = RAMAN_normalize_steps(Preprocess)

    region_settings = [
        ("FP_Spectra", "FP", FP_full, FP_crop, FP_band),
        ("EXT_Spectra", "EXT", EXT_full, EXT_crop, EXT_band),
    ]

    for sample_num, sample_data in data_dict.items():
        treated_full = {"FP_Spectra": {}, "EXT_Spectra": {}}
        treated_crop = {"FP_Spectra": {}, "EXT_Spectra": {}}

        for region_key, region_label, full_rng, crop_rng, band_rng in region_settings:
            if region_key not in sample_data:
                continue

            step_fns_full, step_fns_final = RAMAN_build_region_steps(
                full_rng,
                crop_rng,
                selected,
            )

            crop_fn = step_fns_final[-1]

            items = sorted(
                sample_data[region_key].items(),
                key=lambda kv: (round(kv[0][0], 9), round(kv[0][1], 9)),
            )

            for idx, ((x, y), spec) in enumerate(items):
                wave = np.asarray(spec["Wave"], float)
                inten = np.asarray(spec["Intensity"], float)

                if wave.size and np.any(np.diff(wave) < 0):
                    order = np.argsort(wave)
                    wave = wave[order]
                    inten = inten[order]

                sc_raw = ramanspy.SpectralContainer(inten, wave)
                sc_raw.metadata = {"x": x, "y": y, "index": idx}

                if order_mode == "before":
                    sc_full = sc_raw

                    for fn in step_fns_full:
                        sc_full = fn(sc_full)

                    sc_final = crop_fn(sc_full)

                else:
                    sc_final = crop_fn(sc_raw)

                    for fn in step_fns_full:
                        sc_final = fn(sc_final)

                    sc_full = sc_raw

                    for fn in step_fns_full:
                        sc_full = fn(sc_full)

                sc_final_norm = RAMAN_normalise_sc(
                    sc_final,
                    mode=norm_mode,
                    band=band_rng,
                )

                treated_full[region_key][(x, y)] = sc_full
                treated_crop[region_key][(x, y)] = sc_final_norm

        for region_key in ["FP_Spectra", "EXT_Spectra"]:
            sample_data[f"{region_key}_Treated_Full"] = treated_full[region_key]
            sample_data[f"{region_key}_Treated"] = treated_crop[region_key]

    return data_dict

def RAMAN_TrimRegion(data_dict, sample_manifest_df):
    """
    Trim treated Raman spectra to the dermis region using manifest coordinates.

    Stores:
        FP_Spectra_Treated_Dermis
        EXT_Spectra_Treated_Dermis
    """

    sample_manifest_df = sample_manifest_df.copy()
    sample_manifest_df["Sample Number"] = sample_manifest_df["Sample Number"].astype(str)

    for sample_num, sample_data in data_dict.items():
        manifest_row = sample_manifest_df[
            sample_manifest_df["Sample Number"] == str(sample_num)
        ]

        if manifest_row.empty:
            print(f"Sample {sample_num} not found in Raman manifest")
            continue

        row = manifest_row.iloc[0]

        dermis_start = np.array([row["dermis x"], row["dermis y"]], dtype=float)
        dermis_end = np.array([row["epi x"], row["epi y"]], dtype=float)

        dermis_length_reported = row.get("leng dermis", np.nan)
        direction = str(row["direction"]).strip().lower()

        for region_key in ["FP_Spectra_Treated", "EXT_Spectra_Treated"]:
            if region_key not in sample_data:
                continue

            spectra_dict = sample_data[region_key]

            if not spectra_dict:
                continue

            coords = np.array(list(spectra_dict.keys()), dtype=float)

            idx_start = np.argmin(np.linalg.norm(coords - dermis_start, axis=1))
            idx_end = np.argmin(np.linalg.norm(coords - dermis_end, axis=1))

            i1, i2 = sorted([idx_start, idx_end])
            trimmed_keys = [tuple(coords[i]) for i in range(i1, i2 + 1)]

            if direction == "back":
                trimmed_keys = list(reversed(trimmed_keys))

            trimmed_spectra = {
                key: spectra_dict[key]
                for key in trimmed_keys
                if key in spectra_dict
            }

            sample_data[f"{region_key}_Dermis"] = trimmed_spectra

            if np.isfinite(dermis_length_reported):
                diff = abs(len(trimmed_spectra) - int(dermis_length_reported))

                if diff > 2:
                    print(
                        f"Warning: dermis length mismatch for sample {sample_num} "
                        f"{region_key}: manifest={int(dermis_length_reported)}, "
                        f"trimmed={len(trimmed_spectra)}"
                    )

    return data_dict

def RAMAN_BinData(data_dict, NBins):
    """
    Bin trimmed dermis Raman spectra for each sample.

    Stores per-sample:
        FP_Bins
        EXT_Bins

    Returns:
        data_dict
        binned_avg[subtype][FP/EXT][bin]
    """

    subtype_samples = {}

    for sample_num, sample_data in data_dict.items():
        subtype = sample_data.get("Subtype")

        if subtype:
            subtype_samples.setdefault(subtype, []).append(sample_num)

    binned_avg = {}

    for subtype, samples in subtype_samples.items():
        binned_avg[subtype] = {"FP": [], "EXT": []}

        for region_key in ["FP_Spectra_Treated_Dermis", "EXT_Spectra_Treated_Dermis"]:
            region_short = region_key.split("_")[0]
            samples_binned = []

            for sample_num in samples:
                sample_data = data_dict[sample_num]

                if region_key not in sample_data:
                    continue

                spectra_dict = sample_data[region_key]
                keys_order = list(spectra_dict.keys())

                if not keys_order:
                    continue

                n_points = len(keys_order)
                n_eff = min(NBins, n_points)
                index_bins = np.array_split(np.arange(n_points), n_eff)

                ref_axis = np.asarray(
                    spectra_dict[keys_order[0]].spectral_axis,
                    dtype=float,
                )

                bin_averages = []

                for idxs in index_bins:
                    bin_intensities = []

                    for idx in idxs:
                        sc = spectra_dict[keys_order[idx]]
                        wav = np.asarray(sc.spectral_axis, dtype=float)
                        inten = np.asarray(sc.spectral_data, dtype=float)

                        if wav.shape != ref_axis.shape or not np.allclose(
                            wav,
                            ref_axis,
                            rtol=0,
                            atol=1e-9,
                        ):
                            inten = np.interp(ref_axis, wav, inten)

                        bin_intensities.append(inten)

                    avg_intensity = np.nanmean(
                        np.vstack(bin_intensities),
                        axis=0,
                    )

                    bin_averages.append({
                        "Wave": ref_axis,
                        "Intensity": avg_intensity,
                    })

                samples_binned.append(bin_averages)
                sample_data[f"{region_short}_Bins"] = bin_averages

            if not samples_binned:
                continue

            n_bins_final = min(len(bins) for bins in samples_binned)
            subtype_bins = []

            for b in range(n_bins_final):
                bin_stack = [
                    sample_bins[b]["Intensity"]
                    for sample_bins in samples_binned
                    if sample_bins[b] is not None
                ]

                if not bin_stack:
                    continue

                subtype_bins.append({
                    "Wave": samples_binned[0][b]["Wave"],
                    "Intensity": np.nanmean(np.vstack(bin_stack), axis=0),
                })

            binned_avg[subtype][region_short] = subtype_bins

    return data_dict, binned_avg

def RAMAN_WeightedMoments_ByRegion(
    data_dict,
    TypestoPlot,
    region="dermis",
    peak_regions=None,
    use_FP=True,
    use_treated=True,
):
    """
    Calculate weighted moments from the sample-averaged Raman spectrum.

    Returns
    -------
    df_avg : pd.DataFrame
        One row per sample with Raman metrics.
    """

    if peak_regions is None:
        peak_regions = [
            ("AmideI_1550_1750", (1550, 1750)),
            ("AmideIII_1400_1550", (1400, 1550)),
            ("CH2CH3_1200_1400", (1200, 1400)),
        ]

    typelist = {
        str(t).strip()
        for t in (
            TypestoPlot
            if isinstance(TypestoPlot, (list, tuple, set))
            else [TypestoPlot]
        )
    }

    reg_short = "FP" if use_FP else "EXT"
    treated_tag = "Treated" if use_treated else "Treated_Full"
    region_key = f"{reg_short}_Spectra_{treated_tag}_{region.capitalize()}"

    rows = []

    for sample_num, sample_data in data_dict.items():
        subtype = str(sample_data.get("Subtype", "")).strip()

        if subtype not in typelist:
            continue

        spectra_dict = sample_data.get(region_key)

        if not isinstance(spectra_dict, dict) or not spectra_dict:
            print(f"Missing {region_key} for sample {sample_num} ({subtype})")
            continue

        wave, stack = RAMAN_align_raman_stack(spectra_dict)

        if stack.size == 0:
            continue

        mean_spectrum = np.nanmean(stack, axis=0)

        row = {
            "Sample": str(sample_num),
            "Subtype": subtype,
            "Technique": "Raman",
            "Region": region,
            "RegionKey": region_key,
            "NpointsRegion": int(stack.shape[0]),
        }

        for peak_name, xlim in peak_regions:
            moments = RAMAN_weighted_moments_from_mean_spectrum(
                wave,
                mean_spectrum,
                xlim,
            )

            for metric, value in moments.items():
                row[f"{peak_name}_{metric}"] = value

        rows.append(row)

    return pd.DataFrame(rows)

def RAMAN_align_raman_stack(spectra_dict):
    """
    Align Raman spectra onto a shared spectral axis.
    """

    keys = list(spectra_dict.keys())
    sc0 = spectra_dict[keys[0]]
    x0 = np.asarray(sc0.spectral_axis, float)

    stack = []

    for key in keys:
        sc = spectra_dict[key]
        x = np.asarray(sc.spectral_axis, float)
        y = np.asarray(sc.spectral_data, float)

        if x.shape != x0.shape or not np.allclose(x, x0, rtol=0, atol=1e-9):
            y = np.interp(x0, x, y)

        stack.append(y)

    return x0, np.vstack(stack)

def RAMAN_weighted_moments_from_mean_spectrum(x, y, xlim):
    """
    Weighted moments from one averaged spectrum within a chosen wavenumber range.
    """

    x = np.asarray(x, float)
    y = np.asarray(y, float)

    lo, hi = map(float, xlim)
    mask = (x >= lo) & (x <= hi) & np.isfinite(x) & np.isfinite(y)

    if not np.any(mask):
        return {
            "m1": np.nan,
            "mu2": np.nan,
            "mu3": np.nan,
            "sigma": np.nan,
            "skewness": np.nan,
            "area_w": np.nan,
            "max_intensity": np.nan,
            "neg_area_frac": np.nan,
            "n_points": 0,
        }

    xr = x[mask]
    yr = y[mask]

    area_total = np.trapezoid(np.abs(yr), xr) if xr.size > 1 else np.nan
    area_neg = np.trapezoid(np.clip(-yr, 0, None), xr) if xr.size > 1 else np.nan

    neg_area_frac = (
        area_neg / area_total
        if np.isfinite(area_total) and area_total > 0
        else np.nan
    )

    offset = -min(0.0, float(np.nanmin(yr)))
    weights = np.clip(yr + offset, 0, None)
    wsum = float(np.nansum(weights))

    max_intensity = float(np.nanmax(yr)) if xr.size else np.nan

    if not (np.isfinite(wsum) and wsum > 0):
        return {
            "m1": np.nan,
            "mu2": np.nan,
            "mu3": np.nan,
            "sigma": np.nan,
            "skewness": np.nan,
            "area_w": np.nan,
            "max_intensity": max_intensity,
            "neg_area_frac": neg_area_frac,
            "n_points": int(xr.size),
        }

    m1 = float(np.nansum(weights * xr) / wsum)
    mu2 = float(np.nansum(weights * (xr - m1) ** 2) / wsum)
    mu3 = float(np.nansum(weights * (xr - m1) ** 3) / wsum)

    sigma = float(np.sqrt(mu2)) if np.isfinite(mu2) and mu2 >= 0 else np.nan
    skewness = (
        float(mu3 / (mu2 ** 1.5))
        if np.isfinite(mu2) and mu2 > 0 and np.isfinite(mu3)
        else np.nan
    )

    area_w = np.trapezoid(weights, xr) if xr.size > 1 else np.nan

    return {
        "m1": m1,
        "mu2": mu2,
        "mu3": mu3,
        "sigma": sigma,
        "skewness": skewness,
        "area_w": area_w,
        "max_intensity": max_intensity,
        "neg_area_frac": neg_area_frac,
        "n_points": int(xr.size),
    }



# ===========================================================     NANOINDENTATION     =============================================================

# =============================================================================
# Nanoindentation read-in
# =============================================================================

def NI_ReadManifest(manifest_path, base_path, regions, ST):
    """
    Read nanoindentation manifest and associated processed CSV files.

    Returns
    -------
    data_dict : dict
        Nested dictionary:
            sample folder → metadata + region dictionaries
    """

    col_map = {
        "Bleo": {
            "lowerdermis": "lower dermis",
            "upperdermis": "upper dermis",
            "linescan": "line scan",
        },
        "Bleomycin": {
            "lowerdermis": "lower dermis",
            "upperdermis": "upper dermis",
            "linescan": "line scan",
        },
        "AP1": {
            "linescan": "linescan",
            "horiz_linescan": "horiz_linescan",
        },
        "wounding": {
            "linescan": "linescan",
            "horiz_linescan": "horiz_linescan",
        },
    }

    if ST not in col_map:
        raise ValueError(f"Unknown set_type {ST!r}")

    region_cols = col_map[ST]

    df = pd.read_csv(manifest_path, dtype=str).fillna("yes")
    data_dict = {}

    for _, row in df.iterrows():
        folder = row["FOLDER NAME"]

        sample = {
            "SAMPLE NAME": row["SAMPLE NAME"],
            "TYPE": row["TYPE"],
            "FOLDER NAME": folder,
            "Sample Number": row["Sample Number"],
        }

        for canon, col in region_cols.items():
            sample[canon] = row.get(col, "yes").strip().lower()

        for c in row.index:
            if c not in sample and not c.startswith("Unnamed:"):
                sample[c] = row[c]

        data_dict[folder] = sample

    for folder, sample in data_dict.items():
        sample_type = sample["TYPE"]

        for canon_reg in regions:
            NI_initialise_ni_region(sample, canon_reg)

            if str(sample.get(canon_reg, "yes")).lower() == "no":
                continue

            csv_path = NI_build_ni_csv_path(
                base_path=base_path,
                set_type=ST,
                sample_type=sample_type,
                folder=folder,
                canon_reg=canon_reg,
            )

            if not os.path.exists(csv_path):
                print(f"Missing NI CSV: {csv_path}")
                continue

            try:
                df_raw = pd.read_csv(csv_path)
            except Exception as exc:
                print(f"Could not read NI CSV: {csv_path} | {exc}")
                continue

            region_dict = NI_build_ni_region_dict(df_raw)
            region_dict = NI_screen_ni_region_dict(region_dict)

            if region_dict:
                NI_store_ni_region(sample, canon_reg, region_dict)

    return data_dict

def NI_initialise_ni_region(sample, canon_reg):
    """
    Initialise region key in sample dictionary.
    """

    target = {
        "lowerdermis": "lower dermis",
        "upperdermis": "upper dermis",
        "linescan": "line scan",
        "horiz_linescan": "horiz_linescan",
    }.get(canon_reg)

    if target:
        sample[target] = {}

def NI_store_ni_region(sample, canon_reg, region_dict):
    """
    Store loaded region dictionary under standard readable region key.
    """

    target = {
        "lowerdermis": "lower dermis",
        "upperdermis": "upper dermis",
        "linescan": "line scan",
        "horiz_linescan": "horiz_linescan",
    }.get(canon_reg)

    if target:
        sample[target] = region_dict

def NI_build_ni_csv_path(base_path, set_type, sample_type, folder, canon_reg):
    """
    Build expected processed nanoindentation CSV path.
    """

    st_lower = set_type.lower()

    if st_lower in {"bleo", "bleomycin"}:
        subfolder = {
            "lowerdermis": "lowerdermis",
            "upperdermis": "upperdermis",
            "linescan": "linescan",
        }[canon_reg]

        folder_path = os.path.join(base_path, sample_type, subfolder)

    elif st_lower == "ap1":
        orient = "horiz" if canon_reg == "horiz_linescan" else "vert"
        group = "AP1" if sample_type in ["TS", "VH"] else "CL"
        folder_path = os.path.join(base_path, f"{group} {orient}", sample_type)

    elif st_lower == "wounding":
        folder_path = os.path.join(base_path, set_type, sample_type)

    else:
        raise ValueError(f"Unhandled nanoindentation set_type: {set_type}")

    return os.path.join(folder_path, f"OutputExcel_{folder}_linescan.csv")

def NI_build_ni_region_dict(df_raw):
    """
    Convert one processed nanoindentation CSV to a coordinate-keyed dictionary.
    """

    region_dict = {}

    for _, r in df_raw.iterrows():
        try:
            x = r.get("x", np.nan)
            y = r.get("y", np.nan)
            xy = f"{x}_{y}"

            region_dict[xy] = {
                "File Name": r.get("File Name", ""),
                "x": x,
                "y": y,

                "Hertz - Modulus(Pa) fit": r.get("Hertz - Modulus(Pa) fit", np.nan),
                "Hertz - Rsq": r.get("Hertz - Rsq", np.nan),

                "OP - Modulus": r.get("OP - Modulus", np.nan),
                "OP - Rsq": r.get("OP - Rsq", np.nan),

                "Holding - Time Held (s)": r.get("Holding - Time Held (s)", np.nan),
                "Hold - Load Start": r.get("Hold - Load Start", np.nan),
                "Hold - Load End": r.get("Hold - Load End", np.nan),
                "Hold - Relaxation Fraction": r.get("Hold - Relaxation Fraction", np.nan),

                "Visco (Analytic) - tau (s)": r.get("Visco (Analytic) - tau (s)", np.nan),
                "Visco (Analytic) - G0 (Pa)": r.get("Visco (Analytic) - G0 (Pa)", np.nan),
                "Visco (Analytic) - G1 (Pa)": r.get("Visco (Analytic) - G1 (Pa)", np.nan),
                "Visco (Analytic) - E0 (Pa)": r.get("Visco (Analytic) - E0 (Pa)", np.nan),
                "Visco (Analytic) - E_inf (Pa)": r.get("Visco (Analytic) - E_inf (Pa)", np.nan),

                "ViscoAna_r2": r.get("ViscoAna_r2", np.nan),
            }

        except Exception:
            continue

    return region_dict

def NI_screen_ni_region_dict(region_dict):
    """
    Apply field-specific screening to one NI region dictionary.
    """

    if not region_dict:
        return region_dict

    keys = list(region_dict.keys())

    def _screen_field(field, rsq_field=None):
        vals = np.array(
            [region_dict[k].get(field, np.nan) for k in keys],
            dtype=float,
        )

        rsq = None

        if rsq_field is not None:
            rsq = np.array(
                [region_dict[k].get(rsq_field, np.nan) for k in keys],
                dtype=float,
            )

        vals_f = NI_ScreenData(vals, rsq_values=rsq, field=field)

        for key, val in zip(keys, vals_f):
            region_dict[key][field] = val

    _screen_field("Hertz - Modulus(Pa) fit", "Hertz - Rsq")
    _screen_field("OP - Modulus", "OP - Rsq")

    _screen_field("Holding - Time Held (s)")
    _screen_field("Hold - Load Start")
    _screen_field("Hold - Load End")
    _screen_field("Hold - Relaxation Fraction")

    for field in [
        "Visco (Analytic) - tau (s)",
        "Visco (Analytic) - G0 (Pa)",
        "Visco (Analytic) - G1 (Pa)",
        "Visco (Analytic) - E0 (Pa)",
        "Visco (Analytic) - E_inf (Pa)",
    ]:
        _screen_field(field, "ViscoAna_r2")

    return region_dict

# =============================================================================
# Nanoindentation screening
# =============================================================================
def NI_ScreenData(y, rsq_values=None, field=None, rsq_min=0.3, hi=1_000_000):
    """
    Field-aware screening for nanoindentation outputs.

    Rules:
        - Hertz/OP modulus: R² gate, remove negatives, remove > hi, 3σ outliers
        - Visco fit params: R² gate, remove negatives, 3σ outliers
        - RelaxFrac: remove negatives, 3σ outliers
        - Holding values: 3σ outliers only
    """

    y = np.asarray(y, dtype=np.float64)
    field = "" if field is None else str(field)

    is_hertz_mod = field == "Hertz - Modulus(Pa) fit"
    is_op_mod = field == "OP - Modulus"

    is_visco_fit = field in {
        "Visco (Analytic) - tau (s)",
        "Visco (Analytic) - G0 (Pa)",
        "Visco (Analytic) - G1 (Pa)",
        "Visco (Analytic) - E0 (Pa)",
        "Visco (Analytic) - E_inf (Pa)",
    }

    is_relaxfrac = field == "Hold - Relaxation Fraction"

    if (is_hertz_mod or is_op_mod or is_visco_fit) and rsq_values is not None:
        rsq_values = np.asarray(rsq_values, dtype=np.float64)
        y = np.where(rsq_values < rsq_min, np.nan, y)

    if is_hertz_mod or is_op_mod or is_visco_fit or is_relaxfrac:
        y = np.where(y < 0, np.nan, y)

    if is_hertz_mod or is_op_mod:
        y = np.where(y > hi, np.nan, y)

    finite = np.isfinite(y)

    if finite.sum() < 2:
        return y

    mu = float(np.nanmean(y))
    sd = float(np.nanstd(y, ddof=1))

    if not np.isfinite(sd) or sd == 0:
        return y

    lo, up = mu - 3 * sd, mu + 3 * sd

    return np.where((y < lo) | (y > up), np.nan, y)

# =============================================================================
# Nanoindentation layer/region assignment
# =============================================================================
def NI_CutSampleLengths(data_dict):
    """
    Assign each NI point to a layer/region using measured lengths in the manifest.

    Vertical line scan layers:
        left glass, subcut, dermis, epidermis, right glass

    Horizontal line scan regions:
        left, centre, right
    """

    for sample_key, sample in data_dict.items():
        NI_assign_vertical_layers(sample_key, sample)
        NI_assign_horizontal_regions(sample_key, sample)

    return data_dict

def NI_assign_vertical_layers(sample_key, sample):
    """
    Assign line scan points to vertical anatomical layers.
    """

    line_scan = sample.get("line scan", {})

    if not isinstance(line_scan, dict) or not line_scan:
        return

    try:
        total_pts = int(sample.get("no pts", np.nan))

        lengths = {
            "left glass": int(sample.get("Measured left glass", 0)),
            "subcut": int(sample.get("Measured Subcut layer", 0)),
            "dermis": int(sample.get("Measured dermis", 0)),
            "epidermis": int(sample.get("Measured epi", 0)),
            "right glass": int(sample.get("Measured right glass", 0)),
        }

    except (TypeError, ValueError):
        print(f"Skipping {sample_key}: invalid vertical region lengths")
        return

    if all(v == 0 for v in lengths.values()):
        print(f"Sample {sample_key}: no vertical lengths; assigning all points to dermis.")

        for pt in line_scan.values():
            pt["layer"] = "dermis"

        return

    if sum(lengths.values()) != total_pts:
        print(
            f"Warning: {sample_key}: vertical lengths do not match total points "
            f"({sum(lengths.values())} != {total_pts})"
        )
        return

    sorted_keys = sorted(
        line_scan.keys(),
        key=lambda k: (float(line_scan[k]["y"]), float(line_scan[k]["x"])),
    )

    current_idx = 0

    for layer, length in lengths.items():
        for i in range(current_idx, current_idx + length):
            if i < len(sorted_keys):
                line_scan[sorted_keys[i]]["layer"] = layer

        current_idx += length

def NI_assign_horizontal_regions(sample_key, sample):
    """
    Assign horizontal line scan points to left/centre/right regions.
    """

    horiz_scan = sample.get("horiz_linescan", {})

    if not isinstance(horiz_scan, dict) or not horiz_scan:
        return

    try:
        total_pts = int(sample.get("no pts horiz", np.nan))

        lengths = {
            "left": int(sample.get("Measured left ", 0)),
            "centre": int(sample.get("measured centre", 0)),
            "right": int(sample.get("Measured right", 0)),
        }

    except (TypeError, ValueError):
        print(f"Skipping {sample_key}: invalid horizontal region lengths")
        return

    if all(v == 0 for v in lengths.values()):
        print(f"Sample {sample_key}: no horizontal lengths; skipping horizontal region assignment.")
        return

    if sum(lengths.values()) != total_pts:
        print(
            f"Warning: {sample_key}: horizontal lengths do not match total points "
            f"({sum(lengths.values())} != {total_pts})"
        )
        return

    sorted_keys = sorted(
        horiz_scan.keys(),
        key=lambda k: (float(horiz_scan[k]["x"]), float(horiz_scan[k]["y"])),
    )

    current_idx = 0

    for region, length in lengths.items():
        for i in range(current_idx, current_idx + length):
            if i < len(sorted_keys):
                horiz_scan[sorted_keys[i]]["region"] = region

        current_idx += length

# =============================================================================
# Nanoindentation dataframe conversion and binning
# =============================================================================
def NI_PointsToDataFrame(data_dict, VAR_MAP, region_key="line scan"):
    """
    Flatten nanoindentation dictionary into one long dataframe.

    Returns
    -------
    df_points : pd.DataFrame
        One row per indentation point.
    """

    rows = []

    for sample_key, sample in data_dict.items():
        region_dict = sample.get(region_key, {})

        if not isinstance(region_dict, dict) or not region_dict:
            continue

        for point_index, (point_key, point) in enumerate(region_dict.items()):
            row = {
                "Sample": str(sample.get("Sample Number", sample_key)),
                "SampleKey": sample_key,
                "SampleName": sample.get("SAMPLE NAME", ""),
                "Subtype": str(sample.get("TYPE", "")).strip(),
                "Technique": "Nanoindentation",
                "RegionKey": region_key,
                "PointKey": point_key,
                "PointIndex": point_index,
                "x": point.get("x", np.nan),
                "y": point.get("y", np.nan),
                "layer": point.get("layer", np.nan),
                "region": point.get("region", np.nan),
                "File Name": point.get("File Name", ""),
            }

            for var, source_col in VAR_MAP.items():
                row[var] = point.get(source_col, np.nan)

            rows.append(row)

    df_points = pd.DataFrame(rows)

    if not df_points.empty:
        for col in ["x", "y", *VAR_MAP.keys()]:
            df_points[col] = pd.to_numeric(df_points[col], errors="coerce")

    return df_points

def NI_PrepareBinnedData(data_dict, VAR_MAP, nbins=5, layer="dermis"):
    """
    Bin nanoindentation line scan data by sample and selected layer.

    Returns
    -------
    binned_dict : dict
        Nested dictionary for compatibility with older plotting logic.

    df_points_layer : pd.DataFrame
        Point-level dataframe filtered to selected layer.

    df_binned : pd.DataFrame
        One row per sample/bin with mean, std, count for each variable.
    """

    df_points = NI_PointsToDataFrame(data_dict, VAR_MAP, region_key="line scan")

    if df_points.empty:
        return {}, df_points, pd.DataFrame()

    df_points_layer = df_points[df_points["layer"].astype(str).str.lower() == layer.lower()].copy()

    if df_points_layer.empty:
        print(f"No nanoindentation points found for layer={layer!r}")
        return {}, df_points_layer, pd.DataFrame()

    binned_rows = []
    binned_dict = {}

    for sample_key, df_sample in df_points_layer.groupby("SampleKey", sort=False):
        df_sample = df_sample.sort_values(["y", "x"]).reset_index(drop=True)

        n_points = len(df_sample)

        if n_points == 0:
            continue

        n_eff = min(nbins, n_points)
        index_bins = np.array_split(np.arange(n_points), n_eff)

        sample_type = df_sample["Subtype"].iloc[0]
        sample_num = df_sample["Sample"].iloc[0]

        binned_dict.setdefault(sample_type, {})
        binned_dict[sample_type].setdefault(sample_key, [])

        for bin_idx, idxs in enumerate(index_bins, start=1):
            df_bin = df_sample.iloc[idxs]

            row = {
                "Sample": sample_num,
                "SampleKey": sample_key,
                "Subtype": sample_type,
                "Technique": "Nanoindentation",
                "Layer": layer,
                "Bin": bin_idx,
                "BinTotal": n_eff,
                "N_points": int(len(df_bin)),
                "x_mean": float(np.nanmean(df_bin["x"])),
                "y_mean": float(np.nanmean(df_bin["y"])),
            }

            bin_entry = {
                "Sample": sample_num,
                "SampleKey": sample_key,
                "Subtype": sample_type,
                "Layer": layer,
                "Bin": bin_idx,
                "N_points": int(len(df_bin)),
            }

            for var in VAR_MAP:
                vals = pd.to_numeric(df_bin[var], errors="coerce")

                row[f"{var}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
                row[f"{var}_std"] = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else np.nan
                row[f"{var}_n"] = int(np.isfinite(vals).sum())

                bin_entry[var] = row[f"{var}_mean"]

            binned_rows.append(row)
            binned_dict[sample_type][sample_key].append(bin_entry)

    df_binned = pd.DataFrame(binned_rows)

    return binned_dict, df_points_layer, df_binned

def NI_SampleSummary(df_points_layer, VAR_MAP):
    """
    Create one row per sample from selected-layer nanoindentation points.

    Returns
    -------
    df_summary : pd.DataFrame
    """

    if df_points_layer.empty:
        return pd.DataFrame()

    rows = []

    for (sample, sample_key, subtype), df_sample in df_points_layer.groupby(
        ["Sample", "SampleKey", "Subtype"],
        dropna=False,
    ):
        row = {
            "Sample": sample,
            "SampleKey": sample_key,
            "Subtype": subtype,
            "Technique": "Nanoindentation",
            "Layer": df_sample["layer"].dropna().iloc[0] if df_sample["layer"].notna().any() else "",
            "N_points": int(len(df_sample)),
        }

        for var in VAR_MAP:
            vals = pd.to_numeric(df_sample[var], errors="coerce")

            row[f"{var}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            row[f"{var}_std"] = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else np.nan
            row[f"{var}_n"] = int(np.isfinite(vals).sum())

        rows.append(row)

    return pd.DataFrame(rows)


# ===========================================================     SAXS     =============================================================

# =============================================================================
# SAXS initialisation helpers
# =============================================================================
def SAXS_normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [" ".join(str(c).strip().split()) for c in df.columns]
    return df

def SAXS_pick_col(df: pd.DataFrame, *candidates: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError(
        f"None of these columns found: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )

def SAXS_is_yes(v) -> bool:
    if pd.isna(v):
        return False

    return str(v).strip().upper() in {"Y", "YES", "TRUE", "1"}

def SAXS_clean_str(v) -> str:
    if pd.isna(v):
        return ""

    return str(v).strip()

def SAXS_load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    df = pd.read_csv(path)

    if df is None or df.empty:
        return None

    return SAXS_normalise_columns(df)

def SAXS_infer_grid_shape(df: pd.DataFrame) -> tuple[int, int]:
    x = pd.to_numeric(df["x"], errors="coerce").astype("Int64").dropna().astype(int)
    y = pd.to_numeric(df["y"], errors="coerce").astype("Int64").dropna().astype(int)

    if x.empty or y.empty:
        return 0, 0

    return int(x.max()) + 1, int(y.max()) + 1

# =============================================================================
# SAXS ROI/mask helpers
# =============================================================================
def SAXS_roi_points_from_row(row: pd.Series, region: str, npts: int) -> list[tuple[float, float]]:
    pts = []

    for i in range(1, npts + 1):
        x = row.get(f"{region}_p{i}_x", np.nan)
        y = row.get(f"{region}_p{i}_y", np.nan)

        if pd.notna(x) and pd.notna(y):
            pts.append((float(x), float(y)))

    return pts

def SAXS_roi_points_auto(row: pd.Series, region: str) -> list[tuple[float, float]]:
    pts = []
    i = 1

    while True:
        kx = f"{region}_p{i}_x"
        ky = f"{region}_p{i}_y"

        if kx not in row.index or ky not in row.index:
            break

        x = row.get(kx, np.nan)
        y = row.get(ky, np.nan)

        if pd.notna(x) and pd.notna(y):
            pts.append((float(x), float(y)))

        i += 1

    return pts

def SAXS_poly_mask(points: list[tuple[float, float]], nx: int, ny: int) -> np.ndarray:
    if points is None or len(points) < 3 or nx <= 0 or ny <= 0:
        return np.zeros((ny, nx), dtype=bool)

    poly = MplPath(np.asarray(points, float))

    yy, xx = np.mgrid[0:ny, 0:nx]
    centres = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])

    inside = poly.contains_points(centres)

    return inside.reshape(ny, nx)

def SAXS_parse_coord_list(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.array([], dtype=float)

    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        return (
            pd.to_numeric(pd.Series(list(v)), errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )

    s = str(v).strip()

    if not s:
        return np.array([], dtype=float)

    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)

    if not nums:
        return np.array([], dtype=float)

    return np.asarray([float(x) for x in nums], dtype=float)

def SAXS_parse_xy_points(xv, yv):
    x = SAXS_parse_coord_list(xv)
    y = SAXS_parse_coord_list(yv)

    n = min(len(x), len(y))

    if n == 0:
        return np.empty((0, 2), dtype=float)

    return np.column_stack([x[:n], y[:n]])

def SAXS_fit_line_direction(points: np.ndarray):
    pts = np.asarray(points, dtype=float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]

    if len(pts) == 0:
        return np.array([np.nan, np.nan]), np.array([np.nan, np.nan])

    centroid = pts.mean(axis=0)

    if len(pts) == 1:
        return centroid, np.array([1.0, 0.0])

    centred = pts - centroid
    _, _, vh = np.linalg.svd(centred, full_matrices=False)

    tangent = vh[0]
    tangent = tangent / np.linalg.norm(tangent)

    return centroid, tangent

def SAXS_split_mask_along_epidermis_subcut_axis(
    mask: np.ndarray,
    epidermis_pts: np.ndarray,
    subcut_pts: np.ndarray,
):
    if mask is None or not np.any(mask):
        return mask.copy(), np.zeros_like(mask, dtype=bool)

    epi_c, epi_t = SAXS_fit_line_direction(epidermis_pts)
    sub_c, sub_t = SAXS_fit_line_direction(subcut_pts)

    if not np.all(np.isfinite(epi_c)) or not np.all(np.isfinite(sub_c)):
        return mask.copy(), np.zeros_like(mask, dtype=bool)

    tangent = epi_t if np.all(np.isfinite(epi_t)) else sub_t

    if not np.all(np.isfinite(tangent)):
        return mask.copy(), np.zeros_like(mask, dtype=bool)

    tangent = tangent / np.linalg.norm(tangent)

    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    normal = normal / np.linalg.norm(normal)

    epi_to_sub = sub_c - epi_c

    if np.dot(epi_to_sub, normal) < 0:
        normal = -normal

    yy, xx = np.indices(mask.shape)
    proj = xx * normal[0] + yy * normal[1]

    proj_in = proj[mask]

    if proj_in.size == 0:
        return mask.copy(), np.zeros_like(mask, dtype=bool)

    thresh = 0.5 * (proj_in.min() + proj_in.max())

    epi_half = mask & (proj <= thresh)
    sub_half = mask & (proj > thresh)

    return epi_half, sub_half

def SAXS_add_split_masks_from_orientation(
    masks: dict,
    roi_row,
    split_regions=("dermis", "wound"),
):
    if roi_row is None or masks is None:
        return masks

    epidermis_pts = SAXS_parse_xy_points(
        roi_row.get("epidermis_x", None),
        roi_row.get("epidermis_y", None),
    )

    subcut_pts = SAXS_parse_xy_points(
        roi_row.get("subcutaneous_x", None),
        roi_row.get("subcutaneous_y", None),
    )

    out = dict(masks)

    for region in split_regions:
        if region not in out or out[region] is None:
            continue

        epi_half, sub_half = SAXS_split_mask_along_epidermis_subcut_axis(
            out[region],
            epidermis_pts,
            subcut_pts,
        )

        out[f"{region}_epi"] = epi_half
        out[f"{region}_sub"] = sub_half

    return out

def SAXS_build_region_masks_for_sample(
    roi_row: pd.Series | None,
    *,
    nx: int,
    ny: int,
    roi_specs: dict[str, int],
    valid_regions: list[str],
    sample_region: str = "sample",
) -> dict[str, np.ndarray]:
    blank = np.zeros((ny, nx), dtype=bool)
    masks = {region: blank.copy() for region in valid_regions}

    if roi_row is None or nx <= 0 or ny <= 0:
        return masks

    sample_type = SAXS_clean_str(roi_row.get("sample_type", "")).lower()

    for region in valid_regions:
        if region in roi_specs:
            pts = SAXS_roi_points_from_row(roi_row, region, roi_specs[region])
        else:
            pts = SAXS_roi_points_auto(roi_row, region)

        if sample_type == "control" and region == "wound":
            masks[region] = blank.copy()
        else:
            masks[region] = SAXS_poly_mask(pts, nx=nx, ny=ny) if pts else blank.copy()

    if sample_region in valid_regions and not masks[sample_region].any():
        other_masks = [m for region, m in masks.items() if region != sample_region]

        if other_masks:
            masks[sample_region] = np.logical_or.reduce(other_masks)

    return masks

def SAXS_label_points_by_regions(
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    *,
    include_regions: list[str],
) -> pd.DataFrame:
    """
    Return long-form point table:
        one row per point-region membership.

    Overlap is allowed.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    if not {"x", "y"}.issubset(df.columns):
        return pd.DataFrame()

    if not masks:
        return pd.DataFrame()

    ny, nx = next(iter(masks.values())).shape

    x_raw = pd.to_numeric(df["x"], errors="coerce")
    y_raw = pd.to_numeric(df["y"], errors="coerce")

    ok = x_raw.notna() & y_raw.notna()

    df2 = df.loc[ok].copy()
    x = x_raw.loc[ok].astype(int).to_numpy()
    y = y_raw.loc[ok].astype(int).to_numpy()

    ok2 = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)

    df2 = df2.iloc[np.where(ok2)[0]].copy()
    x = x[ok2]
    y = y[ok2]

    out = []

    for region in include_regions:
        mask = masks.get(region)

        if mask is None:
            continue

        inside = mask[y, x]

        if not np.any(inside):
            continue

        sub = df2.iloc[np.where(inside)[0]].copy()
        sub["region"] = region
        out.append(sub)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

# =============================================================================
# Build SAXS tidy point table
# =============================================================================

def SAXS_build_data_dict_and_tidy(
    *,
    ROI_XLSX: Path,
    MANIFEST_XLSX: Path,
    CSV_ROOT: Path,
    ROI_SPECS: dict[str, int],
    valid_regions: list[str],
    roi_sheet: str = "ROIs",
    sample_region: str = "sample",
    split: bool = False,
    print_dbg: bool = False,
) -> tuple[dict, pd.DataFrame]:
    man = SAXS_normalise_columns(pd.read_excel(MANIFEST_XLSX))
    roi = SAXS_normalise_columns(pd.read_excel(ROI_XLSX, sheet_name=roi_sheet))

    if "Good" in man.columns:
        man = man[man["Good"].apply(SAXS_is_yes)].copy()

        if man.empty:
            raise ValueError("After filtering Good == 'Y', no SAXS samples remain.")

    c_file = SAXS_pick_col(man, "Analysis Number", "Filenumber", "FileNumber", "analysis_number")
    c_exp = SAXS_pick_col(man, "Experiment", "experiment")
    c_type = SAXS_pick_col(man, "TYPE", "Type", "subtype", "Subtype")

    c_folder = next((c for c in ["FOLDER NAME", "Folder Name", "folder_name"] if c in man.columns), None)
    c_snum = next((c for c in ["Sample Number", "sample_number"] if c in man.columns), None)

    r_exp = SAXS_pick_col(roi, "experiment", "Experiment")
    r_file = SAXS_pick_col(roi, "Filenumber", "Analysis Number", "FileNumber")
    r_sample_type = next((c for c in ["sample_type", "Sample Type", "sample type"] if c in roi.columns), None)

    for df, exp_col, file_col in [(man, c_exp, c_file), (roi, r_exp, r_file)]:
        df[exp_col] = df[exp_col].astype(str).str.strip()
        df[file_col] = df[file_col].astype(str).str.strip()

    man[c_type] = man[c_type].astype(str).str.strip()

    if r_sample_type is not None:
        roi[r_sample_type] = roi[r_sample_type].astype(str).str.strip().str.lower()

    roi_idx = {
        (experiment, filenumber): row
        for (experiment, filenumber), row in roi.set_index([r_exp, r_file]).iterrows()
    }

    data_dict = {"meta_by_filenumber": {}}
    tidy_all = []

    if split:
        include_regions = [sample_region]

        for region in ["dermis", "wound"]:
            if region in valid_regions:
                include_regions.extend([f"{region}_epi", f"{region}_sub"])
    else:
        include_regions = list(valid_regions)

    for _, mr in man.iterrows():
        experiment = str(mr[c_exp]).strip()
        filenumber = str(mr[c_file]).strip()
        subtype = str(mr[c_type]).strip()

        folder_name = str(mr[c_folder]).strip() if c_folder else ""
        sample_number = str(mr[c_snum]).strip() if c_snum else ""

        iq_path = CSV_ROOT / experiment / "CSVs" / f"{filenumber} IQ_fitting.csv"
        ichi_path = CSV_ROOT / experiment / "CSVs" / f"{filenumber} IChi_fitting.csv"

        iq = SAXS_load_csv(iq_path)
        ichi = SAXS_load_csv(ichi_path)

        meta = {
            "experiment": experiment,
            "Filenumber": filenumber,
            "subtype": subtype,
            "folder_name": folder_name,
            "sample_number": sample_number,
            "iq_path": str(iq_path),
            "ichi_path": str(ichi_path),
            "has_roi": False,
            "sample_type": "",
        }

        data_dict["meta_by_filenumber"][filenumber] = meta
        data_dict.setdefault(experiment, {})[filenumber] = {
            "meta": meta,
            "iq": iq,
            "ichi": ichi,
            "masks": None,
        }

        if iq is None or iq.empty or not {"x", "y"}.issubset(iq.columns):
            continue

        nx, ny = SAXS_infer_grid_shape(iq)
        roi_row = roi_idx.get((experiment, filenumber))

        sample_type = ""

        if roi_row is not None:
            meta["has_roi"] = True

            if r_sample_type is not None:
                sample_type = SAXS_clean_str(roi_row.get(r_sample_type, "")).lower()

        meta["sample_type"] = sample_type

        masks = SAXS_build_region_masks_for_sample(
            roi_row,
            nx=nx,
            ny=ny,
            roi_specs=ROI_SPECS,
            valid_regions=valid_regions,
            sample_region=sample_region,
        )

        if split:
            masks = SAXS_add_split_masks_from_orientation(
                masks,
                roi_row,
                split_regions=tuple(r for r in ["dermis", "wound"] if r in valid_regions),
            )

        data_dict[experiment][filenumber]["masks"] = masks

        tidy_iq = SAXS_label_points_by_regions(iq, masks, include_regions=include_regions)

        if not tidy_iq.empty:
            tidy_iq["source"] = "IQ"
            tidy_iq["experiment"] = experiment
            tidy_iq["Filenumber"] = filenumber
            tidy_iq["subtype"] = subtype
            tidy_iq["sample_type"] = sample_type
            tidy_iq["Sample"] = sample_number
            tidy_all.append(tidy_iq)

        if ichi is not None and not ichi.empty and {"x", "y"}.issubset(ichi.columns):
            tidy_ichi = SAXS_label_points_by_regions(ichi, masks, include_regions=include_regions)

            if not tidy_ichi.empty:
                tidy_ichi["source"] = "IChi"
                tidy_ichi["experiment"] = experiment
                tidy_ichi["Filenumber"] = filenumber
                tidy_ichi["subtype"] = subtype
                tidy_ichi["sample_type"] = sample_type
                tidy_ichi["Sample"] = sample_number
                tidy_all.append(tidy_ichi)

        if print_dbg:
            counts = {k: int(np.sum(v)) for k, v in masks.items() if v is not None}
            print(f"{experiment} | {filenumber} | {subtype} | {counts}")

    tidy = pd.concat(tidy_all, ignore_index=True) if tidy_all else pd.DataFrame()

    if tidy.empty:
        return data_dict, tidy

    if {"secondmoment", "firstmoment"}.issubset(tidy.columns):
        tidy["wa_moment"] = np.sqrt(
            pd.to_numeric(tidy["secondmoment"], errors="coerce")
            - pd.to_numeric(tidy["firstmoment"], errors="coerce") ** 2
        )

    p1 = pd.to_numeric(tidy.get("peak_position", np.nan), errors="coerce")
    p2 = pd.to_numeric(tidy.get("peak_position2", np.nan), errors="coerce")

    p1m = np.mod(p1, 180.0)
    p2m = np.mod(p2, 180.0)

    tidy["peak_position_canonical"] = np.where(
        np.isfinite(p1m) & np.isfinite(p2m),
        np.minimum(p1m, p2m),
        np.where(np.isfinite(p1m), p1m, np.where(np.isfinite(p2m), p2m, np.nan)),
    )

    tidy["peak_position_folded"] = np.mod(tidy["peak_position_canonical"] + 90.0, 180.0)

    return data_dict, tidy

# =============================================================================
# SAXS gate, trim and summarise
# =============================================================================

def SAXS_resolve_order(present: pd.Series, wanted=None, order=None) -> list[str]:
    present_set = set(present.astype(str).str.strip())

    if wanted is not None:
        wanted = [str(x).strip() for x in wanted]
        present_set &= set(wanted)

    if order is not None:
        order = [str(x).strip() for x in order]
        out = [x for x in order if x in present_set]

        if wanted is not None:
            out += [x for x in wanted if x in present_set and x not in set(out)]

        return out

    if wanted is not None:
        return [x for x in wanted if x in present_set]

    return [str(x).strip() for x in pd.unique(present.astype(str)) if str(x).strip() in present_set]

def SAXS_get_param_col(df: pd.DataFrame, canonical: str, registry: dict[str, list[str]]) -> str:
    for raw in registry.get(canonical, []):
        if raw in df.columns:
            return raw

    raise KeyError(
        f"Parameter '{canonical}' not found. Tried {registry.get(canonical, [])}. "
        f"Available: {list(df.columns)}"
    )

def SAXS_resolve_param_any_source(tidy: pd.DataFrame, canonical: str, PARAMS_IQ, PARAMS_ICHI):
    if tidy is None or tidy.empty:
        raise ValueError("Cannot resolve parameter from empty SAXS tidy table.")

    iq = tidy[tidy["source"] == "IQ"]

    if not iq.empty:
        try:
            return "IQ", SAXS_get_param_col(iq, canonical, PARAMS_IQ)
        except KeyError:
            pass

    ichi = tidy[tidy["source"] == "IChi"]

    if not ichi.empty:
        try:
            return "IChi", SAXS_get_param_col(ichi, canonical, PARAMS_ICHI)
        except KeyError:
            pass

    raise KeyError(f"Could not resolve SAXS parameter '{canonical}' in IQ or IChi.")

def SAXS_apply_point_gates(
    df: pd.DataFrame,
    *,
    curvearea_thresh=None,
    saxs_thresh=None,
    param_col=None,
    param_thresh=None,
) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)

    if curvearea_thresh is not None and "collagen_third_norm_0_1" in df.columns:
        v = pd.to_numeric(df["collagen_third_norm_0_1"], errors="coerce").to_numpy()
        mask &= np.isfinite(v) & (v >= curvearea_thresh)

    if saxs_thresh is not None and "total_SAXS_norm_0_1" in df.columns:
        v = pd.to_numeric(df["total_SAXS_norm_0_1"], errors="coerce").to_numpy()
        mask &= np.isfinite(v) & (v >= saxs_thresh)

    if param_col is not None and param_thresh is not None:
        v = pd.to_numeric(df[param_col], errors="coerce").to_numpy()
        mask &= np.isfinite(v) & (v >= param_thresh)

    return df.loc[mask].copy()

def SAXS_trim_by_std(
    df: pd.DataFrame,
    *,
    value_col: str,
    trim_std_devs=None,
    group_cols=("subtype", "region"),
) -> pd.DataFrame:
    if df is None or df.empty or value_col not in df.columns or trim_std_devs is None:
        return df.copy()

    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    keep = pd.Series(True, index=d.index)

    for _, idx in d.groupby(list(group_cols), dropna=False).groups.items():
        vals = d.loc[idx, value_col].to_numpy(dtype=float)
        finite = np.isfinite(vals)

        if finite.sum() < 2:
            continue

        mu = float(np.nanmean(vals[finite]))
        sd = float(np.nanstd(vals[finite], ddof=1))

        if not np.isfinite(sd) or sd <= 0:
            continue

        lo = mu - float(trim_std_devs) * sd
        hi = mu + float(trim_std_devs) * sd

        sub_keep = np.ones(len(vals), dtype=bool)
        sub_keep[finite] = (vals[finite] >= lo) & (vals[finite] <= hi)

        keep.loc[idx] = sub_keep

    return d.loc[keep].copy()

def SAXS_filter_gate_order(
    tidy: pd.DataFrame,
    *,
    param: str,
    subtype_order=None,
    region_order=None,
    curvearea_thresh=None,
    saxs_thresh=None,
    param_thresh=None,
    PARAMS_IQ=None, 
    PARAMS_ICHI=None,
) -> tuple[pd.DataFrame, str, list[str], list[str], str]:
    source, value_col = SAXS_resolve_param_any_source(tidy, param, PARAMS_IQ, PARAMS_ICHI)

    df = tidy[tidy["source"] == source].copy()

    if df.empty:
        raise ValueError(f"No SAXS tidy rows for source={source}")

    for col in ["experiment", "subtype", "region", "Filenumber"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    df = SAXS_apply_point_gates(
        df,
        curvearea_thresh=curvearea_thresh,
        saxs_thresh=saxs_thresh,
        param_col=value_col,
        param_thresh=param_thresh,
    )

    if df.empty:
        raise ValueError(f"No SAXS data left after gating for parameter '{param}'.")

    subtype_list = SAXS_resolve_order(df["subtype"], wanted=None, order=subtype_order)
    region_list = SAXS_resolve_order(df["region"], wanted=None, order=region_order)

    return df, value_col, subtype_list, region_list, source

def SAXS_summarise_saxs(
    df: pd.DataFrame,
    *,
    value_col: str,
    parameter: str,
    pooled: bool = False,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    pooled=True:
        stats across all points per subtype/region.

    pooled=False:
        stats across sample means per subtype/region.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    for col in ["subtype", "region", "Filenumber"]:
        d[col] = d[col].astype(str).str.strip()

    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    group_cols = ["subtype", "region"]

    n_points = d.groupby(group_cols).size().rename("n_points").reset_index()
    n_samples = d.groupby(group_cols)["Filenumber"].nunique().rename("n_samples").reset_index()

    if pooled:
        if agg == "mean":
            out = d.groupby(group_cols)[value_col].agg(mean="mean", std="std").reset_index()
        elif agg == "median":
            out = d.groupby(group_cols)[value_col].agg(median="median").reset_index()
        else:
            raise ValueError("agg must be 'mean' or 'median'")
    else:
        sample_means = (
            d.groupby(group_cols + ["Filenumber"], as_index=False)[value_col]
            .mean()
            .rename(columns={value_col: "sample_mean"})
        )

        if agg == "mean":
            out = sample_means.groupby(group_cols)["sample_mean"].agg(mean="mean", std="std").reset_index()
        elif agg == "median":
            out = sample_means.groupby(group_cols)["sample_mean"].agg(median="median").reset_index()
        else:
            raise ValueError("agg must be 'mean' or 'median'")

    out = out.merge(n_points, on=group_cols, how="left")
    out = out.merge(n_samples, on=group_cols, how="left")

    out["parameter"] = parameter
    out["value_col"] = value_col
    out["Technique"] = "SAXS"

    return out

def SAXS_summarise_saxs_per_sample(
    df: pd.DataFrame,
    *,
    value_col: str,
    parameter: str,
) -> pd.DataFrame:
    """
    One row per sample/region/parameter.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    for col in ["experiment", "subtype", "region", "Filenumber"]:
        d[col] = d[col].astype(str).str.strip()

    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    out = (
        d.groupby(["experiment", "subtype", "Filenumber", "region"], as_index=False)[value_col]
        .agg(mean="mean", std="std", n_points="count")
    )

    if "Sample" in d.columns:
        sample_lookup = (
            d[["experiment", "Filenumber", "Sample"]]
            .drop_duplicates()
            .copy()
        )

        out = out.merge(sample_lookup, on=["experiment", "Filenumber"], how="left")
    else:
        out["Sample"] = out["Filenumber"]

    out["parameter"] = parameter
    out["value_col"] = value_col
    out["Technique"] = "SAXS"

    return out

def SAXS_points_table(
    df: pd.DataFrame,
    *,
    value_col: str,
    parameter: str,
) -> pd.DataFrame:
    """
    Point-level SAXS table for one parameter.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    keep_cols = [
        "experiment",
        "subtype",
        "Filenumber",
        "Sample",
        "region",
        "x",
        "y",
        value_col,
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    out = df[keep_cols].copy()
    out = out.rename(columns={value_col: "value"})

    out["parameter"] = parameter
    out["Technique"] = "SAXS"
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    return out

def SAXS_process_saxs_parameter(
    tidy: pd.DataFrame,
    *,
    parameter: str,
    pooled: bool = False,
    agg: str = "mean",
    subtype_order=None,
    region_order=None,
    curvearea_thresh=None,
    saxs_thresh=None,
    param_thresh=None,
    trim_std_devs=None,
    PARAMS_IQ=None,
    PARAMS_ICHI=None
):
    """
    Process one SAXS parameter into:
        filtered point df
        summary df
        per-sample df
        point table
    """

    df, value_col, subtype_list, region_list, source = SAXS_filter_gate_order(
        tidy,
        param=parameter,
        subtype_order=subtype_order,
        region_order=region_order,
        curvearea_thresh=curvearea_thresh,
        saxs_thresh=saxs_thresh,
        param_thresh=param_thresh,
        PARAMS_IQ=PARAMS_IQ,
        PARAMS_ICHI=PARAMS_ICHI
    )

    df = SAXS_trim_by_std(
        df,
        value_col=value_col,
        trim_std_devs=trim_std_devs,
        group_cols=("subtype", "region"),
    )

    summary = SAXS_summarise_saxs(
        df,
        value_col=value_col,
        parameter=parameter,
        pooled=pooled,
        agg=agg,
    )

    per_sample = SAXS_summarise_saxs_per_sample(
        df,
        value_col=value_col,
        parameter=parameter,
    )

    points = SAXS_points_table(
        df,
        value_col=value_col,
        parameter=parameter,
    )

    summary["source"] = source
    per_sample["source"] = source
    points["source"] = source

    return df, summary, per_sample, points

def SAXS_process_all_saxs_parameters(
    tidy: pd.DataFrame,
    *,
    parameters: list[str],
    pooled: bool = False,
    agg: str = "mean",
    subtype_order=None,
    region_order=None,
    curvearea_thresh=None,
    saxs_thresh=None,
    param_thresh=None,
    trim_std_devs=None,
    PARAMS_IQ=None,
    PARAMS_ICHI=None
):
    """
    Process multiple SAXS parameters and return combined export-ready tables.
    """

    filtered_dfs = []
    summaries = []
    per_samples = []
    points = []

    for parameter in parameters:
        try:
            df_param, summary, per_sample, pts = SAXS_process_saxs_parameter(
                tidy,
                parameter=parameter,
                pooled=pooled,
                agg=agg,
                subtype_order=subtype_order,
                region_order=region_order,
                curvearea_thresh=curvearea_thresh,
                saxs_thresh=saxs_thresh,
                param_thresh=param_thresh,
                trim_std_devs=trim_std_devs,
                PARAMS_IQ=PARAMS_IQ,
                PARAMS_ICHI=PARAMS_ICHI
            )

            df_param["parameter"] = parameter

            filtered_dfs.append(df_param)
            summaries.append(summary)
            per_samples.append(per_sample)
            points.append(pts)

        except Exception as exc:
            print(f"[SAXS] Skipping parameter '{parameter}': {exc}")

    filtered_all = pd.concat(filtered_dfs, ignore_index=True) if filtered_dfs else pd.DataFrame()
    summary_all = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    per_sample_all = pd.concat(per_samples, ignore_index=True) if per_samples else pd.DataFrame()
    points_all = pd.concat(points, ignore_index=True) if points else pd.DataFrame()

    return filtered_all, summary_all, per_sample_all, points_all


# ===========================================================     CELLS     =============================================================

# =============================================================================
# Cell data read-in
# =============================================================================

def CELL_read_spatial_cell_density_data(
    cell_xlsx,
    sheet_name="summary",
    condition_map=None,
    region_map=None,
    value_name="Fibroblasts_per_mm2",
):
    """
    Read spatial fibroblast density data from the new cell-count workbook.

    Expected layout:
        lower dermis
            4 7 10 14 21
            values...

        lower wound bed
            4 7 10 14 21
            values...

        upper dermis
            4 7 10 14 21
            values...

        upper wound bed
            4 7 10 14 21
            values...

    Returns
    -------
    cell_points_df : pd.DataFrame
        One row per raw fibroblast-density value.

    cell_summary_df : pd.DataFrame
        One row per subtype/region with mean, std, n.
    """

    condition_map = condition_map or CELL_CONDITION_MAP
    region_map = region_map or CELL_REGION_MAP

    raw = pd.read_excel(cell_xlsx, sheet_name=sheet_name, header=None)

    rows = []
    expected_headers = {str(k).strip() for k in condition_map}

    for i, row in raw.iterrows():
        first_cell = str(row.iloc[0]).strip().lower()

        if first_cell not in region_map:
            continue

        original_region = first_cell
        region = region_map[original_region]

        header_idx = i + 1

        if header_idx >= len(raw):
            continue

        headers = raw.iloc[header_idx]

        keep_cols = []
        for col_idx, header in headers.items():
            header_clean = str(header).strip()

            # Handles Excel sometimes reading 4 as 4.0
            if header_clean.endswith(".0"):
                header_clean = header_clean[:-2]

            if header_clean in expected_headers:
                keep_cols.append((col_idx, header_clean))

        if not keep_cols:
            continue

        data_start = header_idx + 1
        data_end = len(raw)

        for j in range(data_start, len(raw)):
            possible_region_title = str(raw.iloc[j, 0]).strip().lower()

            if possible_region_title in region_map:
                data_end = j
                break

            vals = raw.iloc[j, [col_idx for col_idx, _ in keep_cols]]

            if vals.isna().all():
                data_end = j
                break

        for col_idx, original_condition in keep_cols:
            subtype = condition_map[str(original_condition).strip()]

            values = pd.to_numeric(
                raw.iloc[data_start:data_end, col_idx],
                errors="coerce",
            ).dropna()

            for repeat_idx, value in enumerate(values, start=1):
                rows.append({
                    "Sample": f"{subtype}_{region}_cell_{repeat_idx}",
                    "CellRepeat": repeat_idx,
                    "Subtype": subtype,
                    "Technique": "Cells",
                    "Metric": value_name,
                    "OriginalCondition": str(original_condition).strip(),
                    "OriginalRegion": original_region,
                    "Region": region,
                    value_name: float(value),
                })

    cell_points_df = pd.DataFrame(rows)

    if cell_points_df.empty:
        return cell_points_df, pd.DataFrame()

    cell_summary_df = (
        cell_points_df
        .groupby(["Subtype", "Region", "Technique", "Metric"], as_index=False)[value_name]
        .agg(mean="mean", std="std", n="count")
    )

    return cell_points_df, cell_summary_df

SUBTYPE_NORMALISE_MAP = {
    # Generic
    "control": "control",
    "unwounded": "control",
    "ct": "control",

    "d4": "d4",
    "pwd4": "d4",

    "d7": "d7",
    "pwd7": "d7",

    "d10": "d10",
    "pwd10": "d10",

    "d14": "d14",
    "pwd14": "d14",

    "d21": "d21",
    "pwd21": "d21",

    # Keep this available, but it will not export unless added to EXPORT_SUBTYPES
    "pbs": "pbs",
    "t5224": "t5224",
}
# ===========================================================     EXOPRT     =============================================================

def export_normalise_subtype(value):
    """
    Convert subtype labels from different techniques into shared export labels.
    """

    if pd.isna(value):
        return ""

    key = str(value).strip().lower()

    return SUBTYPE_NORMALISE_MAP.get(key, key)


def export_blank_df():
    """
    Blank sheet placeholder.
    """

    return pd.DataFrame({"No data": []})


def export_safe_sheet_df(df):
    """
    Ensure a valid dataframe is written even when empty.
    """

    if df is None or df.empty:
        return export_blank_df()

    return df.copy()


def export_filter_by_subtype(df, subtype, subtype_col="Subtype"):
    """
    Filter a dataframe by normalised subtype.
    """

    if df is None or df.empty or subtype_col not in df.columns:
        return pd.DataFrame()

    d = df.copy()
    d["_export_subtype"] = d[subtype_col].apply(export_normalise_subtype)

    d = d[d["_export_subtype"] == subtype].copy()
    d = d.drop(columns=["_export_subtype"], errors="ignore")

    return d.reset_index(drop=True)

def export_filter_saxs_for_export(df, subtype):
    """
    Keep SAXS dermis/wound regions only for one subtype.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    subtype_col = "subtype" if "subtype" in d.columns else "Subtype"

    if subtype_col not in d.columns:
        return pd.DataFrame()

    d["_export_subtype"] = d[subtype_col].apply(export_normalise_subtype)
    d = d[d["_export_subtype"] == subtype].copy()

    if "region" in d.columns:
        d = d[d["region"].astype(str).str.lower() != "sample"].copy()

    d = d.drop(columns=["_export_subtype"], errors="ignore")

    return d.reset_index(drop=True)

def export_filter_ni_for_export(summary_df, binned_df, subtype):
    """
    Combine nanoindentation summary and binned dermis-level outputs for one subtype.
    """

    outputs = []

    if summary_df is not None and not summary_df.empty:
        d = summary_df.copy()

        if "Subtype" in d.columns:
            d["_export_subtype"] = d["Subtype"].apply(export_normalise_subtype)
            d = d[d["_export_subtype"] == subtype].copy()
            d = d.drop(columns=["_export_subtype"], errors="ignore")

            if not d.empty:
                d.insert(0, "ExportLevel", "SampleSummary")
                outputs.append(d)

    if binned_df is not None and not binned_df.empty:
        d = binned_df.copy()

        if "Subtype" in d.columns:
            d["_export_subtype"] = d["Subtype"].apply(export_normalise_subtype)
            d = d[d["_export_subtype"] == subtype].copy()
            d = d.drop(columns=["_export_subtype"], errors="ignore")

            if not d.empty:
                d.insert(0, "ExportLevel", "Binned")
                outputs.append(d)

    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()

def export_filter_raw_ni_for_export(ni_points_df, subtype):
    """
    Filter raw point-level nanoindentation data for one subtype.

    Keeps dermis/raw point-level values for later regression/correlation analysis.
    """

    if ni_points_df is None or ni_points_df.empty:
        return pd.DataFrame()

    d = ni_points_df.copy()

    if "Subtype" not in d.columns:
        return pd.DataFrame()

    d["_export_subtype"] = d["Subtype"].apply(export_normalise_subtype)
    d = d[d["_export_subtype"] == subtype].copy()
    d = d.drop(columns=["_export_subtype"], errors="ignore")

    # Keep dermis only if layer is present.
    if "layer" in d.columns:
        d = d[d["layer"].astype(str).str.lower() == "dermis"].copy()

    return d.reset_index(drop=True)

def export_filter_cell_for_export(cell_points_df, cell_summary_df, subtype):
    """
    Combine raw spatial cell-density values and summary values for one subtype.
    """

    outputs = []

    if cell_points_df is not None and not cell_points_df.empty:
        d = export_filter_by_subtype(cell_points_df, subtype, subtype_col="Subtype")

        if not d.empty:
            d.insert(0, "ExportLevel", "RawValues")
            outputs.append(d)

    if cell_summary_df is not None and not cell_summary_df.empty:
        d = export_filter_by_subtype(cell_summary_df, subtype, subtype_col="Subtype")

        if not d.empty:
            d.insert(0, "ExportLevel", "Summary")
            outputs.append(d)

    return pd.concat(outputs, ignore_index=True, sort=False) if outputs else pd.DataFrame()

def export_filter_raman_summary_for_export(raman_summary_df, subtype):
    """
    Keep Raman weighted-moment rows for one subtype.
    """

    if raman_summary_df is None or raman_summary_df.empty:
        return pd.DataFrame()

    d = raman_summary_df.copy()

    if "Subtype" not in d.columns:
        return pd.DataFrame()

    d["_export_subtype"] = d["Subtype"].apply(export_normalise_subtype)
    d = d[d["_export_subtype"] == subtype].copy()
    d = d.drop(columns=["_export_subtype"], errors="ignore")

    return d.reset_index(drop=True)


def export_RamanRawSpectraToDataFrame(
    raman_dict,
    subtype,
    regions_to_export=("Dermis", "Wound"),
    spectral_regions=("FP", "EXT"),
):
    """
    Export raw/treated Raman spectral values from selected anatomical regions.

    Expected keys include:
        FP_Spectra_Treated_Dermis
        EXT_Spectra_Treated_Dermis

    If wound keys are later added, these will also be exported:
        FP_Spectra_Treated_Wound
        EXT_Spectra_Treated_Wound
    """

    rows = []

    if raman_dict is None:
        return pd.DataFrame()

    for sample_num, sample_data in raman_dict.items():
        if not isinstance(sample_data, dict):
            continue

        sample_subtype = export_normalise_subtype(sample_data.get("Subtype", ""))

        if sample_subtype != subtype:
            continue

        original_subtype = sample_data.get("Subtype", "")

        for spectral_region in spectral_regions:
            for anatomical_region in regions_to_export:
                key = f"{spectral_region}_Spectra_Treated_{anatomical_region}"

                spectra_dict = sample_data.get(key)

                if not isinstance(spectra_dict, dict) or not spectra_dict:
                    continue

                for point_index, ((x, y), sc) in enumerate(spectra_dict.items(), start=1):
                    wave = np.asarray(sc.spectral_axis, dtype=float)
                    intensity = np.asarray(sc.spectral_data, dtype=float)

                    for w, inten in zip(wave, intensity):
                        rows.append({
                            "Sample": str(sample_num),
                            "Subtype": original_subtype,
                            "Technique": "Raman",
                            "SpectralRegion": spectral_region,
                            "AnatomicalRegion": anatomical_region.lower(),
                            "PointIndex": point_index,
                            "x": x,
                            "y": y,
                            "Wave": w,
                            "Intensity": inten,
                        })

    return pd.DataFrame(rows)

def export_multitech_workbooks(
    export_root,
    export_subtypes,
    *,
    raman_summary_df=None,
    raman_dict=None,
    cell_points_df=None,
    cell_summary_df=None,
    saxs_per_sample_all_df=None,
    saxs_points_all_df=None,
    ni_summary_df=None,
    ni_binned_df=None,
    ni_points_df=None,
):
    """
    Create one workbook per subtype.

    Each workbook contains:
        Raman
        RawRaman
        Cell
        SAXS
        Nanoindentation
    """

    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    for subtype in export_subtypes:
        out_path = export_root / f"{subtype}.xlsx"

        raman_df = export_filter_raman_summary_for_export(
            raman_summary_df,
            subtype,
        )

        raw_raman_df = export_RamanRawSpectraToDataFrame(
            raman_dict,
            subtype,
            regions_to_export=("Dermis", "Wound"),
            spectral_regions=("FP", "EXT"),
        )

        cell_df = export_filter_cell_for_export(
            cell_points_df,
            cell_summary_df,
            subtype,
        )

        # SAXS: use per-sample summary as main export.
        saxs_sample_df = export_filter_saxs_for_export(
            saxs_per_sample_all_df,
            subtype,
        )

        # Optionally also include point-level SAXS values underneath.
        saxs_points_df = export_filter_saxs_for_export(
            saxs_points_all_df,
            subtype,
        )

        if not saxs_sample_df.empty and not saxs_points_df.empty:
            saxs_points_df = saxs_points_df.copy()
            saxs_sample_df = saxs_sample_df.copy()

            saxs_sample_df.insert(0, "ExportLevel", "PerSample")
            saxs_points_df.insert(0, "ExportLevel", "PointLevel")

            saxs_df = pd.concat(
                [saxs_sample_df, saxs_points_df],
                ignore_index=True,
                sort=False,
            )

        elif not saxs_sample_df.empty:
            saxs_df = saxs_sample_df.copy()
            saxs_df.insert(0, "ExportLevel", "PerSample")

        elif not saxs_points_df.empty:
            saxs_df = saxs_points_df.copy()
            saxs_df.insert(0, "ExportLevel", "PointLevel")

        else:
            saxs_df = pd.DataFrame()

        ni_df = export_filter_ni_for_export(
            ni_summary_df,
            ni_binned_df,
            subtype,
        )
        
        raw_ni_df = export_filter_raw_ni_for_export(
            ni_points_df,
            subtype,
        )

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            export_safe_sheet_df(raman_df).to_excel(
                writer,
                sheet_name="Raman",
                index=False,
            )

            export_safe_sheet_df(raw_raman_df).to_excel(
                writer,
                sheet_name="RawRaman",
                index=False,
            )

            export_safe_sheet_df(cell_df).to_excel(
                writer,
                sheet_name="Cell",
                index=False,
            )

            export_safe_sheet_df(saxs_df).to_excel(
                writer,
                sheet_name="SAXS",
                index=False,
            )

            export_safe_sheet_df(ni_df).to_excel(
                writer,
                sheet_name="Nanoindentation",
                index=False,
            )
            
            export_safe_sheet_df(raw_ni_df).to_excel(
                writer,
                sheet_name="RawNanoindentation",
                index=False,
            )

        print(f"Exported: {out_path}")
