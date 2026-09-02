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
    

_CANONICAL_ORDER = ["despike", "smooth", "baseline", "normalise"]

# ============================================================
# SHARED SUBTYPE NORMALISATION
# ============================================================

def clean_label(value):
    """
    Clean labels consistently across Raman, NI, SAXS and export.
    """
    if pd.isna(value):
        return ""

    s = str(value).strip().lower()
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = s.replace("+", " ")
    s = " ".join(s.split())

    compact = s.replace(" ", "")

    return s, compact

SUBTYPE_NORMALISE_MAP = {
    # Wound
    "control": "control",
    "ct": "control",
    "unwounded": "control",

    "d4": "d4",
    "pwd4": "d4",
    "4": "d4",

    "d7": "d7",
    "pwd7": "d7",
    "7": "d7",

    "d10": "d10",
    "pwd10": "d10",
    "10": "d10",

    "d14": "d14",
    "pwd14": "d14",
    "14": "d14",

    "d21": "d21",
    "pwd21": "d21",
    "21": "d21",

    # Bleomycin baseline
    "pbs": "pbs",
    "pbscontrol": "pbs",
    "pbs control": "pbs",

    "2w": "2w",
    "2week": "2w",
    "2weeks": "2w",
    "2wbleo": "2w",
    "2wbleomycin": "2w",
    "bleo2w": "2w",
    "bleomycin2w": "2w",

    "4w": "4w",
    "4week": "4w",
    "4weeks": "4w",
    "4wbleo": "4w",
    "4wbleomycin": "4w",
    "bleo4w": "4w",
    "bleomycin4w": "4w",
    "4w3r": "4w",
    "4w 3r": "4w",
    "4w_3r": "4w",
    "4w5r": "4w",
    "4w 5r": "4w",
    "4w_5r": "4w",

    # BM / metformin style names
    "bm": "bm",
    "bleomycinsmetformin": "bm",
    "bleomycinmetformin": "bm",

    "bmmet": "bmmet",
    "bm met": "bmmet",
    "bm metformin": "bmmet",
    "bmmetformin": "bmmet",

    "pbsmet": "pbsmet",
    "pbs met": "pbsmet",
    "pbs metformin": "pbsmet",
    "pbsmetformin": "pbsmet",

    "4wmet": "4wmet",
    "4w met": "4wmet",
    "4w metformin": "4wmet",
    "4wmetformin": "4wmet",
    "4wbleomycinmetformin": "4wmet",

    # OKN
    "pbsokn": "pbsokn",
    "pbs okn": "pbsokn",
    "pbs okn 007": "pbsokn",

    "4wokn": "4wokn",
    "4w okn": "4wokn",
    "4w okn 007": "4wokn",
    "4wbleomycinokn007": "4wokn",

}

def normalise_subtype(value):
    """
    Convert any manifest subtype label to one canonical export label.
    """
    s, compact = clean_label(value)

    if s in SUBTYPE_NORMALISE_MAP:
        return SUBTYPE_NORMALISE_MAP[s]

    if compact in SUBTYPE_NORMALISE_MAP:
        return SUBTYPE_NORMALISE_MAP[compact]

    return compact
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

    Normalises subtype labels so PBS_MET, pbs met, PBSMET etc.
    all become the same canonical label.
    """

    required = {"Sample Number", "TYPE"}
    missing = required.difference(sample_manifest_df.columns)

    if missing:
        raise ValueError(f"Sample manifest missing required columns: {sorted(missing)}")

    data_dict = {}

    subtype_lookup = {
        normalise_subtype(s)
        for s in Subtypes
    }

    for _, row in sample_manifest_df.iterrows():
        original_subtype = str(row["TYPE"]).strip()
        subtype = normalise_subtype(original_subtype)

        if subtype not in subtype_lookup:
            continue

        sample_num = RAMAN_clean_sample_number(row["Sample Number"])

        if not sample_num:
            continue

        data_dict[sample_num] = {
            "Type": Type,
            "Subtype": subtype,
            "OriginalSubtype": original_subtype,
            "direction": str(row.get("direction", "forward")).strip().lower(),
        }

    return data_dict

def RAMAN_find_subtype_dir(DataDir, subtype, original_subtype=None):
    """
    Find subtype folder robustly, allowing case/underscore variations.
    """

    DataDir = Path(DataDir)

    candidates = [
        subtype,
        str(subtype).upper(),
        str(subtype).lower(),
    ]

    if original_subtype:
        candidates.extend([
            original_subtype,
            str(original_subtype).upper(),
            str(original_subtype).lower(),
        ])

    for candidate in candidates:
        path = DataDir / str(candidate)
        if path.exists():
            return path

    target_norm = normalise_subtype(subtype)

    for folder in DataDir.iterdir():
        if folder.is_dir() and normalise_subtype(folder.name) == target_norm:
            return folder

    return DataDir / str(subtype)

# =============================================================================
# Raman read-in functions
# =============================================================================
def RAMAN_readindata(DataDir, data_dict):
    """
    Read Raman fingerprint and extended line scan text files into data_dict.

    Standard expected file structure:
        DataDir / subtype / sampleA_linescan1.txt
        DataDir / subtype / sampleA_linescan1_extended.txt

    Also allows small filename variations via RAMAN_find_spectrum_file().
    """

    DataDir = Path(DataDir)

    for sample_num, sample_data in data_dict.items():
        subtype = sample_data["Subtype"]
        original_subtype = sample_data.get("OriginalSubtype", subtype)
    
        subtype_dir = RAMAN_find_subtype_dir(
            DataDir,
            subtype=subtype,
            original_subtype=original_subtype,
        )

        paths = {
            "FP": RAMAN_find_spectrum_file(subtype_dir, sample_num, "linescan1"),
            "EXT": RAMAN_find_spectrum_file(subtype_dir, sample_num, "linescan1_extended"),
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

                if df.empty:
                    print(f"{key} file read but no valid rows for sample {sample_num}: {path}")
                    continue

                data_dict[sample_num][key] = df

            except FileNotFoundError:
                print(f"{key} file not found for sample {sample_num}: {path}")

            except Exception as exc:
                print(f"Could not read {key} file for sample {sample_num}: {path} | {exc}")

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

def RAMAN_clean_sample_number(value):
    """
    Keep sample IDs consistent across manifests and filenames.

    Handles:
        1       -> "1"
        1.0     -> "1"
        "1"     -> "1"
        "PBS1"  -> "PBS1"
    """

    if pd.isna(value):
        return ""

    s = str(value).strip()

    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return s

def RAMAN_find_spectrum_file(folder, sample_num, suffix):
    """
    Find Raman line-scan file using the standard naming convention first,
    then fall back to a small set of common alternatives.
    """

    folder = Path(folder)
    sample_num = str(sample_num).strip()

    exact_candidates = [
        folder / f"{sample_num}A_{suffix}.txt",
        folder / f"{sample_num}_{suffix}.txt",
        folder / f"{sample_num}a_{suffix}.txt",
    ]

    for path in exact_candidates:
        if path.exists():
            return path

    patterns = [
        f"{sample_num}*{suffix}.txt",
        f"{sample_num}*{suffix.replace('_extended', '')}*extended*.txt",
    ]

    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]

    return exact_candidates[0]

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

    If dermis/epi coordinate columns are missing, all treated spectra are kept
    as Dermis so downstream export still works.
    """

    sample_manifest_df = sample_manifest_df.copy()
    sample_manifest_df["Sample Number"] = sample_manifest_df["Sample Number"].apply(
        RAMAN_clean_sample_number
    )

    required_coord_cols = {"dermis x", "dermis y", "epi x", "epi y"}
    has_coords = required_coord_cols.issubset(sample_manifest_df.columns)

    for sample_num, sample_data in data_dict.items():
        manifest_row = sample_manifest_df[
            sample_manifest_df["Sample Number"] == RAMAN_clean_sample_number(sample_num)
        ]

        if manifest_row.empty:
            print(f"Sample {sample_num} not found in Raman manifest")
            continue

        row = manifest_row.iloc[0]
        direction = str(row.get("direction", sample_data.get("direction", "forward"))).strip().lower()

        for region_key in ["FP_Spectra_Treated", "EXT_Spectra_Treated"]:
            if region_key not in sample_data:
                continue

            spectra_dict = sample_data[region_key]

            if not isinstance(spectra_dict, dict) or not spectra_dict:
                continue

            if not has_coords:
                sample_data[f"{region_key}_Dermis"] = spectra_dict.copy()
                print(
                    f"No dermis/epi coordinate columns found; keeping all {region_key} "
                    f"spectra as Dermis for sample {sample_num}."
                )
                continue

            try:
                dermis_start = np.array([row["dermis x"], row["dermis y"]], dtype=float)
                dermis_end = np.array([row["epi x"], row["epi y"]], dtype=float)
            except Exception:
                sample_data[f"{region_key}_Dermis"] = spectra_dict.copy()
                print(
                    f"Invalid dermis/epi coordinates; keeping all {region_key} "
                    f"spectra as Dermis for sample {sample_num}."
                )
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

            if not trimmed_spectra:
                trimmed_spectra = spectra_dict.copy()
                print(
                    f"Dermis trim returned no spectra; keeping all {region_key} "
                    f"spectra as Dermis for sample {sample_num}."
                )

            sample_data[f"{region_key}_Dermis"] = trimmed_spectra

            dermis_length_um = pd.to_numeric(row.get("leng dermis", np.nan), errors="coerce")
            step_um = pd.to_numeric(row.get("Step size", np.nan), errors="coerce")

            if np.isfinite(dermis_length_um) and np.isfinite(step_um) and step_um > 0:
                expected_points = int(round(float(dermis_length_um) / float(step_um)))
                diff = abs(len(trimmed_spectra) - expected_points)

                if diff > 3:
                    print(
                        f"Warning: dermis length mismatch for sample {sample_num} "
                        f"{region_key}: expected≈{expected_points} points "
                        f"from {dermis_length_um:.2f} µm / {step_um:.2f} µm step, "
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
    st_key = str(ST).strip().lower()
    
    col_map = {
        "bleo": {
            "lowerdermis": "lower dermis",
            "upperdermis": "upper dermis",
            "linescan": "line scan",
        },
        "bleomycin": {
            "lowerdermis": "lower dermis",
            "upperdermis": "upper dermis",
            "linescan": "line scan",
        },
        "ap1": {
            "linescan": "linescan",
            "horiz_linescan": "horiz_linescan",
        },
        "wounding": {
            "linescan": "linescan",
            "horiz_linescan": "horiz_linescan",
        },
    }

    if st_key not in col_map:
        raise ValueError(f"Unknown set_type {ST!r}")

    region_cols = col_map[st_key]

    df = pd.read_csv(manifest_path, dtype=str).fillna("yes")
    df.columns = [str(c).strip() for c in df.columns]

    required = {"FOLDER NAME", "TYPE"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(f"NI manifest missing required columns: {sorted(missing)}")

    data_dict = {}

    for _, row in df.iterrows():
        folder = str(row["FOLDER NAME"]).strip()

        if not folder:
            continue

        original_type = str(row.get("TYPE", "")).strip()

        sample = {
            "SAMPLE NAME": row.get("SAMPLE NAME", folder),
            "TYPE": NI_clean_type(original_type),          # normalised subtype for analysis/export
            "Original TYPE": original_type,                # original label for path matching
            "FOLDER NAME": folder,
            "Sample Number": NI_clean_sample_number(row.get("Sample Number", folder)),
        }

        for canon, col in region_cols.items():
            sample[canon] = str(row.get(col, "yes")).strip().lower()

        for c in row.index:
            if c not in sample and not str(c).startswith("Unnamed:"):
                sample[c] = row[c]

        data_dict[folder] = sample

    for folder, sample in data_dict.items():
        sample_type = sample["TYPE"]
        original_sample_type = sample.get("Original TYPE", sample_type)

        for canon_reg in regions:
            NI_initialise_ni_region(sample, canon_reg)

            if NI_is_no(sample.get(canon_reg, "yes")):
                continue

            csv_path = NI_build_ni_csv_path(
                base_path=base_path,
                set_type=st_key,
                sample_type=sample_type,
                original_sample_type=original_sample_type,
                folder=folder,
                canon_reg=canon_reg,
            )

            if not os.path.exists(csv_path):
                sample.setdefault("_missing_ni_csvs", []).append({
                    "RegionKey": NI_normalise_region_key(canon_reg),
                    "Path": csv_path,
                })
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

def NI_normalise_region_key(region):
    """
    Normalise NI region names from driver/manifest/path conventions.
    """
    key = str(region).strip().lower()
    key = key.replace("_", " ")
    key = " ".join(key.split())

    aliases = {
        "line scan": "line scan",
        "linescan": "line scan",

        "lower dermis": "lower dermis",
        "lowerdermis": "lower dermis",

        "upper dermis": "upper dermis",
        "upperdermis": "upper dermis",

        "horiz linescan": "horiz_linescan",
        "horiz_linescan": "horiz_linescan",
    }

    return aliases.get(key, key)

def NI_initialise_ni_region(sample, canon_reg):
    """
    Initialise region key in sample dictionary.
    """
    target = NI_normalise_region_key(canon_reg)

    if target:
        sample[target] = {}

def NI_store_ni_region(sample, canon_reg, region_dict):
    """
    Store loaded region dictionary under standard readable region key.
    """
    target = NI_normalise_region_key(canon_reg)

    if target:
        sample[target] = region_dict

def NI_find_existing_dir(*candidates):
    """
    Return the first existing directory from candidate paths.
    If none exists, return the first candidate.
    """
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]

def NI_build_ni_csv_path(base_path, set_type, sample_type, folder, canon_reg, original_sample_type=None):
    """
    Build expected processed nanoindentation CSV path.

    Supports:
        bleo line scans
        bleo upper/lower dermis grids
        AP1 line scans
        wounding line scans
    """

    st_lower = str(set_type).strip().lower()
    sample_type = str(sample_type).strip()
    folder = str(folder).strip()
    original_sample_type = str(original_sample_type or sample_type).strip()

    reg_key = str(canon_reg).strip().lower()
    reg_key = reg_key.replace("_", " ")
    reg_key = " ".join(reg_key.split())

    reg_alias = {
        "line scan": "linescan",
        "linescan": "linescan",

        "lower dermis": "lower dermis",
        "lowerdermis": "lower dermis",

        "upper dermis": "upper dermis",
        "upperdermis": "upper dermis",

        "horiz linescan": "horiz_linescan",
        "horiz_linescan": "horiz_linescan",
    }

    reg_key = reg_alias.get(reg_key, reg_key)

    if st_lower in {"bleo", "bleomycin"}:
        subfolder = {
            "lower dermis": "lower dermis",
            "upper dermis": "upper dermis",
            "linescan": "line scan",
        }[reg_key]

        subtype_folder_aliases = {
            "pbs": ["PBS", "pbs"],
            "2w": ["2W", "2w"],
            "4w": ["4W", "4w", "4W_3R", "4w_3r", "4W_5R", "4w_5r"],

            "pbsmet": ["PBS_MET", "pbs_met", "PBSMET", "pbsmet"],
            "bmmet": ["BM_MET", "bm_met", "BMMET", "bmmet"],
            "4wmet": ["BM_MET", "bm_met", "4W_MET", "4w_met", "4WMET", "4wmet"],

            "pbsokn": ["PBS_OKN", "pbs_okn", "PBSOKN", "pbsokn"],
            "4wokn": ["4W_OKN", "4w_okn", "4WOKN", "4wokn"],
        }

        subtype_key = normalise_subtype(sample_type)

        type_candidates = [
            original_sample_type,
            sample_type,
            sample_type.upper(),
            sample_type.lower(),
            original_sample_type.upper(),
            original_sample_type.lower(),
        ]

        type_candidates.extend(subtype_folder_aliases.get(subtype_key, []))

        # remove duplicates while preserving order
        type_candidates = list(
            dict.fromkeys(
                [str(x).strip() for x in type_candidates if str(x).strip()]
            )
        )

        base_candidates = [
            str(base_path),
            str(base_path).replace("/bleo", "/bleomycin"),
            str(base_path).replace("/bleomycin", "/bleo"),
        ]

        base_candidates = list(dict.fromkeys(base_candidates))

        folder_candidates = []

        for base in base_candidates:
            for type_folder in type_candidates:
                folder_candidates.append(
                    os.path.join(base, type_folder, subfolder)
                )

        folder_path = NI_find_existing_dir(*folder_candidates)

        filename_options = [
            f"OutputExcel_{folder}_linescan.csv",
            f"OutputExcel_{folder}_{subfolder}.csv",
            f"OutputExcel_{folder}_{subfolder.replace(' ', '')}.csv",
            f"OutputExcel_{folder}.csv",
        ]

        for filename in filename_options:
            path = os.path.join(folder_path, filename)
            if os.path.exists(path):
                return path

        return os.path.join(folder_path, filename_options[0])

    if st_lower == "ap1":
        orient = "horiz" if reg_key == "horiz_linescan" else "vert"

        sample_type_lower = normalise_subtype(sample_type)
        group = "AP1" if sample_type_lower in {"ts", "vh"} else "CL"

        folder_path = NI_find_existing_dir(
            os.path.join(base_path, f"{group} {orient}", sample_type),
            os.path.join(base_path, f"{group} {orient}", sample_type.upper()),
            os.path.join(base_path, f"{group} {orient}", sample_type.lower()),
        )

        return os.path.join(folder_path, f"OutputExcel_{folder}_linescan.csv")

    if st_lower == "wounding":
        folder_path = NI_find_existing_dir(
            os.path.join(base_path, set_type, sample_type),
            os.path.join(base_path, set_type, sample_type.upper()),
            os.path.join(base_path, set_type, sample_type.lower()),
        )

        return os.path.join(folder_path, f"OutputExcel_{folder}_linescan.csv")

    raise ValueError(f"Unhandled nanoindentation set_type: {set_type}")

def NI_build_ni_region_dict(df_raw):
    """
    Convert one processed nanoindentation CSV to a coordinate-keyed dictionary.
    """

    region_dict = {}

    df_raw = df_raw.copy()
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    for i, r in df_raw.iterrows():
        try:
            x = r.get("x", np.nan)
            y = r.get("y", np.nan)
            xy = f"{x}_{y}_{i}"

            region_dict[xy] = {
                "File Name": r.get("File Name", ""),
                "x": x,
                "y": y,

                "Eff modulus from file": r.get("Eff modulus from file", np.nan),
                "modulus from file": r.get("modulus from file", np.nan),

                "Hertz - Contact Point": r.get("Hertz - Contact Point", np.nan),
                "Hertz - Modulus(Pa) fit": r.get("Hertz - Modulus(Pa) fit", np.nan),
                "Hertz - Rsq": r.get("Hertz - Rsq", np.nan),

                "RoV - Contact Point": r.get("RoV - Contact Point", np.nan),

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

    _screen_field("Eff modulus from file")
    _screen_field("modulus from file")

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

def NI_clean_sample_number(value):
    """
    Keep NI sample IDs consistent across manifests and exports.
    """

    if pd.isna(value):
        return ""

    s = str(value).strip()

    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return s

def NI_clean_type(value):
    """
    Normalise NI subtype labels.
    """
    return normalise_subtype(value)

def NI_is_no(value):
    """
    Interpret manifest flags.
    """

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {"no", "n", "false", "0"}
# =============================================================================
# Nanoindentation screening
# =============================================================================
def NI_ScreenData(y, rsq_values=None, field=None, rsq_min=0.5, hi=1_000_000):
    """
    Field-aware screening for nanoindentation outputs.
    """

    y = np.asarray(y, dtype=np.float64)
    field = "" if field is None else str(field)

    is_file_mod = field in {
        "Eff modulus from file",
        "modulus from file",
    }

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

    if is_file_mod or is_hertz_mod or is_op_mod or is_visco_fit or is_relaxfrac:
        y = np.where(y < 0, np.nan, y)

    if is_file_mod or is_hertz_mod or is_op_mod:
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

    If measured lengths are missing or imperfect, do NOT assign everything
    to dermis. Instead, assign what can be assigned and label the rest as
    'unassigned'.
    """

    line_scan = sample.get("line scan", {})

    if not isinstance(line_scan, dict) or not line_scan:
        return

    sorted_keys = sorted(
        line_scan.keys(),
        key=lambda k: (float(line_scan[k]["y"]), float(line_scan[k]["x"])),
    )

    n_actual = len(sorted_keys)

    # Default: every point starts unassigned
    for key in sorted_keys:
        line_scan[key]["layer"] = "unassigned"

    try:
        total_pts_manifest = int(float(sample.get("no pts", np.nan)))

        lengths = {
            "left glass": int(float(sample.get("Measured left glass", 0))),
            "subcut": int(float(sample.get("Measured Subcut layer", 0))),
            "dermis": int(float(sample.get("Measured dermis", 0))),
            "epidermis": int(float(sample.get("Measured epi", 0))),
            "right glass": int(float(sample.get("Measured right glass", 0))),
        }

    except (TypeError, ValueError):
        print(
            f"Sample {sample_key}: invalid vertical region lengths; "
            f"leaving line scan points as unassigned."
        )
        return

    if all(v == 0 for v in lengths.values()):
        print(
            f"Sample {sample_key}: no vertical lengths; "
            f"leaving line scan points as unassigned."
        )
        return

    n_from_lengths = sum(lengths.values())

    if n_from_lengths != total_pts_manifest:
        print(
            f"Warning: {sample_key}: vertical lengths do not match manifest total "
            f"({n_from_lengths} != {total_pts_manifest}). "
            f"Assigning available regions only; remaining points stay unassigned."
        )

    if n_from_lengths != n_actual:
        print(
            f"Warning: {sample_key}: vertical lengths do not match loaded CSV points "
            f"({n_from_lengths} != {n_actual}). "
            f"Assigning available regions only; remaining points stay unassigned."
        )

    current_idx = 0

    for layer, length in lengths.items():
        if length <= 0:
            continue

        start = current_idx
        end = current_idx + length

        for i in range(start, min(end, n_actual)):
            line_scan[sorted_keys[i]]["layer"] = layer

        current_idx = end

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

    region_key can be:
        "line scan"
        "lower dermis"
        "upper dermis"
        list/tuple of region keys
    """

    if isinstance(region_key, (list, tuple, set)):
        frames = [
            NI_PointsToDataFrame(data_dict, VAR_MAP, region_key=rk)
            for rk in region_key
        ]

        frames = [df for df in frames if df is not None and not df.empty]

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    rows = []

    for sample_key, sample in data_dict.items():
        region_dict = sample.get(region_key, {})

        if not isinstance(region_dict, dict) or not region_dict:
            continue

        for point_index, (point_key, point) in enumerate(region_dict.items()):
            layer = point.get("layer", np.nan)
            region = point.get("region", np.nan)

            if region_key in {"lower dermis", "upper dermis"}:
                layer = "dermis"
                region = region_key

            row = {
                "Sample": NI_clean_sample_number(sample.get("Sample Number", sample_key)),
                "SampleKey": sample_key,
                "SampleName": sample.get("SAMPLE NAME", ""),
                "Subtype": NI_clean_type(sample.get("TYPE", "")),
                "Technique": "Nanoindentation",
                "RegionKey": region_key,
                "PointKey": point_key,
                "PointIndex": point_index,
                "x": point.get("x", np.nan),
                "y": point.get("y", np.nan),
                "layer": layer,
                "region": region,
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

def NI_PrepareBinnedData(data_dict, VAR_MAP, nbins=5, layer="dermis", region_keys=None):
    """
    Bin nanoindentation data by sample and selected layer.

    region_keys controls which loaded NI regions are flattened.
    """

    if region_keys is None:
        region_keys = ["line scan"]

    readable_region_keys = [
        NI_normalise_region_key(rk)
        for rk in region_keys
    ]

    df_points = NI_PointsToDataFrame(
        data_dict,
        VAR_MAP,
        region_key=readable_region_keys,
    )

    if df_points.empty:
        return {}, df_points, pd.DataFrame()

    if layer is not None and "layer" in df_points.columns:
        df_points_layer = df_points[
            df_points["layer"].astype(str).str.lower() == str(layer).lower()
        ].copy()
    else:
        df_points_layer = df_points.copy()

    if df_points_layer.empty:
        print(f"No nanoindentation points found for layer={layer!r}")
        return {}, df_points_layer, pd.DataFrame()

    binned_rows = []
    binned_dict = {}

    group_cols = ["SampleKey", "RegionKey"]

    for (sample_key, region_key), df_sample in df_points_layer.groupby(group_cols, sort=False):
        df_sample = df_sample.sort_values(["y", "x"]).reset_index(drop=True)

        n_points = len(df_sample)

        if n_points == 0:
            continue

        n_eff = min(nbins, n_points)
        index_bins = np.array_split(np.arange(n_points), n_eff)

        sample_type = df_sample["Subtype"].iloc[0]
        sample_num = df_sample["Sample"].iloc[0]
        region_label = df_sample["region"].dropna().iloc[0] if df_sample["region"].notna().any() else ""

        binned_dict.setdefault(sample_type, {})
        binned_dict[sample_type].setdefault(sample_key, {})
        binned_dict[sample_type][sample_key].setdefault(region_key, [])

        for bin_idx, idxs in enumerate(index_bins, start=1):
            df_bin = df_sample.iloc[idxs]

            row = {
                "Sample": sample_num,
                "SampleKey": sample_key,
                "Subtype": sample_type,
                "Technique": "Nanoindentation",
                "Layer": layer,
                "RegionKey": region_key,
                "Region": region_label,
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
                "RegionKey": region_key,
                "Region": region_label,
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
            binned_dict[sample_type][sample_key][region_key].append(bin_entry)

    df_binned = pd.DataFrame(binned_rows)

    return binned_dict, df_points_layer, df_binned

def NI_SampleSummary(df_points_layer, VAR_MAP):
    """
    Create one row per sample and region from selected-layer nanoindentation points.
    """

    if df_points_layer.empty:
        return pd.DataFrame()

    rows = []

    group_cols = ["Sample", "SampleKey", "Subtype", "RegionKey"]

    for keys, df_sample in df_points_layer.groupby(group_cols, dropna=False):
        sample, sample_key, subtype, region_key = keys

        row = {
            "Sample": sample,
            "SampleKey": sample_key,
            "Subtype": subtype,
            "Technique": "Nanoindentation",
            "RegionKey": region_key,
            "Region": df_sample["region"].dropna().iloc[0] if df_sample["region"].notna().any() else "",
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
    """
    Infer grid shape from x/y columns.
    """

    if df is None or df.empty or not {"x", "y"}.issubset(df.columns):
        return 0, 0

    x = pd.to_numeric(df["x"], errors="coerce").dropna()
    y = pd.to_numeric(df["y"], errors="coerce").dropna()

    if x.empty or y.empty:
        return 0, 0

    return int(np.floor(x.max())) + 1, int(np.floor(y.max())) + 1

def SAXS_clean_id(v):
    """
    Clean file/sample IDs while avoiding 1.0 vs 1 mismatches.
    """

    if pd.isna(v):
        return ""

    s = str(v).strip()

    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return s

def SAXS_full_mask(nx, ny):
    """
    Full-grid mask fallback.
    """

    if nx <= 0 or ny <= 0:
        return np.zeros((0, 0), dtype=bool)

    return np.ones((ny, nx), dtype=bool)
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
    """
    Build ROI masks for one SAXS sample.

    If no ROI is available, keep the full grid as sample_region so the sample
    is not lost from the export.
    """

    blank = np.zeros((ny, nx), dtype=bool)
    masks = {region: blank.copy() for region in valid_regions}

    if nx <= 0 or ny <= 0:
        return masks

    if roi_row is None:
        if sample_region in masks:
            masks[sample_region] = SAXS_full_mask(nx, ny)
        elif "dermis" in masks:
            masks["dermis"] = SAXS_full_mask(nx, ny)

        return masks

    sample_type = SAXS_clean_str(roi_row.get("sample_type", "")).lower()

    control_labels = {"control", "ct", "pbs", "unwounded"}

    for region in valid_regions:
        if region in roi_specs:
            pts = SAXS_roi_points_from_row(roi_row, region, roi_specs[region])
        else:
            pts = SAXS_roi_points_auto(roi_row, region)

        if sample_type in control_labels and region == "wound":
            masks[region] = blank.copy()
        else:
            masks[region] = SAXS_poly_mask(pts, nx=nx, ny=ny) if pts else blank.copy()

    if sample_region in valid_regions and not masks[sample_region].any():
        other_masks = [
            m for region, m in masks.items()
            if region != sample_region and m is not None and m.any()
        ]

        if other_masks:
            masks[sample_region] = np.logical_or.reduce(other_masks)
        else:
            masks[sample_region] = SAXS_full_mask(nx, ny)

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

    valid_masks = [m for m in masks.values() if m is not None and m.size > 0]

    if not valid_masks:
        return pd.DataFrame()

    ny, nx = valid_masks[0].shape

    d = df.copy()
    d["x"] = pd.to_numeric(d["x"], errors="coerce")
    d["y"] = pd.to_numeric(d["y"], errors="coerce")

    d = d[d["x"].notna() & d["y"].notna()].copy()

    if d.empty:
        return pd.DataFrame()

    x = d["x"].astype(int).to_numpy()
    y = d["y"].astype(int).to_numpy()

    ok = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)

    d = d.iloc[np.where(ok)[0]].copy()
    x = x[ok]
    y = y[ok]

    out = []

    for region in include_regions:
        mask = masks.get(region)

        if mask is None or mask.size == 0:
            continue

        inside = mask[y, x]

        if not np.any(inside):
            continue

        sub = d.iloc[np.where(inside)[0]].copy()
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

    try:
        roi = SAXS_normalise_columns(pd.read_excel(ROI_XLSX, sheet_name=roi_sheet))
    except Exception as exc:
        print(f"No usable SAXS ROI workbook/sheet found: {ROI_XLSX} | {exc}")
        roi = pd.DataFrame()

    if "Good" in man.columns:
        man = man[man["Good"].apply(SAXS_is_yes)].copy()

        if man.empty:
            raise ValueError("After filtering Good == 'Y', no SAXS samples remain.")

    c_file = SAXS_pick_col(man, "Analysis Number", "Filenumber", "FileNumber", "analysis_number")
    c_exp = SAXS_pick_col(man, "Experiment", "experiment")
    c_type = SAXS_pick_col(man, "TYPE", "Type", "subtype", "Subtype")

    c_folder = next((c for c in ["FOLDER NAME", "Folder Name", "folder_name"] if c in man.columns), None)
    c_snum = next((c for c in ["Sample Number", "sample_number"] if c in man.columns), None)

    man[c_exp] = man[c_exp].apply(SAXS_clean_id)
    man[c_file] = man[c_file].apply(SAXS_clean_id)
    man[c_type] = man[c_type].apply(normalise_subtype)

    roi_idx = {}
    r_sample_type = None

    if not roi.empty:
        try:
            r_exp = SAXS_pick_col(roi, "experiment", "Experiment")
            r_file = SAXS_pick_col(roi, "Filenumber", "Analysis Number", "FileNumber")
            r_sample_type = next((c for c in ["sample_type", "Sample Type", "sample type"] if c in roi.columns), None)

            roi[r_exp] = roi[r_exp].apply(SAXS_clean_id)
            roi[r_file] = roi[r_file].apply(SAXS_clean_id)

            if r_sample_type is not None:
                roi[r_sample_type] = roi[r_sample_type].astype(str).str.strip().str.lower()

            roi_idx = {
                (experiment, filenumber): row
                for (experiment, filenumber), row in roi.set_index([r_exp, r_file]).iterrows()
            }

        except Exception as exc:
            print(f"Could not index SAXS ROI workbook; continuing without ROIs. | {exc}")
            roi_idx = {}

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
        experiment = SAXS_clean_id(mr[c_exp])
        filenumber = SAXS_clean_id(mr[c_file])
        subtype = normalise_subtype(mr[c_type])

        folder_name = str(mr[c_folder]).strip() if c_folder else ""
        sample_number = SAXS_clean_id(mr[c_snum]) if c_snum else filenumber

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
            print(f"Missing or invalid IQ CSV for SAXS file {filenumber}: {iq_path}")
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

        if split and roi_row is not None:
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
        sm = pd.to_numeric(tidy["secondmoment"], errors="coerce")
        fm = pd.to_numeric(tidy["firstmoment"], errors="coerce")
        tidy["wa_moment"] = np.sqrt(np.maximum(sm - fm ** 2, 0))

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

    if param_col is not None:
        v = pd.to_numeric(df[param_col], errors="coerce").to_numpy()
        mask &= np.isfinite(v)

        if param_thresh is not None:
            mask &= v >= param_thresh

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
        if col in d.columns:
            d[col] = d[col].astype(str).str.strip()

    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    group_cols = ["experiment", "subtype", "Filenumber", "region"]

    out = (
        d.groupby(group_cols, as_index=False)[value_col]
        .agg(mean="mean", std="std", n_points="count")
    )

    lookup_cols = [c for c in ["experiment", "Filenumber", "Sample", "sample_type"] if c in d.columns]

    if lookup_cols:
        sample_lookup = d[lookup_cols].drop_duplicates().copy()
        out = out.merge(sample_lookup, on=["experiment", "Filenumber"], how="left")

    if "Sample" not in out.columns:
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
        "sample_type",
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
    Read spatial fibroblast density data from a cell-count workbook.

    Returns
    -------
    cell_points_df : pd.DataFrame
        One row per raw fibroblast-density value.

    cell_summary_df : pd.DataFrame
        One row per subtype/region with mean, std, n.
    """

    if condition_map is None:
        condition_map = {
            "4": "d4",
            "7": "d7",
            "10": "d10",
            "14": "d14",
            "21": "d21",
        }

    if region_map is None:
        region_map = {
            "lower dermis": "dermis_sub",
            "upper dermis": "dermis_epi",
            "lower wound bed": "wound_sub",
            "upper wound bed": "wound_epi",
        }

    try:
        raw = pd.read_excel(cell_xlsx, sheet_name=sheet_name, header=None)
    except Exception as exc:
        print(f"No usable cell data found: {cell_xlsx} | {exc}")
        return pd.DataFrame(), pd.DataFrame()

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

# ===========================================================     EXPORT     =============================================================

def export_normalise_subtype(value):
    return normalise_subtype(value)


def export_blank_df():
    """
    Blank sheet placeholder.
    """

    return pd.DataFrame({"No data": ["No data available for this subtype/technique"]})


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
    Keep SAXS data for one subtype.

    Prefer dermis/wound regions where available, but keep sample-level data
    if that is the only available region.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    subtype_col = "subtype" if "subtype" in d.columns else "Subtype"

    if subtype_col not in d.columns:
        return pd.DataFrame()

    d["_export_subtype"] = d[subtype_col].apply(export_normalise_subtype)
    d = d[d["_export_subtype"] == subtype].copy()

    if d.empty:
        return pd.DataFrame()

    if "region" in d.columns:
        region_lower = d["region"].astype(str).str.lower()
        non_sample = d[region_lower != "sample"].copy()

        if not non_sample.empty:
            d = non_sample

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

def export_filter_ni_region(summary_df, binned_df, subtype, region_keys):
    """
    Filter NI summary/binned outputs by subtype and RegionKey.
    """

    outputs = []

    region_keys = {str(r).strip().lower() for r in region_keys}

    if summary_df is not None and not summary_df.empty:
        d = summary_df.copy()

        if "Subtype" in d.columns and "RegionKey" in d.columns:
            d["_export_subtype"] = d["Subtype"].apply(export_normalise_subtype)
            d["_region_key"] = d["RegionKey"].astype(str).str.strip().str.lower()

            d = d[
                (d["_export_subtype"] == subtype)
                & (d["_region_key"].isin(region_keys))
            ].copy()

            d = d.drop(columns=["_export_subtype", "_region_key"], errors="ignore")

            if not d.empty:
                d.insert(0, "ExportLevel", "SampleSummary")
                outputs.append(d)

    if binned_df is not None and not binned_df.empty:
        d = binned_df.copy()

        if "Subtype" in d.columns and "RegionKey" in d.columns:
            d["_export_subtype"] = d["Subtype"].apply(export_normalise_subtype)
            d["_region_key"] = d["RegionKey"].astype(str).str.strip().str.lower()

            d = d[
                (d["_export_subtype"] == subtype)
                & (d["_region_key"].isin(region_keys))
            ].copy()

            d = d.drop(columns=["_export_subtype", "_region_key"], errors="ignore")

            if not d.empty:
                d.insert(0, "ExportLevel", "Binned")
                outputs.append(d)

    return pd.concat(outputs, ignore_index=True, sort=False) if outputs else pd.DataFrame()

def export_filter_raw_ni_region(ni_points_df, subtype, region_keys, layer=None):
    """
    Filter raw NI point-level data by subtype and RegionKey.

    For line scan, layer can be left as None if you want all layers:
        glass, subcut, dermis/wound, epidermis.

    For dermis-only line scan, pass layer='dermis'.
    """

    if ni_points_df is None or ni_points_df.empty:
        return pd.DataFrame()

    d = ni_points_df.copy()

    if "Subtype" not in d.columns or "RegionKey" not in d.columns:
        return pd.DataFrame()

    region_keys = {str(r).strip().lower() for r in region_keys}

    d["_export_subtype"] = d["Subtype"].apply(export_normalise_subtype)
    d["_region_key"] = d["RegionKey"].astype(str).str.strip().str.lower()

    d = d[
        (d["_export_subtype"] == subtype)
        & (d["_region_key"].isin(region_keys))
    ].copy()

    d = d.drop(columns=["_export_subtype", "_region_key"], errors="ignore")

    if layer is not None and "layer" in d.columns:
        d = d[d["layer"].astype(str).str.lower() == str(layer).lower()].copy()

    return d.reset_index(drop=True)

def export_RamanRawSpectraToDataFrame(
    raman_dict,
    subtype,
    regions_to_export=("Dermis", "Wound"),
    spectral_regions=("FP", "EXT"),
):
    """
    Export raw/treated Raman spectral values from selected anatomical regions.
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
                    try:
                        wave = np.asarray(sc.spectral_axis, dtype=float)
                        intensity = np.asarray(sc.spectral_data, dtype=float)
                    except Exception:
                        continue

                    if wave.size == 0 or intensity.size == 0:
                        continue

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

        ni_linescan_df = export_filter_ni_region(
            ni_summary_df,
            ni_binned_df,
            subtype,
            region_keys=["line scan"],
        )

        raw_ni_linescan_df = export_filter_raw_ni_region(
            ni_points_df,
            subtype,
            region_keys=["line scan"],
            layer=None,
        )

        ni_grid_df = export_filter_ni_region(
            ni_summary_df,
            ni_binned_df,
            subtype,
            region_keys=["upper dermis", "lower dermis"],
        )

        raw_ni_grid_df = export_filter_raw_ni_region(
            ni_points_df,
            subtype,
            region_keys=["upper dermis", "lower dermis"],
            layer=None,
        )

        diagnostics_df = export_build_diagnostics(
            subtype=subtype,
            raman_df=raman_df,
            raw_raman_df=raw_raman_df,
            cell_df=cell_df,
            saxs_df=saxs_df,
            ni_linescan_df=ni_linescan_df,
            raw_ni_linescan_df=raw_ni_linescan_df,
            ni_grid_df=ni_grid_df,
            raw_ni_grid_df=raw_ni_grid_df,
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

            export_safe_sheet_df(ni_linescan_df).to_excel(
                writer,
                sheet_name="NI_LineScan",
                index=False,
            )

            export_safe_sheet_df(raw_ni_linescan_df).to_excel(
                writer,
                sheet_name="RawNI_LineScan",
                index=False,
            )

            export_safe_sheet_df(ni_grid_df).to_excel(
                writer,
                sheet_name="NI_Grid",
                index=False,
            )

            export_safe_sheet_df(raw_ni_grid_df).to_excel(
                writer,
                sheet_name="RawNI_Grid",
                index=False,
            )

            export_safe_sheet_df(diagnostics_df).to_excel(
                writer,
                sheet_name="Diagnostics",
                index=False,
            )

        print(f"Exported: {out_path}")

def export_build_diagnostics(
    subtype,
    raman_df,
    raw_raman_df,
    cell_df,
    saxs_df,
    ni_linescan_df,
    raw_ni_linescan_df,
    ni_grid_df,
    raw_ni_grid_df,
):
    """
    Build a small diagnostics table for each exported workbook.
    """

    tables = {
        "Raman": raman_df,
        "RawRaman": raw_raman_df,
        "Cell": cell_df,
        "SAXS": saxs_df,
        "NI_LineScan": ni_linescan_df,
        "RawNI_LineScan": raw_ni_linescan_df,
        "NI_Grid": ni_grid_df,
        "RawNI_Grid": raw_ni_grid_df,
    }

    rows = []

    for sheet, df in tables.items():
        rows.append({
            "Subtype": subtype,
            "Sheet": sheet,
            "Rows": 0 if df is None else len(df),
            "Columns": 0 if df is None else len(df.columns),
            "HasData": df is not None and not df.empty,
        })

    return pd.DataFrame(rows)


# ============================================================
# ===========================================================     REPORT     =============================================================
# ============================================================

def REPORT_line(title, char="="):
    print("\n" + char * 70)
    print(title)
    print(char * 70)


def REPORT_normalised_group_members(group_subtypes):
    return {normalise_subtype(s) for s in group_subtypes}


def REPORT_format_count_dict(count_dict):
    if not count_dict:
        return "none"

    return ", ".join(
        f"{k}: {v}"
        for k, v in count_dict.items()
    )


def REPORT_print_ni_summary(ni_dict, ni_points_df, groups):
    """
    Print a clean NI processing summary by biological group.
    """

    REPORT_line("NANOINDENTATION")

    if ni_points_df is None or ni_points_df.empty:
        print("No nanoindentation data found.")
        return

    d = ni_points_df.copy()
    d["SubtypeNorm"] = d["Subtype"].apply(normalise_subtype)
    d["RegionKeyNorm"] = d["RegionKey"].astype(str).str.strip().str.lower()

    missing_rows = []

    for sample_key, sample in ni_dict.items():
        if not isinstance(sample, dict):
            continue

        subtype = normalise_subtype(sample.get("TYPE", ""))

        for miss in sample.get("_missing_ni_csvs", []):
            missing_rows.append({
                "SampleKey": sample_key,
                "Subtype": subtype,
                "RegionKey": miss.get("RegionKey", ""),
                "Path": miss.get("Path", ""),
            })

    missing_df = pd.DataFrame(missing_rows)

    for group_name, group_subtypes in groups.items():
        group_set = REPORT_normalised_group_members(group_subtypes)

        dg = d[d["SubtypeNorm"].isin(group_set)].copy()

        if missing_df.empty:
            mg = pd.DataFrame()
        else:
            mg = missing_df[missing_df["Subtype"].isin(group_set)].copy()

        print(f"\n{group_name}")

        if dg.empty and mg.empty:
            print("  No NI data available.")
            continue

        # Line scans
        line_df = dg[dg["RegionKeyNorm"] == "line scan"].copy()
        n_line_scans = line_df["SampleKey"].nunique() if not line_df.empty else 0

        if not line_df.empty and "layer" in line_df.columns:
            layer_counts = (
                line_df
                .dropna(subset=["layer"])
                .groupby("layer")["SampleKey"]
                .nunique()
                .to_dict()
            )
        else:
            layer_counts = {}

        n_line_missing = 0
        if not mg.empty:
            n_line_missing = int((mg["RegionKey"].astype(str).str.lower() == "line scan").sum())

        print(
            f"  Line scans: {n_line_scans} found, {n_line_missing} missing"
        )
        print(
            f"  Line scan layer labels: {REPORT_format_count_dict(layer_counts)}"
        )

        # Grids
        upper_df = dg[dg["RegionKeyNorm"] == "upper dermis"].copy()
        lower_df = dg[dg["RegionKeyNorm"] == "lower dermis"].copy()

        n_upper = upper_df["SampleKey"].nunique() if not upper_df.empty else 0
        n_lower = lower_df["SampleKey"].nunique() if not lower_df.empty else 0

        n_upper_missing = 0
        n_lower_missing = 0

        if not mg.empty:
            reg = mg["RegionKey"].astype(str).str.lower()
            n_upper_missing = int((reg == "upper dermis").sum())
            n_lower_missing = int((reg == "lower dermis").sum())

        print(
            f"  Upper grids: {n_upper} found, {n_upper_missing} missing"
        )
        print(
            f"  Lower grids: {n_lower} found, {n_lower_missing} missing"
        )


def REPORT_print_saxs_summary(saxs_tidy_df, groups):
    """
    Print a clean SAXS processing summary by biological group.
    """

    REPORT_line("SAXS")

    if saxs_tidy_df is None or saxs_tidy_df.empty:
        print("No SAXS data found.")
        return

    d = saxs_tidy_df.copy()
    d["SubtypeNorm"] = d["subtype"].apply(normalise_subtype)

    for group_name, group_subtypes in groups.items():
        group_set = REPORT_normalised_group_members(group_subtypes)
        dg = d[d["SubtypeNorm"].isin(group_set)].copy()

        print(f"\n{group_name}")

        if dg.empty:
            print("  No SAXS data available.")
            continue

        iq = dg[dg["source"] == "IQ"].copy()
        ichi = dg[dg["source"] == "IChi"].copy()

        n_iq_files = iq["Filenumber"].nunique() if not iq.empty else 0
        n_ichi_files = ichi["Filenumber"].nunique() if not ichi.empty else 0

        print(f"  IQ files found: {n_iq_files}")
        print(f"  IChi files found: {n_ichi_files}")

        if "region" in dg.columns:
            region_counts = (
                dg.groupby("region")["Filenumber"]
                .nunique()
                .to_dict()
            )
            print(f"  Regions: {REPORT_format_count_dict(region_counts)}")


def REPORT_print_raman_summary(raman_dict, raman_summary_df, groups):
    """
    Print a clean Raman processing summary by biological group.
    """

    REPORT_line("RAMAN")

    if raman_dict is None or len(raman_dict) == 0:
        print("No Raman dictionary found.")
        return

    rows = []

    for sample_num, sample_data in raman_dict.items():
        if not isinstance(sample_data, dict):
            continue

        subtype = normalise_subtype(sample_data.get("Subtype", ""))

        fp_total = sample_data.get("FP_Spectra_Treated")
        ext_total = sample_data.get("EXT_Spectra_Treated")
        fp_dermis = sample_data.get("FP_Spectra_Treated_Dermis")
        ext_dermis = sample_data.get("EXT_Spectra_Treated_Dermis")

        rows.append({
            "Sample": str(sample_num),
            "Subtype": subtype,
            "FP_found": isinstance(fp_total, dict) and len(fp_total) > 0,
            "EXT_found": isinstance(ext_total, dict) and len(ext_total) > 0,
            "FP_points_total": len(fp_total) if isinstance(fp_total, dict) else 0,
            "EXT_points_total": len(ext_total) if isinstance(ext_total, dict) else 0,
            "FP_points_dermis": len(fp_dermis) if isinstance(fp_dermis, dict) else 0,
            "EXT_points_dermis": len(ext_dermis) if isinstance(ext_dermis, dict) else 0,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print("No Raman spectra found.")
        return

    for group_name, group_subtypes in groups.items():
        group_set = REPORT_normalised_group_members(group_subtypes)
        dg = df[df["Subtype"].isin(group_set)].copy()

        print(f"\n{group_name}")

        if dg.empty:
            print("  No Raman data available.")
            continue

        n_samples = dg["Sample"].nunique()
        n_fp = int(dg["FP_found"].sum())
        n_ext = int(dg["EXT_found"].sum())

        fp_points = dg["FP_points_dermis"].dropna().astype(int).tolist()
        ext_points = dg["EXT_points_dermis"].dropna().astype(int).tolist()

        print(f"  Samples with FP spectra: {n_fp}/{n_samples}")
        print(f"  Samples with EXT spectra: {n_ext}/{n_samples}")

        if fp_points:
            print(
                f"  FP dermis line-scan points: "
                f"min {min(fp_points)}, median {int(np.median(fp_points))}, max {max(fp_points)}"
            )

        if ext_points:
            print(
                f"  EXT dermis line-scan points: "
                f"min {min(ext_points)}, median {int(np.median(ext_points))}, max {max(ext_points)}"
            )


def REPORT_print_cell_summary(cell_points_df, cell_summary_df, groups, type_label=None):
    """
    Print a clean cell-data summary.
    """

    REPORT_line("CELL DATA")

    if cell_points_df is None or cell_points_df.empty:
        if type_label is None:
            print("Cell data not available.")
        else:
            print(f"Cell data not available for Type = {type_label}.")
        return

    d = cell_points_df.copy()
    d["SubtypeNorm"] = d["Subtype"].apply(normalise_subtype)

    for group_name, group_subtypes in groups.items():
        group_set = REPORT_normalised_group_members(group_subtypes)
        dg = d[d["SubtypeNorm"].isin(group_set)].copy()

        print(f"\n{group_name}")

        if dg.empty:
            print("  No cell data available.")
            continue

        n_values = len(dg)
        region_counts = dg.groupby("Region").size().to_dict()

        print(f"  Cell values found: {n_values}")
        print(f"  Regions: {REPORT_format_count_dict(region_counts)}")






