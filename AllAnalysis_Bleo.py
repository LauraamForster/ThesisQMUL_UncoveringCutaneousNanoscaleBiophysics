#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:10:38 2026

@author: lauraforster
"""

# =============================================================================
# Multitech analysis script
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from matplotlib.colors import to_rgb
from matplotlib.patches import Wedge, Patch
import os 
from scipy.stats import ttest_ind
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 20,
})

start_time = time.time()

# ====================================================================================================================================================
# ====================================================================================================================================================
# ====================================================================================================================================================

# =============================================================================
# Input setup
# =============================================================================

DATA_ROOT = Path("/Volumes/LauraDrive/Multitech_Export_bleo")

SUBTYPES = [
    "control", 
    "pbs",
    "2w",
    "4w",
    "pbsmet",
    "bmmet",
    "pbsokn",
    "4wokn",
]

SHEETS = [
    "Raman",
    "RawRaman",
    "Cell",
    "SAXS",
    "NI_LineScan",
    "RawNI_LineScan",
    "NI_Grid",
    "RawNI_Grid",
    # Old export format

   "Nanoindentation",
   "RawNanoindentation",
   
    "Diagnostics",
]

SUBTYPE_ORDER = [
    "control", 
    "pbs",
    "2w",
    "4w",
    "pbsmet",
    "bmmet",
    "pbsokn",
    "4wokn",
]

SUBTYPE_LABELS = {
    # Bleo group 1
    "control": "Unwounded",
    "pbs": "PBS",
    "2w": "2W",
    "4w": "4W",

    # Bleo group 2
    "pbsmet": "PBS + Met",
    "bmmet": "4W + Met",

    # Bleo group 3
    "pbsokn": "PBS + OKN",
    "4wokn": "4W + OKN",
}

NI_COLOURS = {
    # Bleo group 1
    "control": "black",

    "pbs": "grey",
    "2w": "tomato",
    "4w": "royalblue",

    # Bleo group 2
    "pbsmet": "lightgrey",
    "bmmet": "mediumseagreen",

    # Bleo group 3
    "pbsokn": "darkgrey",
    "4wokn": "darkorchid",
}

NI_LABELS = SUBTYPE_LABELS

NI_MARKERS = {
    "control": "o",
    "pbs": "o",
    "2w": "s",
    "4w": "^",

    "pbsmet": "o",
    "bmmet": "s",

    "pbsokn": "o",
    "4wokn": "^",

    "ac": "D",
    "cl": "P",
    "ts": "X",
    "vh": "v",
}

SUBTYPE_ALIASES = {
    "control": "control",
    "ctrl": "control",
    "ct": "control",
    "unwounded": "control",
    
    "pbs": "pbs",
    "pbs control": "pbs",

    "2w": "2w",
    "2 week": "2w",
    "2 weeks": "2w",
    "2w bleo": "2w",

    "4w": "4w",
    "4 week": "4w",
    "4 weeks": "4w",
    "4w bleo": "4w",
    "4w 3r": "4w",
    "4w3r": "4w",
    "4w 5r": "4w",
    "4w5r": "4w",

    "pbsmet": "pbsmet",
    "pbs met": "pbsmet",
    "pbs_met": "pbsmet",
    "pbs metformin": "pbsmet",

    "bmmet": "bmmet",
    "bm met": "bmmet",
    "bm_met": "bmmet",
    "bleomycin metformin": "bmmet",
    "bleomycinmetformin": "bmmet",

    "pbsokn": "pbsokn",
    "pbs okn": "pbsokn",
    "pbs_okn": "pbsokn",

    "4wokn": "4wokn",
    "4w okn": "4wokn",
    "4w_okn": "4wokn",
}

CELL_REGION_ORDER_4 = ["dermis_sub", "dermis_epi", "wound_sub", "wound_epi"]
CELL_REGION_LABELS = {
    "dermis_sub": "Lower dermis",
    "dermis_epi": "Upper dermis",
    "wound_sub": "Lower wound",
    "wound_epi": "Upper wound",
    "dermis": "Dermis",
    "wound": "Wound",
    "lower_shift": "Lower wound vs lower dermis",
    "upper_shift": "Upper wound vs upper dermis",
    "dermis_epi_shift": "Upper dermis vs lower dermis",
    "wound_epi_shift": "Upper wound vs lower wound",
}

CELL_VALUE_COL = "Fibroblasts_per_mm2"

SAXS_VALUE_COL = "mean"

SAXS_REGION_ORDER_2 = ["dermis", "wound"]

SAXS_REGION_ORDER_4 = [
    "dermis_sub",
    "dermis_epi",
    "wound_sub",
    "wound_epi",
]

SAXS_REGION_LABELS = {
    "dermis": "Dermis",
    "wound": "Wound",
    "dermis_sub": "Lower dermis",
    "dermis_epi": "Upper dermis",
    "wound_sub": "Lower wound",
    "wound_epi": "Upper wound",
}

SAXS_PARAMETER_LABELS = {
    "curvearea_norm": "Normalised total collagen intensity",
    "SAXS_norm": "Normalised SAXS intensity",
    "D_period": "D-period",
    "wa_moment": "Weighted average peak width",
}

SAXS_DERMIS_REGIONS = ["dermis_sub", "dermis_epi"]
SAXS_WOUND_REGIONS = ["wound_sub", "wound_epi"]

SAXS_DPERIOD_YLIM = (64.5, 65.2)

SAXS_DPERIOD_PARAM = "D_period"
SAXS_FWHM_PARAM = "peak_width_q"   # maps to peak_width from SAXS export

SAXS_WA_PARAM = "wa_moment"

SAXS_PEAK_POSITION_PARAM = "peak_position_folded"
SAXS_PEAK_RSQ_PARAM = "rsq"

SAXS_PEAK_REGION_ORDER_2 = ["dermis", "wound"]
SAXS_PEAK_REGION_ORDER_4 = ["dermis_sub", "dermis_epi", "wound_sub", "wound_epi"]

SAXS_PEAK_SPREAD_PARAM = "peak_position_folded"
SAXS_PEAK_SPREAD_RSQ_PARAM = "rsq"

RAMAN_PEAK_REGIONS = [
    ("CH2CH3_LEFT_1220_1330", (1220, 1300)),
    ("CH2CH3_RIGHT_1330_1380", (1310, 1380)),
    ("AmideIII_1410_1500", (1410, 1500)),
    ("AmideI_LEFT_1530_1590", (1530, 1589)),
    ("AmideI_MIDDLE_1590_1635", (1590, 1634)),
    ("AmideI_RIGHT_1635_1700", (1635, 1700)),
]

RAMAN_PEAK_REGIONS = [
    (" ", (1220, 1300)),
    (" ", (1310, 1380)),
    (" ", (1410, 1500)),
    (" ", (1530, 1589)),
    (" ", (1590, 1634)),
    (" ", (1635, 1700)),
]

RAMAN_SUBTYPES = ["CT", "PBS", "D7", "D10", "D14", "D21"]

RAMAN_COLOURS = {
    "CT": "black",
    "CT": "grey",
    "PBS": "blue",
    "D7": "tomato",
    "D10": "mediumorchid",
    "D14": "royalblue",
    "D21": "mediumseagreen",
}

RAMAN_SUBTYPES = SUBTYPE_ORDER.copy()

RAMAN_COLOURS = NI_COLOURS.copy()

RAMAN_LINESTYLES = {
    "CT": "-",
    "pbs": "-",
    "2w": "--",
    "4w": ":",

    "pbsmet": "-",
    "bmmet": "--",

    "pbsokn": "-",
    "4wokn": "--",
}

# =============================================================================
# Excel read-in helpers
# =============================================================================

def clean_excel_df(df):
    """
    Clean a dataframe loaded from Excel.
    Removes blank placeholder sheets and normalises column names lightly.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    df.columns = [str(c).strip() for c in df.columns]

    # Remove completely empty rows/columns
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    # Blank placeholder sheet from export
    if list(df.columns) == ["No data"] and df.empty:
        return pd.DataFrame()

    if "No data" in df.columns and len(df.columns) == 1:
        return pd.DataFrame()

    return df.reset_index(drop=True)

def read_one_workbook(path, subtype, sheets=SHEETS):
    """
    Read one subtype workbook.

    Handles both old and new export formats.

    Old format:
        Nanoindentation, RawNanoindentation

    New format:
        NI_LineScan, RawNI_LineScan, NI_Grid, RawNI_Grid
    """

    workbook_data = {}

    if not path.exists():
        print(f"Missing workbook: {path}")

        for sheet in sheets:
            workbook_data[sheet] = pd.DataFrame()

        return workbook_data

    available_sheets = pd.ExcelFile(path).sheet_names
    available_set = set(available_sheets)

    optional_ni_sheets = {
        "NI_LineScan",
        "RawNI_LineScan",
        "NI_Grid",
        "RawNI_Grid",
        "Nanoindentation",
        "RawNanoindentation",
        "Diagnostics",
    }

    for sheet in sheets:
        if sheet not in available_set:
            # Old/new NI sheet names are optional because formats differ.
            if sheet not in optional_ni_sheets:
                print(f"Missing sheet '{sheet}' in {path.name}")

            workbook_data[sheet] = pd.DataFrame()
            continue

        df = pd.read_excel(path, sheet_name=sheet)
        df = clean_excel_df(df)

        if not df.empty:
            df["ExportSubtype"] = subtype

        workbook_data[sheet] = df

    return workbook_data

def read_multitech_workbooks(data_root=DATA_ROOT, subtypes=SUBTYPES, sheets=SHEETS):
    """
    Read all subtype workbooks.

    Returns
    -------
    data : dict
        data[subtype][sheet] = dataframe

    combined : dict
        combined[sheet] = all subtypes concatenated
    """

    data_root = Path(data_root)

    data = {}

    for subtype in subtypes:
        path = data_root / f"{subtype}.xlsx"
        data[subtype] = read_one_workbook(path, subtype, sheets=sheets)

    combined = {}

    for sheet in sheets:
        dfs = []

        for subtype in subtypes:
            df = data[subtype].get(sheet, pd.DataFrame())

            if df is not None and not df.empty:
                dfs.append(df)

        combined[sheet] = (
            pd.concat(dfs, ignore_index=True, sort=False)
            if dfs
            else pd.DataFrame()
        )

    return data, combined

def split_loaded_tables(
    raman_df,
    raw_raman_df,
    cell_df,
    saxs_df,
    ni_df,
    raw_ni_df,
):
    """
    Split the loaded Excel sheets into clean analysis-ready tables.
    """

    tables = {}

    # -------------------------
    # Raman
    # -------------------------
    tables["raman_summary"] = raman_df.copy() if raman_df is not None else pd.DataFrame()
    tables["raman_raw"] = raw_raman_df.copy() if raw_raman_df is not None else pd.DataFrame()

    # -------------------------
    # Cell data
    # -------------------------
    if cell_df is not None and not cell_df.empty:
        tables["cell_raw"] = cell_df[cell_df["ExportLevel"] == "RawValues"].copy()
        tables["cell_summary"] = cell_df[cell_df["ExportLevel"] == "Summary"].copy()
    else:
        tables["cell_raw"] = pd.DataFrame()
        tables["cell_summary"] = pd.DataFrame()

    # -------------------------
    # SAXS
    # -------------------------
    if saxs_df is not None and not saxs_df.empty:
        tables["saxs_sample"] = saxs_df[saxs_df["ExportLevel"] == "PerSample"].copy()
        tables["saxs_points"] = saxs_df[saxs_df["ExportLevel"] == "PointLevel"].copy()
    else:
        tables["saxs_sample"] = pd.DataFrame()
        tables["saxs_points"] = pd.DataFrame()

    # -------------------------
    # Nanoindentation
    # -------------------------
    if ni_df is not None and not ni_df.empty:
        tables["ni_summary"] = ni_df[ni_df["ExportLevel"] == "SampleSummary"].copy()
        tables["ni_binned"] = ni_df[ni_df["ExportLevel"] == "Binned"].copy()
    else:
        tables["ni_summary"] = pd.DataFrame()
        tables["ni_binned"] = pd.DataFrame()
    # -------------------------
    # Raw nanoindentation
    # -------------------------
    tables["ni_raw"] = raw_ni_df.copy() if raw_ni_df is not None else pd.DataFrame()
    return tables

def normalise_subtype(value):
    if pd.isna(value):
        return ""

    key = str(value).strip().lower()
    key = key.replace("_", " ")
    key = key.replace("-", " ")
    key = " ".join(key.split())
    compact = key.replace(" ", "")

    mapping = {
        # -------------------------
        # Control / unwounded
        # -------------------------
        "control": "control",
        "ct": "control",
        "ctrl": "control",
        "unwounded": "control",
        "un wounded": "control",

        # -------------------------
        # Bleomycin baseline
        # -------------------------
        "pbs": "pbs",
        "pbscontrol": "pbs",
        "pbs control": "pbs",

        "2w": "2w",
        "2week": "2w",
        "2weeks": "2w",
        "2wbleo": "2w",
        "2wbleomycin": "2w",

        "4w": "4w",
        "4week": "4w",
        "4weeks": "4w",
        "4wbleo": "4w",
        "4wbleomycin": "4w",
        "4w3r": "4w",
        "4w5r": "4w",

        # -------------------------
        # Metformin
        # -------------------------
        "pbsmet": "pbsmet",
        "pbsmetformin": "pbsmet",

        "bmmet": "bmmet",
        "bmmetformin": "bmmet",
        "bleomycinmetformin": "bmmet",
        "bleomycinsmetformin": "bmmet",

        # -------------------------
        # OKN
        # -------------------------
        "pbsokn": "pbsokn",
        "4wokn": "4wokn",
    }

    return mapping.get(key, mapping.get(compact, compact))

def subtype_label(value):
    """
    Convert any subtype value to display label:
        Unwounded, PW4, PW7, PW10, PW14, PW21
    """

    clean = normalise_subtype(value)
    return SUBTYPE_LABELS.get(clean, str(value))

def add_clean_subtype(df, source_col=None, out_col="SubtypeClean"):
    """
    Add a shared subtype column.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    if source_col is None:
        if "ExportSubtype" in d.columns:
            source_col = "ExportSubtype"
        elif "Subtype" in d.columns:
            source_col = "Subtype"
        elif "subtype" in d.columns:
            source_col = "subtype"
        else:
            return d

    d[out_col] = d[source_col].apply(normalise_subtype)

    return d

# =============================================================================
# Quick checks
# =============================================================================

def preview_loaded_data(combined_data):
    for sheet, df in combined_data.items():
        print("\n" + "=" * 80)
        print(sheet)
        print("=" * 80)

        if df.empty:
            print("No data")
            continue

        print(df.head())
        print("\nColumns:")
        print(list(df.columns))

# =============================================================================
# Nanoindentation plotting helpers

def prepare_ni_line_data(
    ni_binned,
    variable,
    subtype_order=SUBTYPE_ORDER,
    convert_modulus_to_kpa=True,
):
    """
    Prepare binned NI data for line-scan plots.

    Uses:
        x = normalised bin position
        y = variable_mean

    Example variables:
        mod_Hertz
        tau_Visco
    """

    if ni_binned is None or ni_binned.empty:
        return pd.DataFrame()

    value_col = f"{variable}_mean"

    if value_col not in ni_binned.columns:
        raise KeyError(f"Column not found in ni_binned: {value_col}")

    d = ni_binned.copy()
    d = add_clean_subtype(d)

    d = d[d["SubtypeClean"].isin(subtype_order)].copy()

    d["Bin"] = pd.to_numeric(d["Bin"], errors="coerce")
    d["BinTotal"] = pd.to_numeric(d["BinTotal"], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    d = d.dropna(subset=["Bin", "BinTotal", value_col])

    # Bin centre on 0–100% scale
    d["NormalisedPosition"] = ((d["Bin"] - 0.5) / d["BinTotal"]) * 100

    d["Value"] = d[value_col]

    if convert_modulus_to_kpa and variable in {
        "mod_Hertz",
        "mod_OP",
        "E0_Visco",
        "Einf_Visco",
        "G0_Visco",
        "G1_Visco",
        "Eff_file",
        "mod_file",
    }:
        d["Value"] = d["Value"] / 1000

    d["SubtypeClean"] = pd.Categorical(
        d["SubtypeClean"],
        categories=subtype_order,
        ordered=True,
    )

    return d.sort_values(["SubtypeClean", "Sample", "Bin"]).reset_index(drop=True)

def summarise_ni_line_data(df):
    """
    Mean ± confidence interval per subtype/bin.

    CI is calculated as:
        mean ± 1.96 * SEM
    """

    if df is None or df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(["SubtypeClean", "Bin", "NormalisedPosition"], observed=True)
        ["Value"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )

    summary["sem"] = summary["std"] / np.sqrt(summary["n"])
    summary["ci95"] = 1.96 * summary["sem"]

    return summary

def summarise_ni_raw_line_data(df, n_position_bins=20):
    """
    Summarise raw NI data by normalised-position bins.

    This avoids plotting thousands of raw points as a jagged mean line.
    """

    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    d = df.copy()
    d = d.dropna(subset=["NormalisedPosition", "Value"])

    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    bins = np.linspace(0, 100, n_position_bins + 1)
    labels = 0.5 * (bins[:-1] + bins[1:])

    d["PositionBin"] = pd.cut(
        d["NormalisedPosition"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    d["PositionBin"] = d["PositionBin"].astype(float)

    sample_col = "SampleKey" if "SampleKey" in d.columns else "Sample"

    sample_summary = (
        d.groupby(["SubtypeClean", sample_col, "PositionBin"], observed=True)["Value"]
        .mean()
        .reset_index()
        .rename(columns={"Value": "SampleMean"})
    )

    subtype_summary = (
        sample_summary
        .groupby(["SubtypeClean", "PositionBin"], observed=True)["SampleMean"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )

    subtype_summary["sem"] = subtype_summary["std"] / np.sqrt(subtype_summary["n"])
    subtype_summary["ci95"] = 1.96 * subtype_summary["sem"]

    return sample_summary, subtype_summary

def _facet_setup(n_panels, ncols=2, width=5.0, height=4.0, sharey=True):
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(width * ncols, height * nrows),
        sharex=True,
        sharey=sharey,
    )

    axes = np.atleast_1d(axes).ravel()

    for ax in axes[n_panels:]:
        ax.axis("off")

    return fig, axes

def plot_ni_linescan_by_subtype(
    ni_binned,
    variable,
    ylabel,
    title=None,
    subtype_order=SUBTYPE_ORDER,
    colours=NI_COLOURS,
    labels=NI_LABELS,
    ylim=None,
    ncols=2,
    show_sample_lines=True,
    sample_alpha=0.25,
    ci_alpha=0.20,
):
    """
    Plot NI binned line scan as normalised position vs variable.

    One panel per subtype.
    Shows:
        - individual sample lines
        - mean line
        - 95% CI band
    """

    df = prepare_ni_line_data(
        ni_binned=ni_binned,
        variable=variable,
    )

    if df.empty:
        print(f"No NI binned data available for {variable}")
        return None, None

    summary = summarise_ni_line_data(df)

    present_subtypes = [
        st for st in subtype_order
        if st in set(df["SubtypeClean"].astype(str))
    ]

    fig, axes = _facet_setup(
        n_panels=len(present_subtypes),
        ncols=ncols,
        sharey=True,
    )

    for ax, subtype in zip(axes, present_subtypes):
        colour = colours.get(subtype, "grey")
        label = labels.get(subtype, subtype)

        d_sub = df[df["SubtypeClean"].astype(str) == subtype].copy()
        s_sub = summary[summary["SubtypeClean"].astype(str) == subtype].copy()

        if show_sample_lines:
            for _, d_sample in d_sub.groupby("Sample"):
                d_sample = d_sample.sort_values("NormalisedPosition")
                ax.plot(
                    d_sample["NormalisedPosition"],
                    d_sample["Value"],
                    color=colour,
                    alpha=sample_alpha,
                    linewidth=1.0,
                )

        s_sub = s_sub.sort_values("NormalisedPosition")

        x = s_sub["NormalisedPosition"].to_numpy(dtype=float)
        y = s_sub["mean"].to_numpy(dtype=float)
        ci = s_sub["ci95"].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            color=colour,
            linewidth=2.5,
            label=label,
        )

        ax.fill_between(
            x,
            y - ci,
            y + ci,
            color=colour,
            alpha=ci_alpha,
            linewidth=0,
        )

        ax.set_title(label)
        ax.set_xlabel("Normalised position through dermis (%)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 100)

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.grid(False)

    fig.suptitle(title or f"Nanoindentation {variable} across normalised position", y=0.995)
    plt.tight_layout()
    plt.show()

    return fig, axes

def _linear_fit_with_ci(x, y, x_grid=None):
    """
    Simple linear fit with 95% confidence interval.

    Returns
    -------
    x_grid, y_fit, y_lower, y_upper, slope, intercept, r
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if len(x) < 3:
        return None

    if x_grid is None:
        x_grid = np.linspace(np.nanmin(x), np.nanmax(x), 100)

    slope, intercept = np.polyfit(x, y, deg=1)
    y_fit = intercept + slope * x_grid

    y_pred = intercept + slope * x
    residuals = y - y_pred

    n = len(x)
    dof = n - 2

    if dof <= 0:
        return None

    s_err = np.sqrt(np.sum(residuals ** 2) / dof)
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean) ** 2)

    if sxx == 0:
        return None

    se_fit = s_err * np.sqrt(
        1 / n + ((x_grid - x_mean) ** 2 / sxx)
    )

    ci = 1.96 * se_fit

    r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan

    return {
        "x_grid": x_grid,
        "y_fit": y_fit,
        "y_lower": y_fit - ci,
        "y_upper": y_fit + ci,
        "slope": slope,
        "intercept": intercept,
        "r": r,
        "n": n,
    }

def smooth_ni_raw_line_data(
    df,
    window=18,
    grid_size=200,
    min_n=2,
):
    """
    Smooth raw NI data using a sliding position window.

    For each subtype:
      - for each sample
      - for each x-grid position
      - average points within +/- window
      - then average those sample means across samples

    Returns one smoothed mean/CI table.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    sample_col = "SampleKey" if "SampleKey" in df.columns else "Sample"
    xgrid = np.linspace(0, 100, grid_size)

    rows = []

    for subtype, d_sub in df.groupby("SubtypeClean", observed=True):
        sample_arrays = {}

        for sample, d_sample in d_sub.groupby(sample_col):
            x = pd.to_numeric(d_sample["NormalisedPosition"], errors="coerce").to_numpy(float)
            y = pd.to_numeric(d_sample["Value"], errors="coerce").to_numpy(float)

            keep = np.isfinite(x) & np.isfinite(y)

            if np.any(keep):
                sample_arrays[sample] = (x[keep], y[keep])

        for gx in xgrid:
            vals = []

            for sx, sy in sample_arrays.values():
                mask = np.abs(sx - gx) <= window
                v = sy[mask]
                v = v[np.isfinite(v)]

                if v.size:
                    vals.append(np.mean(v))

            vals = np.asarray(vals, float)
            vals = vals[np.isfinite(vals)]

            n = vals.size

            if n < min_n:
                continue

            mean = float(np.mean(vals))

            if n == 1:
                std = np.nan
                ci95 = 0.0
            else:
                std = float(np.std(vals, ddof=1))
                ci95 = 1.96 * std / np.sqrt(n)

            rows.append({
                "SubtypeClean": str(subtype),
                "NormalisedPosition": gx,
                "mean": mean,
                "std": std,
                "n": n,
                "ci95": ci95,
                "lo": mean - ci95,
                "hi": mean + ci95,
            })

    return pd.DataFrame(rows)

def plot_ni_linear_fit_by_subtype(
    ni_raw=None,
    ni_binned=None,
    variable="mod_Hertz",
    ylabel=None,
    title=None,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    colours=NI_COLOURS,
    labels=NI_LABELS,
    ylim=None,
    ncols=2,
    panel=True,
    scatter_alpha=0.25,
    scatter_size=14,
    ci_alpha=0.20,
):
    """
    Plot line of best fit for NI variable vs normalised position.

    subtypes_to_plot:
        None -> plot all subtypes present in subtype_order
        list/tuple -> only plot those subtypes, in subtype_order order

    Preferred input:
        ni_raw = raw point-level NI data

    Fallback:
        ni_binned = binned NI data

    panel=True:
        one panel per subtype, with raw scatter.

    panel=False:
        all subtype fit lines on one plot.
        Raw scatter is hidden automatically to avoid clutter.
    """

    if ni_raw is not None and not ni_raw.empty:
        df = prepare_ni_raw_line_data(
            ni_raw=ni_raw,
            variable=variable,
        )
        data_level = "Raw point-level"
    elif ni_binned is not None and not ni_binned.empty:
        df = prepare_ni_line_data(
            ni_binned=ni_binned,
            variable=variable,
        )
        data_level = "Binned"
    else:
        print("No NI raw or binned data provided.")
        return None, None, pd.DataFrame()

    if df.empty:
        print(f"No NI data available for {variable}")
        return None, None, pd.DataFrame()

    if ylabel is None:
        ylabel = variable

    if subtypes_to_plot is None:
        wanted_subtypes = [str(st).strip() for st in subtype_order]
    else:
        wanted_set = {str(st).strip() for st in subtypes_to_plot}
        wanted_subtypes = [
            str(st).strip()
            for st in subtype_order
            if str(st).strip() in wanted_set
        ]

        wanted_subtypes += [
            str(st).strip()
            for st in subtypes_to_plot
            if str(st).strip() not in wanted_subtypes
        ]

    present_subtypes = [
        st for st in wanted_subtypes
        if st in set(df["SubtypeClean"].astype(str))
    ]

    if not present_subtypes:
        print("No matching NI subtypes to plot.")
        print("Requested:", subtypes_to_plot)
        print("Available:", sorted(df["SubtypeClean"].astype(str).unique()))
        return None, None, pd.DataFrame()

    fit_rows = []

    # ------------------------------------------------------------------
    # Panel mode
    # ------------------------------------------------------------------
    if panel:
        fig, axes = _facet_setup(
            n_panels=len(present_subtypes),
            ncols=ncols,
            sharey=True,
        )

        for ax, subtype in zip(axes, present_subtypes):
            colour = colours.get(subtype, "grey")
            label = labels.get(subtype, subtype)

            d_sub = df[df["SubtypeClean"].astype(str) == subtype].copy()

            x = d_sub["NormalisedPosition"].to_numpy(dtype=float)
            y = d_sub["Value"].to_numpy(dtype=float)

            ax.scatter(
                x,
                y,
                color=colour,
                alpha=scatter_alpha,
                s=scatter_size,
                edgecolors="none",
            )

            fit = _linear_fit_with_ci(x, y, x_grid=np.linspace(0, 100, 100))

            if fit is not None:
                ax.plot(
                    fit["x_grid"],
                    fit["y_fit"],
                    color=colour,
                    linewidth=2.5,
                )

                ax.fill_between(
                    fit["x_grid"],
                    fit["y_lower"],
                    fit["y_upper"],
                    color=colour,
                    alpha=ci_alpha,
                    linewidth=0,
                )

                ax.text(
                    0.03,
                    0.97,
                    f"slope={fit['slope']:.4g}\nr={fit['r']:.3f}\nn={fit['n']}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                )

                fit_rows.append({
                    "Subtype": subtype,
                    "Variable": variable,
                    "DataLevel": data_level,
                    "Slope": fit["slope"],
                    "Intercept": fit["intercept"],
                    "r": fit["r"],
                    "n": fit["n"],
                })

            ax.set_title(label)
            ax.set_xlabel("Normalised position through dermis (%)")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0, 100)

            if ylim is not None:
                ax.set_ylim(*ylim)

            ax.grid(False)

        fig.suptitle(
            title or f"Linear fit: {variable} vs normalised position ({data_level})",
            y=0.995,
        )

        plt.tight_layout()
        plt.show()

        fit_df = pd.DataFrame(fit_rows)

        return fig, axes, fit_df

    # ------------------------------------------------------------------
    # Combined mode
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    for subtype in present_subtypes:
        colour = colours.get(subtype, "grey")
        label = labels.get(subtype, subtype)

        d_sub = df[df["SubtypeClean"].astype(str) == subtype].copy()

        x = d_sub["NormalisedPosition"].to_numpy(dtype=float)
        y = d_sub["Value"].to_numpy(dtype=float)

        fit = _linear_fit_with_ci(x, y, x_grid=np.linspace(0, 100, 100))

        if fit is None:
            continue

        ax.plot(
            fit["x_grid"],
            fit["y_fit"],
            color=colour,
            linewidth=2.5,
            label=f"{label} fit",
        )

        ax.fill_between(
            fit["x_grid"],
            fit["y_lower"],
            fit["y_upper"],
            color=colour,
            alpha=ci_alpha,
            linewidth=0,
        )

        fit_rows.append({
            "Subtype": subtype,
            "Variable": variable,
            "DataLevel": data_level,
            "Slope": fit["slope"],
            "Intercept": fit["intercept"],
            "r": fit["r"],
            "n": fit["n"],
        })

    ax.set_xlabel("Normalised position through dermis (%)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 100)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_title(
        title or f"Linear fit: {variable} vs normalised position ({data_level})"
    )
    ax.legend(frameon=False)
    ax.grid(False)

    plt.tight_layout()
    plt.show()

    fit_df = pd.DataFrame(fit_rows)

    return fig, ax, fit_df

def print_ni_smooth_values_at_positions(
    smooth,
    positions=(0, 50, 100),
    value_col="mean",
    subtype_order=SUBTYPE_ORDER,
    labels=NI_LABELS,
):
    """
    Print smoothed NI values at selected normalised positions.

    Uses the nearest available smoothed x-position.
    """

    if smooth is None or smooth.empty:
        print("No smoothed NI data to print.")
        return pd.DataFrame()

    rows = []

    for subtype in subtype_order:
        st = normalise_subtype(subtype)

        s_sub = smooth[smooth["SubtypeClean"].astype(str) == str(st)].copy()

        if s_sub.empty:
            continue

        s_sub["NormalisedPosition"] = pd.to_numeric(
            s_sub["NormalisedPosition"],
            errors="coerce",
        )

        s_sub[value_col] = pd.to_numeric(
            s_sub[value_col],
            errors="coerce",
        )

        s_sub = s_sub.dropna(subset=["NormalisedPosition", value_col])

        if s_sub.empty:
            continue

        for pos in positions:
            idx = (s_sub["NormalisedPosition"] - pos).abs().idxmin()

            rows.append(
                {
                    "SubtypeClean": st,
                    "Subtype": labels.get(st, st),
                    "RequestedPosition": pos,
                    "ActualPosition": s_sub.loc[idx, "NormalisedPosition"],
                    "Value": s_sub.loc[idx, value_col],
                    "lo": s_sub.loc[idx, "lo"] if "lo" in s_sub.columns else np.nan,
                    "hi": s_sub.loc[idx, "hi"] if "hi" in s_sub.columns else np.nan,
                    "n": s_sub.loc[idx, "n"] if "n" in s_sub.columns else np.nan,
                }
            )

    out = pd.DataFrame(rows)

    if out.empty:
        print("No position values found.")
        return out

    print("\nSmoothed NI values at selected positions:")
    print(
        out[
            [
                "Subtype",
                "RequestedPosition",
                "ActualPosition",
                "Value",
                "lo",
                "hi",
                "n",
            ]
        ].to_string(index=False)
    )

    return out


def plot_ni_raw_linescan_by_subtype_smooth(
    ni_raw,
    variable,
    ylabel,
    title=None,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    colours=NI_COLOURS,
    labels=NI_LABELS,
    ylim=None,
    ncols=2,
    window=18,
    grid_size=200,
    min_n=2,
    panel=True,
    show_sample_lines=True,
    sample_alpha=0.15,
    ci_alpha=0.18,
    print_positions=(0, 50, 100),
):
    """
    Plot raw NI line scan using sliding-window smoothing.

    subtypes_to_plot:
        None -> plot all subtypes present in subtype_order
        list/tuple -> only plot those subtypes, in subtype_order order

    panel=True:
        one panel per subtype, with optional thin sample lines.

    panel=False:
        all subtype mean lines on one plot.
        Raw/sample lines are hidden automatically to avoid clutter.
    """

    df = prepare_ni_raw_line_data(
        ni_raw=ni_raw,
        variable=variable,
    )

    if df.empty:
        print(f"No raw NI data available for {variable}")
        return None, None, pd.DataFrame()

    smooth = smooth_ni_raw_line_data(
        df,
        window=window,
        grid_size=grid_size,
        min_n=min_n,
    )

    if smooth.empty:
        print(f"No smoothed NI curve could be created for {variable}")
        return None, None, pd.DataFrame()
    
    if print_positions is not None:
        ni_position_values = print_ni_smooth_values_at_positions(
            smooth=smooth,
            positions=print_positions,
            value_col="mean",
            subtype_order=subtypes_to_plot if subtypes_to_plot is not None else subtype_order,
            labels=labels,
        )
    else:
        ni_position_values = pd.DataFrame()

    if subtypes_to_plot is None:
        wanted_subtypes = [str(st).strip() for st in subtype_order]
    else:
        wanted_set = {str(st).strip() for st in subtypes_to_plot}
        wanted_subtypes = [
            str(st).strip()
            for st in subtype_order
            if str(st).strip() in wanted_set
        ]

        # Keep any requested subtypes that are not in subtype_order, at the end
        wanted_subtypes += [
            str(st).strip()
            for st in subtypes_to_plot
            if str(st).strip() not in wanted_subtypes
        ]

    present_subtypes = [
        st for st in wanted_subtypes
        if st in set(df["SubtypeClean"].astype(str))
    ]

    if not present_subtypes:
        print("No matching NI subtypes to plot.")
        print("Requested:", subtypes_to_plot)
        print("Available:", sorted(df["SubtypeClean"].astype(str).unique()))
        return None, None, smooth

    sample_col = "SampleKey" if "SampleKey" in df.columns else "Sample"

    # ------------------------------------------------------------------
    # Panel mode
    # ------------------------------------------------------------------
    if panel:
        fig, axes = _facet_setup(
            n_panels=len(present_subtypes),
            ncols=ncols,
            sharey=True,
        )

        for ax, subtype in zip(axes, present_subtypes):
            colour = colours.get(subtype, "grey")
            label = labels.get(subtype, subtype)

            d_sub = df[df["SubtypeClean"].astype(str) == subtype].copy()
            s_sub = smooth[smooth["SubtypeClean"].astype(str) == subtype].copy()

            if show_sample_lines:
                for _, d_sample in d_sub.groupby(sample_col):
                    d_sample = d_sample.sort_values("NormalisedPosition")

                    # ax.plot(
                    #     d_sample["NormalisedPosition"],
                    #     d_sample["Value"],
                    #     color=colour,
                    #     alpha=sample_alpha,
                    #     linewidth=1.0,
                    # )

            s_sub = s_sub.sort_values("NormalisedPosition")

            x = s_sub["NormalisedPosition"].to_numpy(float)
            y = s_sub["mean"].to_numpy(float)
            lo = s_sub["lo"].to_numpy(float)
            hi = s_sub["hi"].to_numpy(float)

            valid = np.isfinite(x) & np.isfinite(y)

            ax.plot(
                x[valid],
                y[valid],
                color=colour,
                linewidth=2.8,
                label=label,
            )

            ax.fill_between(
                x[valid],
                lo[valid],
                hi[valid],
                color=colour,
                alpha=ci_alpha,
                linewidth=0,
            )

            ax.plot(
                x[valid],
                lo[valid],
                color=colour,
                linewidth=0.8,
                alpha=0.65,
            )

            ax.plot(
                x[valid],
                hi[valid],
                color=colour,
                linewidth=0.8,
                alpha=0.65,
            )

            ax.set_title(label)
            ax.set_xlabel("Normalised position through dermis (%)")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0, 100)

            if ylim is not None:
                ax.set_ylim(*ylim)

            ax.grid(False)

        fig.suptitle(
            title or f"Raw nanoindentation {variable} across normalised position",
            y=0.995,
        )

        plt.tight_layout()
        plt.show()

        return fig, axes, smooth

    # ------------------------------------------------------------------
    # Combined mode
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    for subtype in present_subtypes:
        colour = colours.get(subtype, "grey")
        label = labels.get(subtype, subtype)

        s_sub = smooth[smooth["SubtypeClean"].astype(str) == subtype].copy()
        s_sub = s_sub.sort_values("NormalisedPosition")

        x = s_sub["NormalisedPosition"].to_numpy(float)
        y = s_sub["mean"].to_numpy(float)
        lo = s_sub["lo"].to_numpy(float)
        hi = s_sub["hi"].to_numpy(float)

        valid = np.isfinite(x) & np.isfinite(y)

        ax.plot(
            x[valid],
            y[valid],
            color=colour,
            linewidth=2.8,
            label=label,
        )

        ax.fill_between(
            x[valid],
            lo[valid],
            hi[valid],
            color=colour,
            alpha=ci_alpha,
            linewidth=0,
        )

    ax.set_xlabel("Normalised position through dermis (%)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 100)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_title(
        title or f"Raw nanoindentation {variable} across normalised position"
    )
    ax.legend(frameon=False)
    ax.grid(False)

    plt.tight_layout()
    plt.show()

    return fig, ax, smooth


def prepare_ni_raw_line_data(
    ni_raw,
    variable,
    subtype_order=SUBTYPE_ORDER,
    convert_modulus_to_kpa=True,
    require_line_scan=True,
    require_dermis=True,
    verbose=True,
):
    """
    Prepare raw point-level NI line-scan data.

    Uses:
        x = normalised point position through the filtered line scan
        y = raw variable value

    By default this keeps:
        RegionKey == line scan, if RegionKey exists
        layer/Layer == dermis, if a dermis layer exists

    NormalisedPosition is then calculated only across the remaining points
    within each sample.
    """

    if ni_raw is None or ni_raw.empty:
        return pd.DataFrame()

    if variable not in ni_raw.columns:
        raise KeyError(f"Column not found in ni_raw: {variable}")

    d = ni_raw.copy()
    d = add_clean_subtype(d)

    d = d[d["SubtypeClean"].astype(str).isin(subtype_order)].copy()

    if d.empty:
        return pd.DataFrame()

    # ------------------------------------------------------------
    # Restrict to line scan if possible
    # ------------------------------------------------------------
    if require_line_scan:
        line_col = None

        for c in ["RegionKey", "Region", "region", "TechniqueRegion"]:
            if c in d.columns:
                vals = d[c].astype(str).str.lower().str.strip()
                if vals.str.contains("line", na=False).any():
                    line_col = c
                    break

        if line_col is not None:
            before = len(d)
            d = d[
                d[line_col]
                .astype(str)
                .str.lower()
                .str.strip()
                .str.contains("line", na=False)
            ].copy()

            if verbose:
                print(f"[NI raw prep] Kept line-scan rows using {line_col}: {before} -> {len(d)}")

    if d.empty:
        print(f"No NI line-scan rows left for {variable}.")
        return pd.DataFrame()

    # ------------------------------------------------------------
    # Restrict to dermis if possible
    # ------------------------------------------------------------
    if require_dermis:
        layer_col = None

        for c in ["layer", "Layer", "Region", "region"]:
            if c in d.columns:
                vals = d[c].astype(str).str.lower().str.strip()
                if vals.eq("dermis").any():
                    layer_col = c
                    break

        if layer_col is not None:
            before = len(d)
            d = d[
                d[layer_col]
                .astype(str)
                .str.lower()
                .str.strip()
                .eq("dermis")
            ].copy()

            if verbose:
                print(f"[NI raw prep] Kept dermis rows using {layer_col}: {before} -> {len(d)}")

        else:
            if verbose:
                print("[NI raw prep] No explicit dermis layer column found or no dermis rows detected.")
                print("Available possible layer/region values:")

                for c in ["layer", "Layer", "Region", "region", "RegionKey"]:
                    if c in d.columns:
                        vals = d[c].dropna().astype(str).unique()
                        print(f"  {c}: {vals[:20]}")

    if d.empty:
        print(f"No dermis NI rows left for {variable}.")
        return pd.DataFrame()

    # ------------------------------------------------------------
    # Numeric conversion
    # ------------------------------------------------------------
    d[variable] = pd.to_numeric(d[variable], errors="coerce")

    if "x" in d.columns:
        d["x"] = pd.to_numeric(d["x"], errors="coerce")
    else:
        d["x"] = np.nan

    if "y" in d.columns:
        d["y"] = pd.to_numeric(d["y"], errors="coerce")
    else:
        d["y"] = np.nan

    d = d.dropna(subset=[variable]).copy()

    if d.empty:
        return pd.DataFrame()

    # ------------------------------------------------------------
    # Normalised position within each sample after filtering
    # ------------------------------------------------------------
    sample_col = "SampleKey" if "SampleKey" in d.columns else "Sample"

    # Prefer physical scan axis if available. For mostly vertical line scans this is y.
    d = d.sort_values([sample_col, "y", "x"]).copy()

    d["PointOrder"] = d.groupby(sample_col).cumcount()
    d["N_points_sample"] = d.groupby(sample_col)["PointOrder"].transform("max") + 1

    d["NormalisedPosition"] = np.where(
        d["N_points_sample"] > 1,
        d["PointOrder"] / (d["N_points_sample"] - 1) * 100,
        50,
    )

    d["Value"] = d[variable]

    if convert_modulus_to_kpa and variable in {
        "mod_Hertz",
        "mod_OP",
        "E0_Visco",
        "Einf_Visco",
        "G0_Visco",
        "G1_Visco",
        "Eff_file",
        "mod_file",
    }:
        d["Value"] = d["Value"] / 1000

    d["SubtypeClean"] = pd.Categorical(
        d["SubtypeClean"],
        categories=subtype_order,
        ordered=True,
    )

    return d.sort_values(["SubtypeClean", sample_col, "NormalisedPosition"]).reset_index(drop=True)

def prepare_ni_raw_binned_position_data(
    ni_raw,
    variable,
    n_position_bins=5,
    subtype_order=SUBTYPE_ORDER,
    convert_modulus_to_kpa=True,
):
    """
    Convert raw NI point-level data into normalised position bins.

    Output = one row per raw point with:
        NormalisedPosition
        PositionBin
        Value
    """

    df = prepare_ni_raw_line_data(
        ni_raw=ni_raw,
        variable=variable,
        subtype_order=subtype_order,
        convert_modulus_to_kpa=convert_modulus_to_kpa,
    )

    if df.empty:
        return pd.DataFrame()

    bins = np.linspace(0, 100, n_position_bins + 1)
    labels = np.arange(1, n_position_bins + 1)

    df["PositionBin"] = pd.cut(
        df["NormalisedPosition"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    df["PositionBin"] = pd.to_numeric(df["PositionBin"], errors="coerce")

    return df.dropna(subset=["PositionBin", "Value"]).copy()

def plot_ni_binned_bar_by_position(
    ni_raw,
    variable="mod_Hertz",
    ylabel="Indentation modulus (kPa)",
    title=None,
    n_position_bins=5,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    colours=NI_COLOURS,
    labels=NI_LABELS,
    ylim=None,
    scatter=True,
    scatter_alpha=0.55,
    point_size=18,
    jitter=0.06,
    bar_alpha=0.75,
    errorbar="std",  # "std" or "sem"
):
    """
    Bar chart:
        x = normalised position bin
        grouped bars = subtype
        y = mean variable
        error = SD or SEM
        points = sample means for each bin/subtype

    subtypes_to_plot:
        None -> plot all subtypes present in subtype_order
        list/tuple -> only plot those subtypes, in subtype_order order
    """

    if subtypes_to_plot is None:
        wanted_subtypes = [str(st).strip() for st in subtype_order]
    else:
        wanted_set = {str(st).strip() for st in subtypes_to_plot}
        wanted_subtypes = [
            str(st).strip()
            for st in subtype_order
            if str(st).strip() in wanted_set
        ]
        wanted_subtypes += [
            str(st).strip()
            for st in subtypes_to_plot
            if str(st).strip() not in wanted_subtypes
        ]

    df = prepare_ni_raw_binned_position_data(
        ni_raw=ni_raw,
        variable=variable,
        n_position_bins=n_position_bins,
        subtype_order=subtype_order,
    )

    if df.empty:
        print(f"No raw NI data available for {variable}")
        return None, None, pd.DataFrame()

    df = df[df["SubtypeClean"].astype(str).isin(wanted_subtypes)].copy()

    if df.empty:
        print("No matching NI subtypes to plot.")
        print("Requested:", subtypes_to_plot)
        print("Available:", sorted(prepare_ni_raw_binned_position_data(
            ni_raw=ni_raw,
            variable=variable,
            n_position_bins=n_position_bins,
            subtype_order=subtype_order,
        )["SubtypeClean"].astype(str).unique()))
        return None, None, pd.DataFrame()

    sample_col = "SampleKey" if "SampleKey" in df.columns else "Sample"

    sample_means = (
        df.groupby(["SubtypeClean", sample_col, "PositionBin"], observed=True)["Value"]
        .mean()
        .reset_index()
        .rename(columns={"Value": "SampleMean"})
    )

    summary = (
        sample_means
        .groupby(["SubtypeClean", "PositionBin"], observed=True)["SampleMean"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )

    summary["sem"] = summary["std"] / np.sqrt(summary["n"])
    summary["err"] = summary["sem"] if errorbar == "sem" else summary["std"]

    present_subtypes = [
        st for st in wanted_subtypes
        if st in set(summary["SubtypeClean"].astype(str))
    ]

    if not present_subtypes:
        print("No matching NI subtypes to plot after summary.")
        return None, None, summary

    x_bins = np.arange(1, n_position_bins + 1)
    width = 0.8 / max(len(present_subtypes), 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    rng = np.random.default_rng(0)

    for i, subtype in enumerate(present_subtypes):
        colour = colours.get(subtype, "grey")
        x = x_bins + (i - (len(present_subtypes) - 1) / 2) * width

        s_sub = (
            summary[summary["SubtypeClean"].astype(str) == subtype]
            .set_index("PositionBin")
            .reindex(x_bins)
        )

        y = s_sub["mean"].to_numpy(dtype=float)
        err = s_sub["err"].to_numpy(dtype=float)

        ax.bar(
            x,
            y,
            width=width,
            color=colour,
            alpha=bar_alpha,
            edgecolor="black",
            linewidth=0.6,
            label=labels.get(subtype, subtype),
        )

        ax.errorbar(
            x,
            y,
            yerr=err,
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )

        if scatter:
            pts_sub = sample_means[sample_means["SubtypeClean"].astype(str) == subtype].copy()

            for b in x_bins:
                vals = pts_sub.loc[pts_sub["PositionBin"] == b, "SampleMean"].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]

                if vals.size == 0:
                    continue

                x0 = b + (i - (len(present_subtypes) - 1) / 2) * width
                xj = x0 + rng.normal(0, jitter, size=vals.size)

                ax.scatter(
                    xj,
                    vals,
                    color="black",
                    alpha=scatter_alpha,
                    s=point_size,
                    linewidths=0,
                    zorder=3,
                )

    ax.set_xticks(x_bins)
    ax.set_xticklabels([f"Bin {i}" for i in x_bins])
    ax.set_xlabel("Normalised position bin through dermis")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{variable} by normalised position bin")
    ax.legend(frameon=False)
    ax.grid(False)

    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()

    return fig, ax, summary


def calculate_ni_grid_region_summary(
    ni_grid,
    variable="mod_Hertz",
    subtypes_to_plot=None,
    subtype_order=SUBTYPE_ORDER,
    regions_to_plot=("lower dermis", "upper dermis"),
    errorbar="std",
):
    """
    Calculate NI grid summary for upper/lower dermis regions.

    Recommended input:
        tables["ni_grid"]

    This uses screened exported grid summaries when available:
        mod_Hertz_mean
        tau_Visco_mean

    It then summarises across biological samples by subtype and region.
    """

    if ni_grid is None or ni_grid.empty:
        print("No NI grid data available.")
        return pd.DataFrame(), pd.DataFrame()

    df = ni_grid.copy()

    if "SubtypeClean" not in df.columns:
        df = add_clean_subtype(df)

    # ------------------------------------------------------------
    # Resolve variable column
    # ------------------------------------------------------------
    candidate_cols = {
        "mod_Hertz": ["mod_Hertz_mean", "mod_Hertz"],
        "tau_Visco": ["tau_Visco_mean", "tau_Visco"],
        "E0_Visco": ["E0_Visco_mean", "E0_Visco"],
        "Einf_Visco": ["Einf_Visco_mean", "Einf_Visco"],
        "G0_Visco": ["G0_Visco_mean", "G0_Visco"],
        "G1_Visco": ["G1_Visco_mean", "G1_Visco"],
    }.get(variable, [f"{variable}_mean", variable])

    variable_col = None
    for col in candidate_cols:
        if col in df.columns:
            variable_col = col
            break

    if variable_col is None:
        print("\nAvailable NI grid columns:")
        print(df.columns.tolist())
        raise KeyError(
            f"Variable '{variable}' not found in NI grid dataframe. "
            f"Tried: {candidate_cols}"
        )

    # Prefer exported n column if present
    n_col = None
    for col in [f"{variable}_n", variable_col.replace("_mean", "_n")]:
        if col in df.columns:
            n_col = col
            break

    # ------------------------------------------------------------
    # Find sample column
    # ------------------------------------------------------------
    sample_col = None
    for c in ["SampleKey", "Sample", "Sample Number", "SampleNumber", "FOLDER NAME"]:
        if c in df.columns:
            sample_col = c
            break

    if sample_col is None:
        raise KeyError(
            "Could not find a sample column. Expected one of: "
            "'SampleKey', 'Sample', 'Sample Number', 'SampleNumber', 'FOLDER NAME'."
        )

    # ------------------------------------------------------------
    # Find region column
    # ------------------------------------------------------------
    region_col = None
    for c in ["Region", "region", "RegionClean", "RegionKey", "Layer", "layer", "NIRegion"]:
        if c in df.columns:
            region_col = c
            break

    if region_col is None:
        raise KeyError(
            "Could not find a region column. Expected one of: "
            "'Region', 'region', 'RegionClean', 'RegionKey', 'Layer', 'layer', 'NIRegion'."
        )

    def clean_grid_region(x):
        key = str(x).strip().lower()
        key = key.replace("_", " ")
        key = key.replace("-", " ")
        key = " ".join(key.split())
        compact = key.replace(" ", "")

        mapping = {
            "lower dermis": "lower dermis",
            "lowerdermis": "lower dermis",
            "lower": "lower dermis",
            "dermis lower": "lower dermis",
            "sub dermis": "lower dermis",
            "dermis sub": "lower dermis",

            "upper dermis": "upper dermis",
            "upperdermis": "upper dermis",
            "upper": "upper dermis",
            "dermis upper": "upper dermis",
            "epi dermis": "upper dermis",
            "dermis epi": "upper dermis",
        }

        return mapping.get(key, mapping.get(compact, key))

    df["GridRegionClean"] = df[region_col].apply(clean_grid_region)

    regions_to_plot = [clean_grid_region(r) for r in regions_to_plot]

    if subtypes_to_plot is not None:
        subtype_order = [normalise_subtype(st) for st in subtypes_to_plot]
    else:
        subtype_order = [normalise_subtype(st) for st in subtype_order]

    df = df[
        df["SubtypeClean"].astype(str).isin(subtype_order)
        & df["GridRegionClean"].astype(str).isin(regions_to_plot)
    ].copy()

    if df.empty:
        print("No NI grid data left after subtype/region filtering.")
        print("Available subtypes:")
        print(ni_grid.get("SubtypeClean", pd.Series(dtype=str)).dropna().unique())
        print("Available regions:")
        print(ni_grid[region_col].dropna().astype(str).unique())
        return pd.DataFrame(), pd.DataFrame()

    df[variable_col] = pd.to_numeric(df[variable_col], errors="coerce")
    df = df.dropna(subset=[variable_col]).copy()

    if df.empty:
        print(f"No numeric NI grid values found for {variable}.")
        return pd.DataFrame(), pd.DataFrame()

    # Convert modulus-like variables from Pa to kPa
    if variable in ["mod_Hertz", "E0_Visco", "Einf_Visco", "G0_Visco", "G1_Visco"]:
        df["_PlotValue"] = df[variable_col] / 1000.0
    else:
        df["_PlotValue"] = df[variable_col]

    # ------------------------------------------------------------
    # Sample-level summary
    # ------------------------------------------------------------
    # If NI_Grid already contains screened sample-region means, keep those.
    # If RawNI_Grid is accidentally passed, average raw points per sample-region.
    sample_summary = (
        df.groupby(["SubtypeClean", sample_col, "GridRegionClean"], observed=True)
        .agg(
            SampleMean=("_PlotValue", "mean"),
            N_rows=("_PlotValue", "count"),
        )
        .reset_index()
    )

    if n_col is not None:
        n_lookup = (
            df.groupby(["SubtypeClean", sample_col, "GridRegionClean"], observed=True)[n_col]
            .sum()
            .reset_index()
            .rename(columns={n_col: "N_screened_points"})
        )

        sample_summary = sample_summary.merge(
            n_lookup,
            on=["SubtypeClean", sample_col, "GridRegionClean"],
            how="left",
        )
    else:
        sample_summary["N_screened_points"] = np.nan

    # ------------------------------------------------------------
    # Group-level summary across biological samples
    # ------------------------------------------------------------
    group_summary = (
        sample_summary
        .groupby(["SubtypeClean", "GridRegionClean"], observed=True)["SampleMean"]
        .agg(
            mean="mean",
            std="std",
            n="count",
        )
        .reset_index()
    )

    group_summary["sem"] = group_summary["std"] / np.sqrt(group_summary["n"])

    if errorbar == "sem":
        group_summary["error"] = group_summary["sem"]
    else:
        group_summary["error"] = group_summary["std"]

    group_summary["SubtypeClean"] = pd.Categorical(
        group_summary["SubtypeClean"],
        categories=subtype_order,
        ordered=True,
    )

    group_summary["GridRegionClean"] = pd.Categorical(
        group_summary["GridRegionClean"],
        categories=regions_to_plot,
        ordered=True,
    )

    group_summary = group_summary.sort_values(
        ["SubtypeClean", "GridRegionClean"]
    ).reset_index(drop=True)

    sample_summary["SubtypeClean"] = pd.Categorical(
        sample_summary["SubtypeClean"],
        categories=subtype_order,
        ordered=True,
    )

    sample_summary["GridRegionClean"] = pd.Categorical(
        sample_summary["GridRegionClean"],
        categories=regions_to_plot,
        ordered=True,
    )

    sample_summary = sample_summary.sort_values(
        ["SubtypeClean", "GridRegionClean", sample_col]
    ).reset_index(drop=True)

    print(f"\nUsing NI grid value column: {variable_col}")
    if n_col is not None:
        print(f"Using NI grid screened n column: {n_col}")

    return sample_summary, group_summary

def plot_ni_grid_upper_lower_bar(
    ni_grid,
    variable="mod_Hertz",
    ylabel=None,
    title=None,
    subtypes_to_plot=None,
    subtype_order=SUBTYPE_ORDER,
    regions_to_plot=("lower dermis", "upper dermis"),
    errorbar="std",
    ylim=None,
    hatches=True,
    show_points=True,
    point_alpha=0.7,
    run_stats=True,
    print_stats=True,
):
    """
    Plot NI grid upper/lower region means as grouped bars.

    Bar order:
        subtype lower, subtype upper, subtype2 lower, subtype2 upper, ...

    The 5x5 grid is first averaged per sample/region, then bars show
    mean ± error across biological samples.

    Statistical testing:
        Paired upper vs lower dermis comparison within each subtype.

        For each subtype:
            1. Match lower and upper dermis values by sample
            2. Calculate paired differences: upper - lower
            3. Run paired t-test

        Shapiro-Wilk normality is still calculated and printed,
        but the selected test is always paired t-test.
    """

    sample_summary, group_summary = calculate_ni_grid_region_summary(
        ni_grid=ni_grid,
        variable=variable,
        subtypes_to_plot=subtypes_to_plot,
        subtype_order=subtype_order,
        regions_to_plot=regions_to_plot,
        errorbar=errorbar,
    )

    if group_summary.empty:
        print(f"No NI grid summary to plot for {variable}.")
        return None, None, sample_summary, group_summary, pd.DataFrame()

    if subtypes_to_plot is not None:
        subtype_order = [
            normalise_subtype(st)
            for st in subtypes_to_plot
        ]
    else:
        subtype_order = [
            normalise_subtype(st)
            for st in subtype_order
        ]

    regions_to_plot = list(
        group_summary["GridRegionClean"].cat.categories
    )

    if ylabel is None:
        if variable == "mod_Hertz":
            ylabel = "Indentation modulus (kPa)"
        elif variable == "tau_Visco":
            ylabel = "Tau (s)"
        else:
            ylabel = variable

    if title is None:
        title = "Nanoindentation grid region summary"

    fig, ax = plt.subplots(
        figsize=(
            max(7, len(subtype_order) * 1.4),
            5,
        )
    )

    n_regions = len(regions_to_plot)
    x = np.arange(len(subtype_order))
    total_width = 0.75
    bar_width = total_width / n_regions

    hatch_map = {
        "lower dermis": "",
        "upper dermis": "//",
    }

    region_labels = {
        "lower dermis": "Lower",
        "upper dermis": "Upper",
    }

    # Store bar x-positions for stats annotations
    bar_positions = {}

    for i, region in enumerate(regions_to_plot):

        offset = (
            i - (n_regions - 1) / 2
        ) * bar_width

        region_summary = (
            group_summary[
                group_summary["GridRegionClean"].astype(str) == str(region)
            ]
            .set_index("SubtypeClean")
            .reindex(subtype_order)
        )

        means = region_summary["mean"].to_numpy(dtype=float)
        errors = region_summary["error"].to_numpy(dtype=float)

        colours = [
            NI_COLOURS.get(st, "grey")
            for st in subtype_order
        ]

        bars = ax.bar(
            x + offset,
            means,
            yerr=errors,
            width=bar_width,
            color=colours,
            edgecolor="black",
            linewidth=0.8,
            capsize=4,
            label=region_labels.get(str(region), str(region)),
        )

        for j, subtype in enumerate(subtype_order):
            bar_positions[
                (
                    str(subtype),
                    str(region),
                )
            ] = x[j] + offset

        if hatches:
            for bar in bars:
                bar.set_hatch(
                    hatch_map.get(str(region), "")
                )

        # --------------------------------------------------------
        # Overlay sample-level points
        # --------------------------------------------------------
        if show_points and not sample_summary.empty:

            region_points = sample_summary[
                sample_summary["GridRegionClean"].astype(str) == str(region)
            ].copy()

            for j, subtype in enumerate(subtype_order):

                vals = (
                    region_points[
                        region_points["SubtypeClean"].astype(str) == str(subtype)
                    ]["SampleMean"]
                    .dropna()
                    .to_numpy(dtype=float)
                )

                if len(vals) == 0:
                    continue

                jitter = np.linspace(
                    -bar_width * 0.18,
                    bar_width * 0.18,
                    len(vals),
                )

                if len(vals) == 1:
                    jitter = np.array([0.0])

                ax.scatter(
                    np.full(len(vals), x[j] + offset) + jitter,
                    vals,
                    s=28,
                    color="black",
                    alpha=point_alpha,
                    zorder=5,
                )

    labels = [
        SUBTYPE_LABELS.get(st, st)
        for st in subtype_order
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(
        title="Region",
        frameon=False,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    print("\nNI grid region summary:")
    print(
        group_summary[
            [
                "SubtypeClean",
                "GridRegionClean",
                "mean",
                "std",
                "sem",
                "n",
                "error",
            ]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------
    # Paired upper vs lower statistical tests within each subtype
    # ------------------------------------------------------------
    stats_df = pd.DataFrame()

    if run_stats and sample_summary is not None and not sample_summary.empty:

        stats_input = sample_summary.copy()

        required_cols = [
            "SubtypeClean",
            "GridRegionClean",
            "SampleMean",
        ]

        missing_cols = [
            c for c in required_cols
            if c not in stats_input.columns
        ]

        if missing_cols:

            print("\n[NI grid stats] Missing required columns:")
            print(missing_cols)
            print("Available columns:")
            print(stats_input.columns.tolist())

        else:

            # --------------------------------------------------------
            # Auto-detect sample column
            # --------------------------------------------------------
            possible_sample_cols = [
                "SampleKey",
                "Sample",
                "SampleID",
                "sample",
                "Filenumber",
                "FileNumber",
                "File number",
                "Sample Number",
                "SampleNumber",
                "experiment",
                "Experiment",
            ]

            sample_col = None

            for c in possible_sample_cols:
                if c in stats_input.columns:
                    sample_col = c
                    break

            if sample_col is None:

                print("\n[NI grid stats] Could not find sample column.")
                print("Available columns:")
                print(stats_input.columns.tolist())

            else:

                rows = []

                region_a = "lower dermis"
                region_b = "upper dermis"

                for subtype in subtype_order:

                    d_sub = stats_input[
                        stats_input["SubtypeClean"].astype(str) == str(subtype)
                    ].copy()

                    d_pair = d_sub[
                        d_sub["GridRegionClean"].astype(str).isin(
                            [
                                region_a,
                                region_b,
                            ]
                        )
                    ].copy()

                    if d_pair.empty:
                        continue

                    wide = d_pair.pivot_table(
                        index=sample_col,
                        columns="GridRegionClean",
                        values="SampleMean",
                        aggfunc="mean",
                    )

                    if region_a not in wide.columns or region_b not in wide.columns:
                        continue

                    wide = wide[
                        [
                            region_a,
                            region_b,
                        ]
                    ].dropna()

                    n_pairs = len(wide)

                    if n_pairs < 2:

                        rows.append(
                            {
                                "SubtypeClean": subtype,
                                "Label": SUBTYPE_LABELS.get(subtype, subtype),
                                "RegionA": region_a,
                                "RegionB": region_b,
                                "n_pairs": n_pairs,
                                "MeanA": wide[region_a].mean() if region_a in wide else np.nan,
                                "MeanB": wide[region_b].mean() if region_b in wide else np.nan,
                                "MeanDifference_B_minus_A": np.nan,
                                "NormalityTest": "not enough pairs",
                                "Normality_p": np.nan,
                                "NormalityInterpretation": "not tested",
                                "SelectedTest": "not tested",
                                "statistic": np.nan,
                                "p_value": np.nan,
                                "stars": "ns",
                            }
                        )

                        continue

                    a = wide[region_a].astype(float)
                    b = wide[region_b].astype(float)

                    differences = b - a

                    mean_a = a.mean()
                    mean_b = b.mean()
                    mean_diff = differences.mean()

                    # ------------------------------------------------
                    # Shapiro-Wilk on paired differences
                    # Still printed, but does not control test choice
                    # ------------------------------------------------
                    if n_pairs >= 3:

                        try:
                            shapiro_stat, shapiro_p = stats.shapiro(differences)

                            normality_test = "Shapiro-Wilk on paired differences"

                            normality_interpretation = (
                                "normal"
                                if shapiro_p >= 0.05
                                else "non-normal"
                            )

                        except Exception:

                            shapiro_p = np.nan
                            normality_test = "Shapiro-Wilk failed"
                            normality_interpretation = "not tested"

                    else:

                        shapiro_p = np.nan
                        normality_test = "not enough pairs for Shapiro-Wilk"
                        normality_interpretation = "not tested"

                    # ------------------------------------------------
                    # Always use paired t-test
                    # ------------------------------------------------
                    selected_test = "paired t-test"

                    try:
                        stat, p_value = stats.ttest_rel(
                            a,
                            b,
                            nan_policy="omit",
                        )

                    except Exception:
                        stat, p_value = np.nan, np.nan

                    rows.append(
                        {
                            "SubtypeClean": subtype,
                            "Label": SUBTYPE_LABELS.get(subtype, subtype),
                            "RegionA": region_a,
                            "RegionB": region_b,
                            "n_pairs": n_pairs,
                            "MeanA": mean_a,
                            "MeanB": mean_b,
                            "MeanDifference_B_minus_A": mean_diff,
                            "NormalityTest": normality_test,
                            "Normality_p": shapiro_p,
                            "NormalityInterpretation": normality_interpretation,
                            "SelectedTest": selected_test,
                            "statistic": stat,
                            "p_value": p_value,
                            "stars": p_to_stars(p_value),
                        }
                    )

                stats_df = pd.DataFrame(rows)

                # --------------------------------------------------------
                # Add significance annotations
                # --------------------------------------------------------
                if not stats_df.empty:

                    y_min, y_max = ax.get_ylim()
                    y_range = y_max - y_min

                    for _, row in stats_df.iterrows():

                        subtype = str(row["SubtypeClean"])
                        p_value = row["p_value"]
                        stars = row["stars"]

                        if pd.isna(p_value):
                            continue

                        x1 = bar_positions.get(
                            (
                                subtype,
                                region_a,
                            )
                        )

                        x2 = bar_positions.get(
                            (
                                subtype,
                                region_b,
                            )
                        )

                        if x1 is None or x2 is None:
                            continue

                        region_summary = group_summary[
                            group_summary["SubtypeClean"].astype(str) == subtype
                        ].copy()

                        region_summary = region_summary[
                            region_summary["GridRegionClean"].astype(str).isin(
                                [
                                    region_a,
                                    region_b,
                                ]
                            )
                        ]

                        if region_summary.empty:
                            continue

                        y_base = (
                            region_summary["mean"]
                            + region_summary["error"].fillna(0)
                        ).max()

                        y = y_base + 0.06 * y_range
                        h = 0.025 * y_range

                        ax.plot(
                            [
                                x1,
                                x1,
                                x2,
                                x2,
                            ],
                            [
                                y,
                                y + h,
                                y + h,
                                y,
                            ],
                            color="black",
                            linewidth=1,
                        )

                        ax.text(
                            (x1 + x2) / 2,
                            y + h,
                            stars,
                            ha="center",
                            va="bottom",
                            fontsize=11,
                        )

                    # Expand ylim if needed
                    current_ymin, current_ymax = ax.get_ylim()
                    ax.set_ylim(
                        current_ymin,
                        current_ymax + 0.08 * y_range,
                    )

                # --------------------------------------------------------
                # Print stats
                # --------------------------------------------------------
                if print_stats and not stats_df.empty:

                    stats_print = stats_df.copy()

                    for c in [
                        "MeanA",
                        "MeanB",
                        "MeanDifference_B_minus_A",
                        "Normality_p",
                        "statistic",
                        "p_value",
                    ]:
                        if c in stats_print.columns:
                            stats_print[c] = pd.to_numeric(
                                stats_print[c],
                                errors="coerce",
                            )

                    print("\n[NI grid paired t-tests: lower dermis vs upper dermis]")
                    print(
                        stats_print[
                            [
                                "Label",
                                "RegionA",
                                "RegionB",
                                "n_pairs",
                                "MeanA",
                                "MeanB",
                                "MeanDifference_B_minus_A",
                                "NormalityTest",
                                "Normality_p",
                                "NormalityInterpretation",
                                "SelectedTest",
                                "statistic",
                                "p_value",
                                "stars",
                            ]
                        ].to_string(index=False)
                    )

    fig.tight_layout()

    return fig, ax, sample_summary, group_summary, stats_df

def screen_ni_raw_dataframe(
    df,
    variables=("mod_Hertz", "tau_Visco"),
    rsq_min_hertz=0.8,
    rsq_min_visco=0.8,
    hi_modulus_pa=100_000,
    lo_modulus_pa=0,
    std_devs=3,
):
    """
    Screen raw NI dataframe in the analysis file.

    Keeps the original units:
        mod_Hertz stays in Pa
        tau_Visco stays in seconds

    Sets failed values to NaN, so downstream plotting functions
    automatically ignore them.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    if "SubtypeClean" not in d.columns:
        d = add_clean_subtype(d)

    # Choose grouping for std trimming.
    # This screens within each sample/region/layer where possible.
    group_cols = [
        c for c in ["SubtypeClean", "SampleKey", "RegionKey", "layer", "region"]
        if c in d.columns
    ]

    for variable in variables:
        if variable not in d.columns:
            print(f"[NI screening] Skipping {variable}: column not found.")
            continue

        d[variable] = pd.to_numeric(d[variable], errors="coerce")

        # --------------------------------------------------------
        # Rsq screening
        # --------------------------------------------------------
        if variable == "mod_Hertz":
            if "Rsq_Hertz" in d.columns:
                d["Rsq_Hertz"] = pd.to_numeric(d["Rsq_Hertz"], errors="coerce")
                d.loc[d["Rsq_Hertz"] < rsq_min_hertz, variable] = np.nan

            d.loc[d[variable] < lo_modulus_pa, variable] = np.nan
            d.loc[d[variable] > hi_modulus_pa, variable] = np.nan

        elif variable == "tau_Visco":
            rsq_col = None

            for c in ["Rsq_Visco", "Rsq_ViscoAna", "ViscoAna_r2"]:
                if c in d.columns:
                    rsq_col = c
                    break

            if rsq_col is not None:
                d[rsq_col] = pd.to_numeric(d[rsq_col], errors="coerce")
                d.loc[d[rsq_col] < rsq_min_visco, variable] = np.nan

            d.loc[d[variable] < 0, variable] = np.nan

        else:
            d.loc[d[variable] < 0, variable] = np.nan

        # --------------------------------------------------------
        # Group-wise std trimming
        # --------------------------------------------------------
        if group_cols:
            keep = pd.Series(True, index=d.index)

            for _, idx in d.groupby(group_cols, dropna=False).groups.items():
                vals = d.loc[idx, variable].to_numpy(dtype=float)
                finite = np.isfinite(vals)

                if finite.sum() < 2:
                    continue

                mu = float(np.nanmean(vals[finite]))
                sd = float(np.nanstd(vals[finite], ddof=1))

                if not np.isfinite(sd) or sd == 0:
                    continue

                lo = mu - std_devs * sd
                hi = mu + std_devs * sd

                sub_keep = np.ones(len(vals), dtype=bool)
                sub_keep[finite] = (vals[finite] >= lo) & (vals[finite] <= hi)

                keep.loc[idx] = sub_keep

            d.loc[~keep, variable] = np.nan

    return d

def screen_ni_values_for_plot(
    df,
    variable="mod_Hertz",
    rsq_min_hertz=0.5,
    rsq_min_visco=0.5,
    hi_modulus_pa=100000,
    std_devs=3,
):
    """
    Apply NI-style screening to point-level values for plotting.

    For mod_Hertz:
        Rsq_Hertz >= rsq_min_hertz
        value >= 0
        value <= hi_modulus_pa
        remove values > std_devs from mean

    For tau_Visco:
        Rsq_Visco >= rsq_min_visco
        value >= 0
        remove values > std_devs from mean
    """

    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    if variable not in d.columns:
        raise KeyError(f"Variable '{variable}' not found. Available columns: {list(d.columns)}")

    d[variable] = pd.to_numeric(d[variable], errors="coerce")

    if variable == "mod_Hertz":
        if "Rsq_Hertz" in d.columns:
            d["Rsq_Hertz"] = pd.to_numeric(d["Rsq_Hertz"], errors="coerce")
            d.loc[d["Rsq_Hertz"] < rsq_min_hertz, variable] = np.nan

        d.loc[d[variable] < 0, variable] = np.nan
        d.loc[d[variable] > hi_modulus_pa, variable] = np.nan

    elif variable == "tau_Visco":
        if "Rsq_Visco" in d.columns:
            d["Rsq_Visco"] = pd.to_numeric(d["Rsq_Visco"], errors="coerce")
            d.loc[d["Rsq_Visco"] < rsq_min_visco, variable] = np.nan
        elif "Rsq_ViscoAna" in d.columns:
            d["Rsq_ViscoAna"] = pd.to_numeric(d["Rsq_ViscoAna"], errors="coerce")
            d.loc[d["Rsq_ViscoAna"] < rsq_min_visco, variable] = np.nan

        d.loc[d[variable] < 0, variable] = np.nan

    else:
        d.loc[d[variable] < 0, variable] = np.nan

    # Convert modulus-like variables from Pa to kPa for plotting
    if variable in ["mod_Hertz", "E0_Visco", "Einf_Visco", "G0_Visco", "G1_Visco"]:
        d["_PlotValue"] = d[variable] / 1000.0
    else:
        d["_PlotValue"] = d[variable]

    # Remove group-wise outliers within each sample/region
    group_cols = []
    for c in ["SubtypeClean", "SampleKey", "RegionDiagnostic"]:
        if c in d.columns:
            group_cols.append(c)

    if group_cols:
        keep = pd.Series(True, index=d.index)

        for _, idx in d.groupby(group_cols, dropna=False).groups.items():
            vals = d.loc[idx, "_PlotValue"].to_numpy(dtype=float)
            finite = np.isfinite(vals)

            if finite.sum() < 2:
                continue

            mu = float(np.nanmean(vals[finite]))
            sd = float(np.nanstd(vals[finite], ddof=1))

            if not np.isfinite(sd) or sd == 0:
                continue

            lo = mu - std_devs * sd
            hi = mu + std_devs * sd

            sub_keep = np.ones(len(vals), dtype=bool)
            sub_keep[finite] = (vals[finite] >= lo) & (vals[finite] <= hi)

            keep.loc[idx] = sub_keep

        d = d.loc[keep].copy()

    return d

def plot_ni_repeat_diagnostic_by_group(
    ni_raw,
    ni_grid_raw,
    variable="mod_Hertz",
    groups=None,
    ylabel=None,
    title_prefix=None,
    ylim=None,
    errorbar="std",
    rsq_min_hertz=0.5,
    rsq_min_visco=0.5,
    hi_modulus_pa=100000,
    std_devs=3,
    show_points=True,
    point_alpha=0.85,
    hatches=True,
    scatter_by_rsq=True,
    rsq_cmap="jet",
    rsq_vmin=0,
    rsq_vmax=1,
):    
    """
    Produce one diagnostic bar chart per biological group.

    Each graph shows each repeat separately.

    For each repeat:
        Line scan
        Lower grid
        Upper grid

    Bars:
        mean of screened points within that repeat/region

    Error bars:
        std or sem of screened points within that repeat/region

    Scatter:
        individual screened points inside each bar
        line scan = all line-scan points
        lower/upper = 5x5 grid points, usually up to 25 points
    """

    if groups is None:
        groups = {
            "Bleo main": ["control", "pbs", "2w", "4w"],
            "Metformin": ["pbsmet", "bmmet"],
            "OKN": ["pbsokn", "4wokn"],
        }

    if ylabel is None:
        if variable == "mod_Hertz":
            ylabel = "Indentation modulus (kPa)"
        elif variable == "tau_Visco":
            ylabel = "Tau (s)"
        else:
            ylabel = variable

    if title_prefix is None:
        if variable == "mod_Hertz":
            title_prefix = "NI repeat diagnostic: modulus"
        elif variable == "tau_Visco":
            title_prefix = "NI repeat diagnostic: tau"
        else:
            title_prefix = f"NI repeat diagnostic: {variable}"

    all_outputs = {}

    # ------------------------------------------------------------
    # Prepare line-scan data
    # ------------------------------------------------------------
    line_df = ni_raw.copy() if ni_raw is not None else pd.DataFrame()

    if not line_df.empty:
        if "SubtypeClean" not in line_df.columns:
            line_df = add_clean_subtype(line_df)

        if "RegionKey" in line_df.columns:
            line_df = line_df[
                line_df["RegionKey"].astype(str).str.lower().eq("line scan")
            ].copy()

        if "layer" in line_df.columns:
            # For bleo line scans, keep dermis only if layer labels exist.
            dermis_df = line_df[
                line_df["layer"].astype(str).str.lower().eq("dermis")
            ].copy()

            if not dermis_df.empty:
                line_df = dermis_df

        line_df["RegionDiagnostic"] = "Line"

    # ------------------------------------------------------------
    # Prepare grid data
    # ------------------------------------------------------------
    grid_df = ni_grid_raw.copy() if ni_grid_raw is not None else pd.DataFrame()

    if not grid_df.empty:
        if "SubtypeClean" not in grid_df.columns:
            grid_df = add_clean_subtype(grid_df)

        region_col = None
        for c in ["region", "Region", "RegionKey", "layer", "Layer"]:
            if c in grid_df.columns:
                region_col = c
                break

        if region_col is None:
            raise KeyError(
                "Could not find region column in ni_grid_raw. "
                "Expected one of: region, Region, RegionKey, layer, Layer."
            )

        def clean_grid_region_for_diag(x):
            key = str(x).strip().lower()
            key = key.replace("_", " ")
            key = key.replace("-", " ")
            key = " ".join(key.split())
            compact = key.replace(" ", "")

            mapping = {
                "lower dermis": "Lower",
                "lowerdermis": "Lower",
                "lower": "Lower",
                "dermis lower": "Lower",

                "upper dermis": "Upper",
                "upperdermis": "Upper",
                "upper": "Upper",
                "dermis upper": "Upper",
            }

            return mapping.get(key, mapping.get(compact, key))

        grid_df["RegionDiagnostic"] = grid_df[region_col].apply(clean_grid_region_for_diag)

        grid_df = grid_df[
            grid_df["RegionDiagnostic"].isin(["Lower", "Upper"])
        ].copy()

    # ------------------------------------------------------------
    # Combine point-level line + grid data
    # ------------------------------------------------------------
    keep_cols = [
        "Sample",
        "SampleKey",
        "Subtype",
        "SubtypeClean",
        "RegionDiagnostic",
        variable,
        "Rsq_Hertz",
        "Rsq_Visco",
        "Rsq_ViscoAna",
        "PointIndex",
        "PointKey",
        "x",
        "y",
    ]

    frames = []

    if not line_df.empty:
        frames.append(line_df[[c for c in keep_cols if c in line_df.columns]].copy())

    if not grid_df.empty:
        frames.append(grid_df[[c for c in keep_cols if c in grid_df.columns]].copy())

    if not frames:
        print("No NI line/grid data available for diagnostic plot.")
        return {}

    points = pd.concat(frames, ignore_index=True, sort=False)

    if "SampleKey" not in points.columns:
        if "Sample" in points.columns:
            points["SampleKey"] = points["Sample"].astype(str)
        else:
            points["SampleKey"] = np.arange(len(points)).astype(str)

    points = screen_ni_values_for_plot(
        points,
        variable=variable,
        rsq_min_hertz=rsq_min_hertz,
        rsq_min_visco=rsq_min_visco,
        hi_modulus_pa=hi_modulus_pa,
        std_devs=std_devs,
    )

    points = points.dropna(subset=["_PlotValue"]).copy()

    if points.empty:
        print(f"No screened NI values available for {variable}.")
        return {}
    
    # ------------------------------------------------------------
    # Choose Rsq column for scatter colouring
    # ------------------------------------------------------------
    rsq_col = None
    
    if variable == "mod_Hertz":
        for c in ["Rsq_Hertz", "Hertz - Rsq"]:
            if c in points.columns:
                rsq_col = c
                break
    
    elif variable == "tau_Visco":
        for c in ["Rsq_Visco", "Rsq_ViscoAna", "ViscoAna_r2"]:
            if c in points.columns:
                rsq_col = c
                break
    
    if rsq_col is not None:
        points[rsq_col] = pd.to_numeric(points[rsq_col], errors="coerce")
    else:
        print(f"No Rsq column found for {variable}; scatter points will be black.")

    # ------------------------------------------------------------
    # Per-repeat summary
    # ------------------------------------------------------------
    summary = (
        points
        .groupby(["SubtypeClean", "SampleKey", "RegionDiagnostic"], observed=True)["_PlotValue"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )

    summary["sem"] = summary["std"] / np.sqrt(summary["n"])

    if errorbar == "sem":
        summary["error"] = summary["sem"]
    else:
        summary["error"] = summary["std"]

    region_order = ["Line", "Lower", "Upper"]
    hatch_map = {
        "Line": "",
        "Lower": "//",
        "Upper": "..",
    }

    # ------------------------------------------------------------
    # Make one graph per group
    # ------------------------------------------------------------
    for group_name, group_subtypes in groups.items():

        group_subtypes_clean = [normalise_subtype(st) for st in group_subtypes]

        group_summary = summary[
            summary["SubtypeClean"].astype(str).isin(group_subtypes_clean)
        ].copy()

        group_points = points[
            points["SubtypeClean"].astype(str).isin(group_subtypes_clean)
        ].copy()

        if group_summary.empty:
            print(f"No NI diagnostic data for group: {group_name}")
            continue

        # Make repeat order: subtype order, then sample key
        repeat_lookup = (
            group_summary[["SubtypeClean", "SampleKey"]]
            .drop_duplicates()
            .copy()
        )

        repeat_lookup["SubtypeClean"] = pd.Categorical(
            repeat_lookup["SubtypeClean"],
            categories=group_subtypes_clean,
            ordered=True,
        )

        repeat_lookup = repeat_lookup.sort_values(
            ["SubtypeClean", "SampleKey"]
        ).reset_index(drop=True)

        repeat_labels = []
        repeat_keys = []

        for _, row in repeat_lookup.iterrows():
            st = str(row["SubtypeClean"])
            sample_key = str(row["SampleKey"])

            label = f"{SUBTYPE_LABELS.get(st, st)}\n{sample_key}"
            repeat_labels.append(label)
            repeat_keys.append((st, sample_key))

        n_repeats = len(repeat_keys)
        n_regions = len(region_order)

        fig_width = max(9, n_repeats * 1.2)
        fig, ax = plt.subplots(figsize=(fig_width, 5.5))

        x = np.arange(n_repeats)
        total_width = 0.78
        bar_width = total_width / n_regions

        for i, region in enumerate(region_order):
            offset = (i - (n_regions - 1) / 2) * bar_width

            means = []
            errors = []
            ns = []
            colours = []

            for st, sample_key in repeat_keys:
                row = group_summary[
                    (group_summary["SubtypeClean"].astype(str) == st)
                    & (group_summary["SampleKey"].astype(str) == sample_key)
                    & (group_summary["RegionDiagnostic"].astype(str) == region)
                ]

                if row.empty:
                    means.append(np.nan)
                    errors.append(np.nan)
                    ns.append(0)
                else:
                    means.append(float(row["mean"].iloc[0]))
                    errors.append(float(row["error"].iloc[0]))
                    ns.append(int(row["n"].iloc[0]))

                colours.append(NI_COLOURS.get(st, "grey"))

            bars = ax.bar(
                x + offset,
                means,
                yerr=errors,
                width=bar_width,
                color=colours,
                edgecolor="black",
                linewidth=0.8,
                capsize=3,
                label=region,
            )

            if hatches:
                for bar in bars:
                    bar.set_hatch(hatch_map.get(region, ""))

            # ----------------------------------------------------
            # Overlay individual points
            # ----------------------------------------------------
            if show_points:
                for j, (st, sample_key) in enumerate(repeat_keys):
                    vals = group_points[
                        (group_points["SubtypeClean"].astype(str) == st)
                        & (group_points["SampleKey"].astype(str) == sample_key)
                        & (group_points["RegionDiagnostic"].astype(str) == region)
                    ]["_PlotValue"].dropna().to_numpy(dtype=float)

                    if len(vals) == 0:
                        continue

                    jitter = np.linspace(
                        -bar_width * 0.22,
                        bar_width * 0.22,
                        len(vals),
                    )

                    if len(vals) == 1:
                        jitter = np.array([0.0])

                    xvals = np.full(len(vals), x[j] + offset) + jitter

                    if scatter_by_rsq and rsq_col is not None:
                        rsq_vals = group_points[
                            (group_points["SubtypeClean"].astype(str) == st)
                            & (group_points["SampleKey"].astype(str) == sample_key)
                            & (group_points["RegionDiagnostic"].astype(str) == region)
                        ][rsq_col].dropna().to_numpy(dtype=float)
                    
                        # Make sure Rsq array matches plotted values
                        point_df = group_points[
                            (group_points["SubtypeClean"].astype(str) == st)
                            & (group_points["SampleKey"].astype(str) == sample_key)
                            & (group_points["RegionDiagnostic"].astype(str) == region)
                        ][["_PlotValue", rsq_col]].dropna(subset=["_PlotValue"]).copy()
                    
                        vals = point_df["_PlotValue"].to_numpy(dtype=float)
                        rsq_vals = point_df[rsq_col].to_numpy(dtype=float)
                    
                        if len(vals) == 0:
                            continue
                    
                        jitter = np.linspace(
                            -bar_width * 0.22,
                            bar_width * 0.22,
                            len(vals),
                        )
                    
                        if len(vals) == 1:
                            jitter = np.array([0.0])
                    
                        xvals = np.full(len(vals), x[j] + offset) + jitter
                    
                        sc = ax.scatter(
                            xvals,
                            vals,
                            s=24,
                            c=rsq_vals,
                            cmap=rsq_cmap,
                            vmin=rsq_vmin,
                            vmax=rsq_vmax,
                            alpha=point_alpha,
                            edgecolors="black",
                            linewidths=0.25,
                            zorder=5,
                        )
                    
                    else:
                        ax.scatter(
                            xvals,
                            vals,
                            s=20,
                            color="black",
                            alpha=point_alpha,
                            zorder=5,
                        )

        ax.set_xticks(x)
        ax.set_xticklabels(repeat_labels, rotation=45, ha="right")

        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix} — {group_name}")

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.legend(title="Region", frameon=False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        if scatter_by_rsq and rsq_col is not None:
            sm = plt.cm.ScalarMappable(
                cmap=rsq_cmap,
                norm=plt.Normalize(vmin=rsq_vmin, vmax=rsq_vmax),
            )
            sm.set_array([])
        
            cbar = fig.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label("Rsq")

        fig.tight_layout()

        print("\n" + "=" * 80)
        print(f"NI repeat diagnostic summary — {group_name}")
        print("=" * 80)
        print(
            group_summary[
                ["SubtypeClean", "SampleKey", "RegionDiagnostic", "mean", "std", "sem", "n", "error"]
            ].to_string(index=False)
        )

        all_outputs[group_name] = {
            "fig": fig,
            "ax": ax,
            "summary": group_summary,
            "points": group_points,
        }

    return all_outputs
# =============================================================================
# NI PCA

def prepare_ni_pca_data(
    ni_raw,
    pca_vars=("mod_Hertz", "tau_Visco", "NormalisedPosition"),
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    standardise=True,
):
    """
    Prepare matrix for NI PCA from raw point-level data.

    subtypes_to_plot:
        None -> include all subtypes present in subtype_order
        list/tuple -> only include those subtypes, in subtype_order order

    Special variable:
        "NormalisedPosition" is created from point order per sample.

    Modulus-like variables are converted from Pa to kPa.
    """

    if ni_raw is None or ni_raw.empty:
        return pd.DataFrame(), None

    if subtypes_to_plot is None:
        wanted_subtypes = [str(st).strip() for st in subtype_order]
    else:
        wanted_set = {str(st).strip() for st in subtypes_to_plot}
        wanted_subtypes = [
            str(st).strip()
            for st in subtype_order
            if str(st).strip() in wanted_set
        ]

        wanted_subtypes += [
            str(st).strip()
            for st in subtypes_to_plot
            if str(st).strip() not in wanted_subtypes
        ]

    needed_vars = [v for v in pca_vars if v != "NormalisedPosition"]

    # Use first variable just to create NormalisedPosition through existing helper.
    base_var = needed_vars[0] if needed_vars else "mod_Hertz"

    d = prepare_ni_raw_line_data(
        ni_raw=ni_raw,
        variable=base_var,
        subtype_order=subtype_order,
        convert_modulus_to_kpa=False,
    )

    if d.empty:
        return pd.DataFrame(), None

    d = d[d["SubtypeClean"].astype(str).isin(wanted_subtypes)].copy()

    if d.empty:
        print("No matching NI subtypes available for PCA.")
        print("Requested:", subtypes_to_plot)
        print("Available:", sorted(prepare_ni_raw_line_data(
            ni_raw=ni_raw,
            variable=base_var,
            subtype_order=subtype_order,
            convert_modulus_to_kpa=False,
        )["SubtypeClean"].astype(str).unique()))
        return pd.DataFrame(), None

    for var in needed_vars:
        if var not in d.columns:
            raise KeyError(f"Missing PCA variable in ni_raw: {var}")

        d[var] = pd.to_numeric(d[var], errors="coerce")

        if var in {
            "mod_Hertz",
            "mod_OP",
            "E0_Visco",
            "Einf_Visco",
            "G0_Visco",
            "G1_Visco",
            "Eff_file",
            "mod_file",
        }:
            d[var] = d[var] / 1000

    pca_cols = []

    for var in pca_vars:
        if var == "NormalisedPosition":
            pca_cols.append("NormalisedPosition")
        else:
            pca_cols.append(var)

    d = d.dropna(subset=pca_cols + ["SubtypeClean"]).copy()

    if d.empty:
        print("No complete NI rows available for PCA after dropping missing values.")
        return pd.DataFrame(), None

    X = d[pca_cols].to_numpy(dtype=float)

    if X.shape[0] < 3:
        print("Not enough NI rows for PCA after filtering.")
        return pd.DataFrame(), None

    if standardise:
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=1)
        sd = np.where(sd == 0, 1.0, sd)
        X_used = (X - mu) / sd
    else:
        mu = np.zeros(X.shape[1])
        sd = np.ones(X.shape[1])
        X_used = X.copy()

    Xc = X_used - np.mean(X_used, axis=0)

    U, S, VT = np.linalg.svd(Xc, full_matrices=False)

    scores = U * S
    loadings = VT.T

    var = (S ** 2) / (Xc.shape[0] - 1)
    evr = var / np.sum(var)

    d["PC1"] = scores[:, 0]

    if scores.shape[1] > 1:
        d["PC2"] = scores[:, 1]
    else:
        d["PC2"] = np.nan

    if scores.shape[1] > 2:
        d["PC3"] = scores[:, 2]

    pca_info = {
        "pca_vars": pca_vars,
        "pca_cols": pca_cols,
        "standardised": standardise,
        "subtypes_used": wanted_subtypes,
        "mu": mu,
        "sd": sd,
        "loadings": loadings,
        "explained_variance_ratio": evr,
    }

    return d, pca_info

def plot_ni_pca_scores(
    pca_df,
    pca_info,
    subtype_order=SUBTYPE_ORDER,
    colours=NI_COLOURS,
    labels=NI_LABELS,
    alpha=0.35,
    point_size=14,
    title=None,
):
    """
    PC1 vs PC2 scatter coloured by subtype.
    """

    if pca_df is None or pca_df.empty:
        print("No PCA data to plot.")
        return None, None

    evr = pca_info["explained_variance_ratio"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for subtype in subtype_order:
        d_sub = pca_df[pca_df["SubtypeClean"].astype(str) == subtype]

        if d_sub.empty:
            continue

        ax.scatter(
            d_sub["PC1"],
            d_sub["PC2"],
            color=colours.get(subtype, "grey"),
            alpha=alpha,
            s=point_size,
            label=labels.get(subtype, subtype),
        )

    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
    ax.set_title(title or f"NI PCA: {', '.join(pca_info['pca_vars'])}")
    ax.legend(frameon=False)
    ax.grid(False)

    plt.tight_layout()
    plt.show()

    return fig, ax

def plot_ni_pca_by_position_with_markers(
    pca_df,
    pca_info,
    subtype_order=SUBTYPE_ORDER,
    markers=NI_MARKERS,
    cmap="jet",
    alpha=0.60,
    point_size=35,
    title=None,
):
    """
    Single PCA plot:
        PC1 vs PC2
        colour = normalised position
        marker = subtype
    """

    if pca_df is None or pca_df.empty:
        print("No PCA data to plot.")
        return None, None

    evr = pca_info["explained_variance_ratio"]

    fig, ax = plt.subplots(figsize=(7, 5.8))
    last_scatter = None

    for subtype in subtype_order:
        d_sub = pca_df[pca_df["SubtypeClean"].astype(str) == subtype]

        if d_sub.empty:
            continue

        last_scatter = ax.scatter(
            d_sub["PC1"],
            d_sub["PC2"],
            c=d_sub["NormalisedPosition"],
            cmap=cmap,
            vmin=0,
            vmax=100,
            marker=markers.get(subtype, "o"),
            alpha=alpha,
            s=point_size,
            label=NI_LABELS.get(subtype, subtype),
        )

    if last_scatter is not None:
        cbar = plt.colorbar(last_scatter, ax=ax)
        cbar.set_label("Normalised position through dermis (%)")

    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
    ax.set_title(title or "NI PCA coloured by spatial position")
    ax.legend(title="Subtype", frameon=False)
    ax.grid(False)

    plt.tight_layout()
    plt.show()

    return fig, ax

def plot_ni_pca_by_position_with_alpha(
    pca_df,
    pca_info,
    subtype_order=SUBTYPE_ORDER,
    colours=NI_COLOURS,
    markers=NI_MARKERS,
    labels=NI_LABELS,
    alpha_min=0.20,
    alpha_max=0.90,
    point_size=35,
    title=None,
):
    """
    Single PCA plot:
        PC1 vs PC2
        colour = subtype
        marker = subtype
        transparency = normalised position through dermis

    NormalisedPosition:
        0%   -> alpha_min
        100% -> alpha_max
    """

    if pca_df is None or pca_df.empty:
        print("No PCA data to plot.")
        return None, None

    evr = pca_info["explained_variance_ratio"]

    fig, ax = plt.subplots(figsize=(7, 5.8))

    for subtype in subtype_order:
        d_sub = pca_df[pca_df["SubtypeClean"].astype(str) == subtype].copy()

        if d_sub.empty:
            continue

        d_sub["NormalisedPosition"] = pd.to_numeric(
            d_sub["NormalisedPosition"],
            errors="coerce",
        )

        d_sub = d_sub.dropna(subset=["PC1", "PC2", "NormalisedPosition"]).copy()

        if d_sub.empty:
            continue

        pos = d_sub["NormalisedPosition"].clip(0, 100).to_numpy(float)
        alphas = alpha_min + (pos / 100.0) * (alpha_max - alpha_min)

        colour = colours.get(subtype, "grey")
        marker = markers.get(subtype, "o")
        label = labels.get(subtype, subtype)

        # Scatter does not accept per-point alpha cleanly with a single colour,
        # so plot each point with its own alpha.
        first = True
        for _, row in d_sub.iterrows():
            p = float(row["NormalisedPosition"])
            p = min(max(p, 0.0), 100.0)
            a = alpha_min + (p / 100.0) * (alpha_max - alpha_min)

            ax.scatter(
                row["PC1"],
                row["PC2"],
                color=colour,
                marker=marker,
                alpha=a,
                s=point_size,
                label=label if first else None,
                linewidths=1.0 if marker in {"x", "+", "*"} else 0.5,
                edgecolors="none" if marker not in {"x", "+", "*"} else colour,
            )
            first = False

    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
    ax.set_title(title or "NI PCA coloured by subtype, alpha by spatial position")
    ax.legend(title="Subtype", frameon=False)
    ax.grid(False)

    plt.tight_layout()
    plt.show()

    return fig, ax

def plot_ni_pca_by_position_faceted(
    pca_df,
    pca_info,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    markers=NI_MARKERS,
    labels=NI_LABELS,
    cmap="jet",
    alpha=0.70,
    point_size=18,
    ncols=3,
    figsize_per_panel=(4.2, 3.8),
    title=None,
):
    """
    Faceted PCA plot:
        one panel per subtype
        colour = normalised position
        marker = subtype
        shared PC1/PC2 scales

    figsize_per_panel controls the size/aspect ratio of each subplot.
    """

    if pca_df is None or pca_df.empty:
        print("No PCA data to plot.")
        return None, None

    if pca_info is None:
        print("No PCA info supplied.")
        return None, None

    evr = pca_info["explained_variance_ratio"]

    if subtypes_to_plot is not None:
        subtype_order = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)

    present_subtypes = [
        st for st in subtype_order
        if st in set(pca_df["SubtypeClean"].astype(str))
    ]

    if not present_subtypes:
        print("No matching subtypes in PCA dataframe.")
        return None, None

    n_panels = len(present_subtypes)
    ncols = min(int(ncols), n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes = np.atleast_1d(axes).ravel()

    pc1 = pd.to_numeric(pca_df["PC1"], errors="coerce").to_numpy(float)
    pc2 = pd.to_numeric(pca_df["PC2"], errors="coerce").to_numpy(float)

    pc1 = pc1[np.isfinite(pc1)]
    pc2 = pc2[np.isfinite(pc2)]

    if pc1.size == 0 or pc2.size == 0:
        print("No finite PC1/PC2 values to plot.")
        return None, None

    x_range = np.nanmax(pc1) - np.nanmin(pc1)
    y_range = np.nanmax(pc2) - np.nanmin(pc2)

    xpad = 0.06 * x_range if x_range > 0 else 1
    ypad = 0.06 * y_range if y_range > 0 else 1

    xlim = (np.nanmin(pc1) - xpad, np.nanmax(pc1) + xpad)
    ylim = (np.nanmin(pc2) - ypad, np.nanmax(pc2) + ypad)

    norm = plt.Normalize(vmin=0, vmax=100)
    cm = plt.get_cmap(cmap)

    for ax, subtype in zip(axes, present_subtypes):
        d_sub = pca_df[pca_df["SubtypeClean"].astype(str) == subtype].copy()

        if d_sub.empty:
            ax.set_title(f"{labels.get(subtype, subtype)} (no data)")
            ax.axis("off")
            continue

        ax.scatter(
            d_sub["PC1"],
            d_sub["PC2"],
            c=d_sub["NormalisedPosition"],
            cmap=cm,
            norm=norm,
            marker=markers.get(subtype, "o"),
            alpha=alpha,
            s=point_size,
            edgecolors="none",
        )

        ax.set_title(labels.get(subtype, subtype))
        ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(False)

    for ax in axes[n_panels:]:
        ax.axis("off")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes[:n_panels],
        location="right",
        shrink=0.85,
        pad=0.03,
    )
    cbar.set_label("Normalised position through dermis (%)")

    fig.suptitle(
        title or "NI PCA coloured by spatial position, faceted by subtype",
        y=1.03,
    )

    plt.show()

    return fig, axes

def plot_ni_grid_pca_by_region_faceted(
    pca_df,
    pca_info,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    markers=NI_MARKERS,
    labels=NI_LABELS,
    alpha=0.70,
    point_size=20,
    ncols=3,
    figsize_per_panel=(4.2, 3.8),
    title=None,
):
    """
    Faceted PCA plot for NI grid scans.

    One panel per subtype.

    Axes:
        PC1 vs PC2

    Colour:
        lower dermis = blue
        upper dermis = red

    Works with grid PCA dataframe from:
        prepare_ni_grid_region_pca_data()
    """

    if pca_df is None or pca_df.empty:
        print("No PCA data to plot.")
        return None, None

    if pca_info is None:
        print("No PCA info supplied.")
        return None, None

    evr = pca_info["explained_variance_ratio"]

    if subtypes_to_plot is not None:
        subtype_order = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)

    present_subtypes = [
        st for st in subtype_order
        if st in set(pca_df["SubtypeClean"].astype(str))
    ]

    if not present_subtypes:
        print("No matching subtypes in PCA dataframe.")
        print("Requested:", subtypes_to_plot)
        print("Available:", sorted(pca_df["SubtypeClean"].astype(str).unique()))
        return None, None

    # ------------------------------------------------------------
    # Resolve region column
    # ------------------------------------------------------------
    region_col = None

    for c in ["GridRegionClean", "RegionDiagnostic", "region", "Region", "RegionKey", "layer", "Layer"]:
        if c in pca_df.columns:
            region_col = c
            break

    if region_col is None and "RegionCode" not in pca_df.columns:
        raise KeyError(
            "Could not find region information. Expected one of: "
            "GridRegionClean, RegionDiagnostic, region, Region, RegionKey, layer, Layer, or RegionCode."
        )

    d = pca_df.copy()

    if region_col is not None:
        def clean_grid_region_for_plot(x):
            key = str(x).strip().lower()
            key = key.replace("_", " ")
            key = key.replace("-", " ")
            key = " ".join(key.split())
            compact = key.replace(" ", "")

            mapping = {
                "lower dermis": "lower dermis",
                "lowerdermis": "lower dermis",
                "lower": "lower dermis",
                "dermis lower": "lower dermis",

                "upper dermis": "upper dermis",
                "upperdermis": "upper dermis",
                "upper": "upper dermis",
                "dermis upper": "upper dermis",
            }

            return mapping.get(key, mapping.get(compact, key))

        d["PCARegionClean"] = d[region_col].apply(clean_grid_region_for_plot)

    else:
        # Fallback from RegionCode
        d["RegionCode"] = pd.to_numeric(d["RegionCode"], errors="coerce")
        d["PCARegionClean"] = d["RegionCode"].map(
            {
                0: "lower dermis",
                1: "upper dermis",
            }
        )

    region_colours = {
        "lower dermis": "blue",
        "upper dermis": "red",
    }

    region_labels = {
        "lower dermis": "Lower dermis",
        "upper dermis": "Upper dermis",
    }

    # ------------------------------------------------------------
    # Shared PC limits
    # ------------------------------------------------------------
    pc1 = pd.to_numeric(d["PC1"], errors="coerce").to_numpy(float)
    pc2 = pd.to_numeric(d["PC2"], errors="coerce").to_numpy(float)

    pc1 = pc1[np.isfinite(pc1)]
    pc2 = pc2[np.isfinite(pc2)]

    if pc1.size == 0 or pc2.size == 0:
        print("No finite PC1/PC2 values to plot.")
        return None, None

    x_range = np.nanmax(pc1) - np.nanmin(pc1)
    y_range = np.nanmax(pc2) - np.nanmin(pc2)

    xpad = 0.06 * x_range if x_range > 0 else 1
    ypad = 0.06 * y_range if y_range > 0 else 1

    xlim = (np.nanmin(pc1) - xpad, np.nanmax(pc1) + xpad)
    ylim = (np.nanmin(pc2) - ypad, np.nanmax(pc2) + ypad)

    # ------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------
    n_panels = len(present_subtypes)
    ncols = min(int(ncols), n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes = np.atleast_1d(axes).ravel()

    # ------------------------------------------------------------
    # Plot one subtype per panel
    # ------------------------------------------------------------
    for ax, subtype in zip(axes, present_subtypes):
        d_sub = d[d["SubtypeClean"].astype(str) == subtype].copy()

        if d_sub.empty:
            ax.set_title(f"{labels.get(subtype, subtype)} (no data)")
            ax.axis("off")
            continue

        for region in ["lower dermis", "upper dermis"]:
            d_reg = d_sub[d_sub["PCARegionClean"].astype(str) == region].copy()

            if d_reg.empty:
                continue

            ax.scatter(
                d_reg["PC1"],
                d_reg["PC2"],
                color=region_colours.get(region, "grey"),
                marker=markers.get(subtype, "o"),
                alpha=alpha,
                s=point_size,
                edgecolors="none",
                label=region_labels.get(region, region),
            )

        ax.set_title(labels.get(subtype, subtype))
        ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(False)

    for ax in axes[n_panels:]:
        ax.axis("off")

    # ------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="blue",
            markeredgecolor="none",
            markersize=7,
            label="Lower dermis",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="red",
            markeredgecolor="none",
            markersize=7,
            label="Upper dermis",
        ),
    ]

    # fig.legend(
    #     handles=handles,
    #     title="Grid region",
    #     loc="center right",
    #     frameon=False,
    #     bbox_to_anchor=(1.02, 0.5),
    # )

    # fig.suptitle(
    #     title or "NI grid PCA coloured by upper/lower dermis, faceted by subtype",
    #     y=1.03,
    # )

    plt.show()

    return fig, axes

def _fit_simple_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if len(x) < 3:
        return np.nan

    slope, intercept = np.polyfit(x, y, deg=1)

    return float(slope)

def _mean_ci95(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return np.nan, np.nan, np.nan, 0

    mean = float(np.mean(vals))

    if vals.size == 1:
        return mean, np.nan, np.nan, 1

    sd = float(np.std(vals, ddof=1))
    sem = sd / np.sqrt(vals.size)
    ci = 1.96 * sem

    return mean, mean - ci, mean + ci, int(vals.size)

def calculate_ni_pc_slopes(
    pca_df,
    pcs=("PC1", "PC2"),
    subtype_order=SUBTYPE_ORDER,
):
    """
    Calculate PC slope vs normalised position for each sample.

    Returns
    -------
    slopes_df : one row per sample/PC
    summary_df : one row per subtype/PC
    """

    if pca_df is None or pca_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    sample_col = "SampleKey" if "SampleKey" in pca_df.columns else "Sample"

    rows = []

    for (subtype, sample), d_sample in pca_df.groupby(["SubtypeClean", sample_col], observed=True):
        for pc in pcs:
            slope = _fit_simple_slope(
                d_sample["NormalisedPosition"],
                d_sample[pc],
            )

            rows.append({
                "SubtypeClean": str(subtype),
                "Sample": sample,
                "PC": pc,
                "Slope": slope,
                "n": int(len(d_sample)),
            })

    slopes_df = pd.DataFrame(rows)

    summary_rows = []

    for subtype in subtype_order:
        for pc in pcs:
            vals = slopes_df.loc[
                (slopes_df["SubtypeClean"] == subtype)
                & (slopes_df["PC"] == pc),
                "Slope",
            ].to_numpy(dtype=float)

            mean, lo, hi, n = _mean_ci95(vals)

            summary_rows.append({
                "SubtypeClean": subtype,
                "PC": pc,
                "SlopeMean": mean,
                "CI95_Low": lo,
                "CI95_High": hi,
                "n_samples": n,
            })

    summary_df = pd.DataFrame(summary_rows)

    return slopes_df, summary_df

def plot_ni_pc_slope_points(
    slopes_df,
    summary_df,
    pc="PC1",
    subtype_order=SUBTYPE_ORDER,
    colours=NI_COLOURS,
    labels=NI_LABELS,
    title=None,
    ylabel=None,
    jitter=0.055,
    point_size=36,
    mean_size=70,
):
    """
    Plot per-sample PC slope for one PC.

    x = subtype
    y = slope of PC vs normalised position

    Shows:
        coloured scatter = individual sample slopes
        black dot = mean slope
        black error bar = 95% CI
    """

    if slopes_df is None or slopes_df.empty:
        print("No PC slope data to plot.")
        return None, None

    d = slopes_df[slopes_df["PC"] == pc].copy()
    s = summary_df[summary_df["PC"] == pc].copy()

    if d.empty:
        print(f"No slope data for {pc}.")
        return None, None

    fig, ax = plt.subplots(figsize=(8, 5.2))
    rng = np.random.default_rng(0)

    x = np.arange(len(subtype_order))

    for i, subtype in enumerate(subtype_order):
        vals = d.loc[d["SubtypeClean"] == subtype, "Slope"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]

        if vals.size:
            xj = x[i] + rng.normal(0, jitter, size=vals.size)

            ax.scatter(
                xj,
                vals,
                color=colours.get(subtype, "grey"),
                s=point_size,
                alpha=0.75,
                linewidths=0,
                zorder=2,
            )

        row = s[s["SubtypeClean"] == subtype]

        if not row.empty:
            mean = float(row["SlopeMean"].iloc[0])
            lo = float(row["CI95_Low"].iloc[0])
            hi = float(row["CI95_High"].iloc[0])

            if np.isfinite(mean):
                yerr = None

                if np.isfinite(lo) and np.isfinite(hi):
                    yerr = [[mean - lo], [hi - mean]]

                ax.errorbar(
                    x[i],
                    mean,
                    yerr=yerr,
                    fmt="o",
                    color="black",
                    markersize=np.sqrt(mean_size),
                    capsize=5,
                    linewidth=1.8,
                    zorder=4,
                )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [labels.get(st, st) for st in subtype_order],
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(ylabel or f"Slope of {pc} vs normalised position")
    ax.set_title(title or f"{pc} slope per sample")
    ax.grid(False)

    plt.tight_layout()
    plt.show()

    return fig, ax

def print_ni_pca_loadings(ni_pca_info, pca_vars=None):
    """
    Print explained variance and PCA loadings from NI PCA info.

    Compatible with both formats:

    Format 1:
        pca_cols
        loadings
        explained_variance_ratio

    Format 2:
        pca
        feature_cols
        components
        explained_variance_ratio
    """

    if ni_pca_info is None or len(ni_pca_info) == 0:
        print("No PCA info available.")
        return pd.DataFrame()

    explained = ni_pca_info.get("explained_variance_ratio", None)

    # ------------------------------------------------------------
    # Get feature names
    # ------------------------------------------------------------
    feature_cols = ni_pca_info.get("pca_cols", None)

    if feature_cols is None:
        feature_cols = ni_pca_info.get("feature_cols", None)

    if feature_cols is None:
        feature_cols = ni_pca_info.get("pca_vars", None)

    if feature_cols is None and pca_vars is not None:
        feature_cols = list(pca_vars)

    # ------------------------------------------------------------
    # Get loadings/components
    # ------------------------------------------------------------
    loadings = ni_pca_info.get("loadings", None)

    if loadings is None:
        components = ni_pca_info.get("components", None)

        if components is not None:
            # sklearn PCA components are shape:
            #     n_components x n_features
            # We transpose to:
            #     n_features x n_components
            loadings = np.asarray(components, dtype=float).T

    if loadings is None:
        pca = ni_pca_info.get("pca", None)

        if pca is not None:
            loadings = np.asarray(pca.components_, dtype=float).T

            if explained is None:
                explained = pca.explained_variance_ratio_

    if explained is None or loadings is None:
        print("Could not find explained variance or loadings in ni_pca_info.")
        print("Available ni_pca_info keys:", list(ni_pca_info.keys()))
        return pd.DataFrame()

    explained = np.asarray(explained, dtype=float)

    # ------------------------------------------------------------
    # Convert loadings to dataframe
    # ------------------------------------------------------------
    if isinstance(loadings, pd.DataFrame):
        loadings_df = loadings.copy()

    else:
        loadings_arr = np.asarray(loadings, dtype=float)

        if feature_cols is None:
            feature_cols = [f"Variable {i + 1}" for i in range(loadings_arr.shape[0])]

        loadings_df = pd.DataFrame(
            loadings_arr,
            index=list(feature_cols),
            columns=[f"PC{i}" for i in range(1, loadings_arr.shape[1] + 1)],
        )

    # Ensure clean PC names
    loadings_df.columns = [f"PC{i}" for i in range(1, loadings_df.shape[1] + 1)]

    # ------------------------------------------------------------
    # Print explained variance
    # ------------------------------------------------------------
    print("\nNI PCA explained variance:")
    for i, var in enumerate(explained, start=1):
        print(f"PC{i}: {var * 100:.2f}%")

    # ------------------------------------------------------------
    # Print loadings
    # ------------------------------------------------------------
    print("\nNI PCA loadings:")
    print(loadings_df.round(3).to_string())

    # ------------------------------------------------------------
    # Print ranked contributors
    # ------------------------------------------------------------
    print("\nMain contributors by absolute loading:")

    for pc in loadings_df.columns:
        ranked = (
            loadings_df[pc]
            .abs()
            .sort_values(ascending=False)
            .reset_index()
        )

        ranked.columns = ["Variable", "AbsLoading"]

        print(f"\n{pc}:")
        print(ranked.to_string(index=False))

    return loadings_df

def prepare_ni_grid_region_pca_data(
    ni_grid_raw,
    pca_vars=("mod_Hertz", "tau_Visco", "RegionCode"),
    subtypes_to_plot=None,
    subtype_order=SUBTYPE_ORDER,
    standardise=True,
):
    """
    Prepare NI PCA data from upper/lower dermis grid data.

    Uses point-level screened grid data.

    RegionCode:
        lower dermis = 0
        upper dermis = 1
    """

    if ni_grid_raw is None or ni_grid_raw.empty:
        print("No NI grid raw data available for PCA.")
        return pd.DataFrame(), {}

    df = ni_grid_raw.copy()

    if "SubtypeClean" not in df.columns:
        df = add_clean_subtype(df)

    if subtypes_to_plot is not None:
        wanted = [normalise_subtype(st) for st in subtypes_to_plot]
    else:
        wanted = [normalise_subtype(st) for st in subtype_order]

    df = df[df["SubtypeClean"].astype(str).isin(wanted)].copy()

    if df.empty:
        print("No NI grid data left after subtype filtering.")
        return pd.DataFrame(), {}

    # ------------------------------------------------------------
    # Find region column
    # ------------------------------------------------------------
    region_col = None

    for c in ["region", "Region", "RegionKey", "layer", "Layer", "GridRegionClean"]:
        if c in df.columns:
            region_col = c
            break

    if region_col is None:
        raise KeyError(
            "Could not find grid region column. Expected one of: "
            "region, Region, RegionKey, layer, Layer, GridRegionClean."
        )

    def clean_grid_region(x):
        key = str(x).strip().lower()
        key = key.replace("_", " ")
        key = key.replace("-", " ")
        key = " ".join(key.split())
        compact = key.replace(" ", "")

        mapping = {
            "lower dermis": "lower dermis",
            "lowerdermis": "lower dermis",
            "lower": "lower dermis",
            "dermis lower": "lower dermis",

            "upper dermis": "upper dermis",
            "upperdermis": "upper dermis",
            "upper": "upper dermis",
            "dermis upper": "upper dermis",
        }

        return mapping.get(key, mapping.get(compact, key))

    df["GridRegionClean"] = df[region_col].apply(clean_grid_region)

    df = df[df["GridRegionClean"].isin(["lower dermis", "upper dermis"])].copy()

    if df.empty:
        print("No lower/upper dermis grid rows available for PCA.")
        return pd.DataFrame(), {}

    df["RegionCode"] = df["GridRegionClean"].map(
        {
            "lower dermis": 0,
            "upper dermis": 1,
        }
    )

    # ------------------------------------------------------------
    # Numeric variables
    # ------------------------------------------------------------
    for var in pca_vars:
        if var not in df.columns:
            raise KeyError(
                f"PCA variable '{var}' not found in NI grid dataframe. "
                f"Available columns: {list(df.columns)}"
            )

        df[var] = pd.to_numeric(df[var], errors="coerce")

    df = df.dropna(subset=list(pca_vars)).copy()

    if df.empty:
        print("No complete rows available for NI grid PCA after dropping NaNs.")
        return pd.DataFrame(), {}

    # Convert modulus-like values from Pa to kPa before PCA
    for var in pca_vars:
        if var in ["mod_Hertz", "E0_Visco", "Einf_Visco", "G0_Visco", "G1_Visco"]:
            df[var] = df[var] / 1000.0

    X = df[list(pca_vars)].to_numpy(dtype=float)

    if standardise:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = None
        X_scaled = X

    pca = PCA()
    pcs = pca.fit_transform(X_scaled)

    for i in range(pcs.shape[1]):
        df[f"PC{i + 1}"] = pcs[:, i]

    ni_pca_info = {
        "pca": pca,
        "scaler": scaler,
        "feature_cols": list(pca_vars),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "components": pca.components_,
        "standardise": standardise,
    }

    print("\nNI grid PCA prepared:")
    print(f"Rows used: {len(df)}")
    print(f"Variables: {list(pca_vars)}")
    print(f"Standardised: {standardise}")

    print("\nRows by subtype and region:")
    print(
        df.groupby(["SubtypeClean", "GridRegionClean"])
        .size()
        .to_string()
    )

    return df, ni_pca_info

# =============================================================================
# cell plotting

def prepare_cell_raw(cell_raw, value_col=CELL_VALUE_COL):
    """
    Prepare spatial cell-density data for plotting.

    Adds a synthetic control for dermis regions:
        control dermis_sub = all dermis_sub values pooled from d4-d21
        control dermis_epi = all dermis_epi values pooled from d4-d21

    No control wound values are created.
    """

    if cell_raw is None or cell_raw.empty:
        return pd.DataFrame()

    d = cell_raw.copy()
    d = add_clean_subtype(d)

    if "Region" not in d.columns:
        raise KeyError("cell_raw must contain a 'Region' column.")

    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d["Region"] = d["Region"].astype(str).str.strip()

    d = d.dropna(subset=[value_col, "Region", "SubtypeClean"]).copy()

    control_rows = []

    for region in ["dermis_sub", "dermis_epi"]:
        r = d[d["Region"] == region].copy()

        if r.empty:
            continue

        r["Subtype"] = "control"
        r["SubtypeClean"] = "control"
        r["OriginalCondition"] = "derived_control"
        r["Sample"] = [
            f"control_{region}_cell_{i + 1}"
            for i in range(len(r))
        ]

        control_rows.append(r)

    if control_rows:
        d = pd.concat([pd.concat(control_rows, ignore_index=True), d], ignore_index=True, sort=False)

    d["SubtypeClean"] = pd.Categorical(
        d["SubtypeClean"],
        categories=SUBTYPE_ORDER,
        ordered=True,
    )

    return d.sort_values(["SubtypeClean", "Region"]).reset_index(drop=True)

def resolve_subtypes_to_plot(subtype_order, subtypes_to_plot=None):
    if subtypes_to_plot is None:
        return [str(st).strip() for st in subtype_order]

    wanted = [str(st).strip() for st in subtypes_to_plot]
    wanted_set = set(wanted)

    ordered = [
        str(st).strip()
        for st in subtype_order
        if str(st).strip() in wanted_set
    ]

    ordered += [
        st for st in wanted
        if st not in ordered
    ]

    return ordered

def summarise_cell_by_region(
    cell_df,
    value_col=CELL_VALUE_COL,
    region_col="Region",
):
    """
    Summarise cell density by subtype and region.
    """

    if cell_df is None or cell_df.empty:
        return pd.DataFrame()

    d = cell_df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    out = (
        d.groupby(["SubtypeClean", region_col], observed=True)[value_col]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .rename(columns={region_col: "Region"})
    )

    return out

def _plot_grouped_cell_bars(
    summary,
    points,
    regions,
    title,
    ylabel=f"Fibroblasts / mm²",
    subtype_order=SUBTYPE_ORDER,
    colours=NI_COLOURS,
    labels=NI_LABELS,
    region_labels=CELL_REGION_LABELS,
    ylim=None,
    scatter=True,
    jitter=0.045,
    point_size=24,
    alpha_range=(0.95, 0.35),
    hatches=True,
    region_hatches=None,
    hatch_linewidth=0.7,
):
    """
    Generic grouped bar chart for cell data.

    x = subtype
    grouped bars = regions
    bars = mean
    error bars = SD
    points = raw values

    hatches=True:
        Adds region-specific hatch patterns and shows them in the legend.

    hatches=False:
        Uses colour/alpha only.
    """

    if summary is None or summary.empty:
        print("No cell summary data to plot.")
        return None, None

    if region_hatches is None:
        region_hatches = {
            "dermis_sub": "///",
            "dermis_epi": "\\\\\\",
            "wound_sub": "...",
            "wound_epi": "xxx",
            "dermis": "///",
            "wound": "xxx",
        }

    present_subtypes = [
        st for st in subtype_order
        if st in set(summary["SubtypeClean"].astype(str))
    ]

    if not present_subtypes:
        print("No matching cell subtypes to plot.")
        return None, None

    x = np.arange(len(present_subtypes))
    width = 0.82 / max(len(regions), 1)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    rng = np.random.default_rng(0)

    alpha_start, alpha_end = alpha_range
    region_alphas = np.linspace(alpha_start, alpha_end, len(regions))

    old_hatch_lw = plt.rcParams.get("hatch.linewidth", 1.0)
    plt.rcParams["hatch.linewidth"] = hatch_linewidth

    try:
        for j, region in enumerate(regions):
            offset = (j - (len(regions) - 1) / 2) * width

            means = []
            errs = []

            for subtype in present_subtypes:
                row = summary[
                    (summary["SubtypeClean"].astype(str) == subtype)
                    & (summary["Region"].astype(str) == region)
                ]

                if row.empty:
                    means.append(np.nan)
                    errs.append(np.nan)
                else:
                    means.append(float(row["mean"].iloc[0]))
                    errs.append(float(row["std"].iloc[0]) if pd.notna(row["std"].iloc[0]) else np.nan)

            means = np.asarray(means, dtype=float)
            errs = np.asarray(errs, dtype=float)
            bar_cols = [colours.get(st, "grey") for st in present_subtypes]

            bars = ax.bar(
                x + offset,
                means,
                width=width,
                color=bar_cols,
                alpha=region_alphas[j],
                edgecolor="black",
                linewidth=0.6,
                zorder=2,
            )

            if hatches:
                for bar, subtype in zip(bars, present_subtypes):
                    hatch_colour = "black" if subtype == "control" else "lightgrey"

                    ax.bar(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        width=bar.get_width(),
                        color="none",
                        edgecolor=hatch_colour,
                        linewidth=0.0,
                        hatch=region_hatches.get(region, ""),
                        alpha=1.0,
                        zorder=3,
                    )

            ax.errorbar(
                x + offset,
                means,
                yerr=errs,
                fmt="none",
                ecolor="black",
                capsize=3,
                linewidth=1.1,
                zorder=4,
            )

            if scatter and points is not None and not points.empty:
                for i, subtype in enumerate(present_subtypes):
                    vals = points.loc[
                        (points["SubtypeClean"].astype(str) == subtype)
                        & (points["Region"].astype(str) == region),
                        CELL_VALUE_COL,
                    ].to_numpy(dtype=float)

                    vals = vals[np.isfinite(vals)]

                    if vals.size == 0:
                        continue

                    xj = x[i] + offset + rng.normal(0, jitter, size=vals.size)

                    # ax.scatter(
                    #     xj,
                    #     vals,
                    #     color="black",
                    #     s=point_size,
                    #     alpha=0.65,
                    #     linewidths=0,
                    #     zorder=5,
                    # )

    finally:
        plt.rcParams["hatch.linewidth"] = old_hatch_lw

    ax.set_xticks(x)
    ax.set_xticklabels(
        [labels.get(st, st) for st in present_subtypes],
        rotation=0,
        ha="right",
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)

    legend_handles = []
    for j, region in enumerate(regions):
        if hatches:
            legend_handles.append(
                Patch(
                    facecolor="white",
                    edgecolor="black",
                    hatch=region_hatches.get(region, ""),
                    label=region_labels.get(region, region),
                    alpha=1.0,
                )
            )
        else:
            legend_handles.append(
                Patch(
                    facecolor="grey",
                    edgecolor="black",
                    label=region_labels.get(region, region),
                    alpha=region_alphas[j],
                )
            )

    ax.legend(handles=legend_handles, frameon=False)

    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()

    return fig, ax

def _plot_cell_shift_bars(
    shift_df,
    summary,
    shift_order,
    title,
    ylabel,
    subtype_order=SUBTYPE_ORDER,
    colours=NI_COLOURS,
    labels=NI_LABELS,
    shift_labels=CELL_REGION_LABELS,
    ylim=None,
    jitter=0.045,
    point_size=24,
    alpha_range=(0.95, 0.45),
    hatches=True,
    shift_hatches=None,
    hatch_linewidth=0.7,
):
    """
    Generic grouped bar plot for cell % shifts.

    hatches=True:
        Adds shift-specific hatch patterns and shows them in the legend.

    hatches=False:
        Uses colour/alpha only.
    """

    if summary is None or summary.empty:
        print("No cell shift summary data to plot.")
        return None, None

    if shift_hatches is None:
        shift_hatches = {
            "lower_shift": "///",
            "upper_shift": "\\\\\\",
            "dermis_epi_shift": "///",
            "wound_epi_shift": "xxx",
        }

    present_subtypes = [
        st for st in subtype_order
        if st in set(summary["SubtypeClean"].astype(str))
    ]

    if not present_subtypes:
        print("No matching cell shift subtypes to plot.")
        return None, None

    x = np.arange(len(present_subtypes))
    width = 0.78 / max(len(shift_order), 1)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    rng = np.random.default_rng(0)

    alpha_start, alpha_end = alpha_range
    shift_alphas = np.linspace(alpha_start, alpha_end, len(shift_order))

    old_hatch_lw = plt.rcParams.get("hatch.linewidth", 1.0)
    plt.rcParams["hatch.linewidth"] = hatch_linewidth

    try:
        for j, shift_type in enumerate(shift_order):
            offset = (j - (len(shift_order) - 1) / 2) * width

            means = []
            errs = []

            for subtype in present_subtypes:
                row = summary[
                    (summary["SubtypeClean"].astype(str) == subtype)
                    & (summary["ShiftType"] == shift_type)
                ]

                if row.empty:
                    means.append(np.nan)
                    errs.append(np.nan)
                else:
                    means.append(float(row["mean"].iloc[0]))
                    errs.append(float(row["std"].iloc[0]) if pd.notna(row["std"].iloc[0]) else np.nan)

            means = np.asarray(means, dtype=float)
            errs = np.asarray(errs, dtype=float)
            bar_cols = [colours.get(st, "grey") for st in present_subtypes]

            bars = ax.bar(
                x + offset,
                means,
                width=width,
                color=bar_cols,
                alpha=shift_alphas[j],
                edgecolor="black",
                linewidth=0.6,
                zorder=2,
            )

            if hatches:
                for bar, subtype in zip(bars, present_subtypes):
                    hatch_colour = "black" if subtype == "control" else "lightgrey"

                    ax.bar(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        width=bar.get_width(),
                        color="none",
                        edgecolor=hatch_colour,
                        linewidth=0.0,
                        hatch=shift_hatches.get(shift_type, ""),
                        alpha=1.0,
                        zorder=3,
                    )

            ax.errorbar(
                x + offset,
                means,
                yerr=errs,
                fmt="none",
                ecolor="black",
                capsize=3,
                linewidth=1.1,
                zorder=4,
            )

            if shift_df is not None and not shift_df.empty:
                for i, subtype in enumerate(present_subtypes):
                    vals = shift_df.loc[
                        (shift_df["SubtypeClean"].astype(str) == subtype)
                        & (shift_df["ShiftType"] == shift_type),
                        "PercentShift",
                    ].to_numpy(dtype=float)

                    vals = vals[np.isfinite(vals)]

                    if vals.size == 0:
                        continue

                    xj = x[i] + offset + rng.normal(0, jitter, size=vals.size)

                    # ax.scatter(
                    #     xj,
                    #     vals,
                    #     color="black",
                    #     s=point_size,
                    #     alpha=0.65,
                    #     linewidths=0,
                    #     zorder=5,
                    # )

    finally:
        plt.rcParams["hatch.linewidth"] = old_hatch_lw

    ax.axhline(0, color="black", linewidth=1, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [labels.get(st, st) for st in present_subtypes],
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)

    legend_handles = []
    for j, shift_type in enumerate(shift_order):
        if hatches:
            legend_handles.append(
                Patch(
                    facecolor="white",
                    edgecolor="black",
                    hatch=shift_hatches.get(shift_type, ""),
                    label=shift_labels.get(shift_type, shift_type),
                )
            )
        else:
            legend_handles.append(
                Patch(
                    facecolor="grey",
                    edgecolor="black",
                    alpha=shift_alphas[j],
                    label=shift_labels.get(shift_type, shift_type),
                )
            )

    ax.legend(handles=legend_handles, frameon=False)

    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()

    return fig, ax


def p_to_stars(p):
    """
    Convert p-value to significance stars.
    """
    if pd.isna(p):
        return "ns"
    if p < 0.0001:
        return "****"
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"
    
def _add_sig_bar(ax, x1, x2, y, h, text, lw=1.0):
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        color="black",
        lw=lw,
        clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2,
        y + h,
        text,
        ha="center",
        va="bottom",
        fontsize=10,
    )


def plot_cell_spatial_4region_bar(
    cell_raw,
    value_col=CELL_VALUE_COL,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=None,
    hatches=True,
    colours=NI_COLOURS,
    labels=NI_LABELS,
    region_labels=CELL_REGION_LABELS,
    alpha_range=(0.95, 0.35),
    bar_alpha=0.85,
    errorbar="std",       # "std" or "sem"
    add_ttests=True,
    print_ttests=True,
    ttest_pairs=None,
    sig_bar_pad=0.06,
    sig_bar_height=0.025,
):
    """
    Cell density by subtype with four spatial regions:
        lower dermis
        upper dermis
        lower wound
        upper wound

    T-tests:
        lower dermis vs upper dermis
        lower wound vs upper wound

    Uses Welch independent t-test on raw values.
    """

    subtypes = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)

    d = prepare_cell_raw(cell_raw, value_col=value_col)

    if d.empty:
        print("No cell data to plot.")
        return None, None, pd.DataFrame(), pd.DataFrame()

    d = d[d["SubtypeClean"].astype(str).isin(subtypes)].copy()

    if d.empty:
        print("No matching cell subtypes to plot.")
        return None, None, pd.DataFrame(), pd.DataFrame()

    regions = CELL_REGION_ORDER_4

    summary = summarise_cell_by_region(d, value_col=value_col)

    if summary.empty:
        print("No cell summary data to plot.")
        return None, None, summary, pd.DataFrame()

    present_subtypes = [
        st for st in subtypes
        if st in set(summary["SubtypeClean"].astype(str))
    ]

    if not present_subtypes:
        print("No matching cell subtypes in summary.")
        return None, None, summary, pd.DataFrame()

    if ttest_pairs is None:
        ttest_pairs = [
            ("dermis_sub", "dermis_epi"),
            ("wound_sub", "wound_epi"),
        ]

    region_hatches = {
        "dermis_sub": "///",
        "dermis_epi": "\\\\\\",
        "wound_sub": "xxx",
        "wound_epi": "...",
    }

    x = np.arange(len(present_subtypes))
    width = 0.82 / max(len(regions), 1)
    region_alphas = np.linspace(alpha_range[0], alpha_range[1], len(regions))

    fig, ax = plt.subplots(figsize=(10, 5.8))

    bar_xpos = {}
    y_tops = {}

    for j, region in enumerate(regions):
        offset = (j - (len(regions) - 1) / 2) * width
        xpos = x + offset

        means = []
        errs = []

        for subtype in present_subtypes:
            row = summary[
                (summary["SubtypeClean"].astype(str) == subtype)
                & (summary["Region"].astype(str) == region)
            ]

            if row.empty:
                means.append(np.nan)
                errs.append(np.nan)
            else:
                mean = float(row["mean"].iloc[0])
                std = float(row["std"].iloc[0]) if pd.notna(row["std"].iloc[0]) else np.nan
                n = int(row["n"].iloc[0]) if "n" in row.columns and pd.notna(row["n"].iloc[0]) else 1

                if errorbar == "sem":
                    err = std / np.sqrt(max(n, 1)) if np.isfinite(std) else np.nan
                else:
                    err = std

                means.append(mean)
                errs.append(err)

        means = np.asarray(means, dtype=float)
        errs = np.asarray(errs, dtype=float)

        bar_cols = [colours.get(st, "grey") for st in present_subtypes]

        bars = ax.bar(
            xpos,
            means,
            width=width,
            color=bar_cols,
            alpha=region_alphas[j] * bar_alpha,
            edgecolor="black",
            linewidth=0.6,
            label=region_labels.get(region, region),
            zorder=2,
        )

        if hatches:
            for bar, subtype in zip(bars, present_subtypes):
                hatch_colour = "black" if subtype == "control" else "lightgrey"

                ax.bar(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    width=bar.get_width(),
                    color="none",
                    edgecolor=hatch_colour,
                    linewidth=0.0,
                    hatch=region_hatches.get(region, ""),
                    alpha=1.0,
                    zorder=3,
                )

        ax.errorbar(
            xpos,
            means,
            yerr=errs,
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1.1,
            zorder=4,
        )

        for i, subtype in enumerate(present_subtypes):
            bar_xpos[(subtype, region)] = xpos[i]

            y_top = means[i]
            if np.isfinite(errs[i]):
                y_top += errs[i]

            y_tops[(subtype, region)] = y_top

    # ------------------------------------------------------------------
    # T-tests and significance bars
    # ------------------------------------------------------------------
    ttest_rows = []
    
    if add_ttests:
        finite_tops = [v for v in y_tops.values() if np.isfinite(v)]
        ymax_data = max(finite_tops) if finite_tops else 1.0
    
        if ylim is not None:
            yrange = ylim[1] - ylim[0]
        else:
            yrange = ymax_data if ymax_data > 0 else 1.0
    
        h = sig_bar_height * yrange
        pad = sig_bar_pad * yrange
    
        if print_ttests:
            print("\n[Cell spatial region Welch t-tests]")
            print(f"Value column: {value_col}")
            print("Tests are performed on sample/repeat-level region means, not raw point values.")
    
        # Use sample-level means rather than raw values
        sample_col = "Sample" if "Sample" in d.columns else None
    
        if sample_col is None:
            raise KeyError("Cannot run sample-level t-tests because no 'Sample' column was found.")
    
        cell_sample_means = (
            d.groupby(["SubtypeClean", sample_col, "Region"], observed=True, as_index=False)[value_col]
            .mean()
            .rename(columns={value_col: "SampleMean"})
        )
    
        for subtype in present_subtypes:
            current_y = max(
                [
                    y_tops.get((subtype, region), np.nan)
                    for region in regions
                ]
            )
    
            if not np.isfinite(current_y):
                current_y = ymax_data
    
            current_y += pad
    
            for region_1, region_2 in ttest_pairs:
                vals_1 = cell_sample_means.loc[
                    (cell_sample_means["SubtypeClean"].astype(str) == subtype)
                    & (cell_sample_means["Region"].astype(str) == region_1),
                    "SampleMean",
                ].astype(float).dropna().to_numpy()
    
                vals_2 = cell_sample_means.loc[
                    (cell_sample_means["SubtypeClean"].astype(str) == subtype)
                    & (cell_sample_means["Region"].astype(str) == region_2),
                    "SampleMean",
                ].astype(float).dropna().to_numpy()
    
                if len(vals_1) >= 2 and len(vals_2) >= 2:
                    res = ttest_ind(vals_1, vals_2, equal_var=False, nan_policy="omit")
                    t_stat = float(res.statistic)
                    p_val = float(res.pvalue)
                else:
                    t_stat = np.nan
                    p_val = np.nan
    
                stars = _p_to_stars(p_val)
    
                mean_1 = np.nanmean(vals_1) if len(vals_1) else np.nan
                mean_2 = np.nanmean(vals_2) if len(vals_2) else np.nan
                std_1 = np.nanstd(vals_1, ddof=1) if len(vals_1) > 1 else np.nan
                std_2 = np.nanstd(vals_2, ddof=1) if len(vals_2) > 1 else np.nan
    
                ttest_rows.append({
                    "Subtype": subtype,
                    "Region_1": region_1,
                    "Region_2": region_2,
                    "n_1": len(vals_1),
                    "n_2": len(vals_2),
                    "mean_1": mean_1,
                    "mean_2": mean_2,
                    "std_1": std_1,
                    "std_2": std_2,
                    "t_stat": t_stat,
                    "p_value": p_val,
                    "stars": stars,
                })
    
                if print_ttests:
                    print(
                        f"{labels.get(subtype, subtype)}: "
                        f"{region_labels.get(region_1, region_1)} vs {region_labels.get(region_2, region_2)} | "
                        f"n=({len(vals_1)}, {len(vals_2)}) | "
                        f"means=({mean_1:.4g}, {mean_2:.4g}) | "
                        f"std=({std_1:.4g}, {std_2:.4g}) | "
                        f"p={p_val:.4g} | {stars}"
                    )
    
                if (
                    np.isfinite(p_val)
                    and (subtype, region_1) in bar_xpos
                    and (subtype, region_2) in bar_xpos
                ):
                    _add_sig_bar(
                        ax,
                        bar_xpos[(subtype, region_1)],
                        bar_xpos[(subtype, region_2)],
                        current_y,
                        h,
                        stars,
                    )
    
                    current_y += pad + h
    
    ttest_df = pd.DataFrame(ttest_rows)

    # ------------------------------------------------------------------
    # Axis formatting
    # ------------------------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(
        [labels.get(st, st) for st in present_subtypes],
        rotation=0,
        ha="center",
    )

    ax.set_ylabel("Fibroblasts / mm²")
    # ax.set_title("Fibroblast density by spatial region")
    ax.set_title(" ")
    ax.grid(False)

    legend_handles = []

    for region in regions:
        if hatches:
            legend_handles.append(
                Patch(
                    facecolor="white",
                    edgecolor="black",
                    hatch=region_hatches.get(region, ""),
                    label=region_labels.get(region, region),
                )
            )
        else:
            legend_handles.append(
                Patch(
                    facecolor="grey",
                    edgecolor="black",
                    alpha=0.8,
                    label=region_labels.get(region, region),
                )
            )

    ax.legend(handles=legend_handles, frameon=False)

    if ylim is not None:
        ax.set_ylim(*ylim)
    elif add_ttests and ttest_rows:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.12)

    plt.tight_layout()
    plt.show()

    return fig, ax, summary, ttest_df

def plot_cell_spatial_4region_histograms(
    cell_raw,
    value_col=CELL_VALUE_COL,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    regions=CELL_REGION_ORDER_4,
    colours=NI_COLOURS,
    labels=NI_LABELS,
    region_labels=CELL_REGION_LABELS,
    bins=20,
    density=True,
    histtype="step",
    linewidth=2.0,
    alpha=0.95,
    ncols=3,
    xlim=None,
    ylim=None,
    title="Fibroblast density distributions by spatial region",
):
    """
    Histogram plot of cell density values.

    One panel per subtype.
    Each line = one spatial region:
        dermis_sub
        dermis_epi
        wound_sub
        wound_epi

    Uses shared x and y scales across all panels.
    """

    subtypes = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)

    d = prepare_cell_raw(cell_raw, value_col=value_col)

    if d.empty:
        print("No cell data to plot.")
        return None, None, pd.DataFrame()

    d = d[d["SubtypeClean"].astype(str).isin(subtypes)].copy()
    d = d[d["Region"].astype(str).isin(regions)].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col, "SubtypeClean", "Region"]).copy()

    if d.empty:
        print("No matching cell data after subtype/region filtering.")
        return None, None, pd.DataFrame()

    present_subtypes = [
        st for st in subtypes
        if st in set(d["SubtypeClean"].astype(str))
    ]

    if not present_subtypes:
        print("No matching cell subtypes to plot.")
        return None, None, pd.DataFrame()

    linestyles = {
        "dermis_sub": "-",
        "dermis_epi": "--",
        "wound_sub": "-.",
        "wound_epi": ":",
        "dermis": "-",
        "wound": "--",
    }

    if xlim is None:
        vals = d[value_col].to_numpy(float)
        vals = vals[np.isfinite(vals)]

        xmin = np.nanmin(vals)
        xmax = np.nanmax(vals)

        pad = 0.04 * (xmax - xmin) if xmax > xmin else 1.0
        xlim = (xmin - pad, xmax + pad)

    bin_edges = np.linspace(xlim[0], xlim[1], bins + 1)

    # Work out common y-limit before plotting
    ymax = 0.0

    for subtype in present_subtypes:
        for region in regions:
            vals = d.loc[
                (d["SubtypeClean"].astype(str) == subtype)
                & (d["Region"].astype(str) == region),
                value_col,
            ].to_numpy(float)

            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                continue

            counts, _ = np.histogram(vals, bins=bin_edges, density=density)

            if counts.size and np.isfinite(counts).any():
                ymax = max(ymax, float(np.nanmax(counts)))

    if ylim is None:
        ylim = (0, ymax * 1.12 if ymax > 0 else 1)

    n_panels = len(present_subtypes)
    ncols = min(ncols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.0 * ncols, 4.0 * nrows),
        sharex=True,
        sharey=True,
    )

    axes = np.atleast_1d(axes).ravel()

    hist_rows = []

    for ax, subtype in zip(axes, present_subtypes):
        d_sub = d[d["SubtypeClean"].astype(str) == subtype].copy()
        subtype_colour = colours.get(subtype, "grey")

        for region in regions:
            vals = d_sub.loc[
                d_sub["Region"].astype(str) == region,
                value_col,
            ].to_numpy(float)

            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                continue

            counts, edges = np.histogram(vals, bins=bin_edges, density=density)
            centres = 0.5 * (edges[:-1] + edges[1:])

            ax.hist(
                vals,
                bins=bin_edges,
                density=density,
                histtype=histtype,
                color=subtype_colour,
                linestyle=linestyles.get(region, "-"),
                linewidth=linewidth,
                alpha=alpha,
                label=region_labels.get(region, region),
            )

            hist_rows.append(pd.DataFrame({
                "SubtypeClean": subtype,
                "Region": region,
                "BinCentre": centres,
                "BinLeft": edges[:-1],
                "BinRight": edges[1:],
                "CountOrDensity": counts,
                "N": vals.size,
            }))

        ax.set_title(labels.get(subtype, subtype))
        ax.set_xlabel("Fibroblasts / mm²")
        ax.set_ylabel("Density" if density else "Count")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(False)

    for ax in axes[n_panels:]:
        ax.axis("off")

    handles = [
        plt.Line2D(
            [0],
            [0],
            color="black",
            linestyle=linestyles.get(region, "-"),
            linewidth=linewidth,
            label=region_labels.get(region, region),
        )
        for region in regions
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(handles), 4),
        frameon=False,
    )

    fig.suptitle(title, y=0.995)

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.show()

    hist_df = pd.concat(hist_rows, ignore_index=True) if hist_rows else pd.DataFrame()

    return fig, axes, hist_df

def prepare_cell_dermis_wound_pooled(cell_raw, value_col=CELL_VALUE_COL):
    """
    Pool spatial cell regions into:
        dermis = dermis_sub + dermis_epi
        wound = wound_sub + wound_epi

    Includes derived control dermis from all wound dermis points.
    """

    d = prepare_cell_raw(cell_raw, value_col=value_col)

    if d.empty:
        return d

    region_map = {
        "dermis_sub": "dermis",
        "dermis_epi": "dermis",
        "wound_sub": "wound",
        "wound_epi": "wound",
    }

    d = d[d["Region"].isin(region_map)].copy()
    d["Region"] = d["Region"].map(region_map)

    return d

def plot_cell_dermis_vs_wound_bar(
    cell_raw,
    value_col=CELL_VALUE_COL,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=None,
    hatches=True,
):
    subtypes = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)

    d = prepare_cell_dermis_wound_pooled(cell_raw, value_col=value_col)
    d = d[d["SubtypeClean"].astype(str).isin(subtypes)].copy()

    summary = summarise_cell_by_region(d, value_col=value_col)

    return _plot_grouped_cell_bars(
        summary=summary,
        points=d,
        regions=["dermis", "wound"],
        # title="Fibroblast density in dermis and wound bed",
        title="",
        ylabel="Fibroblasts / mm²",
        subtype_order=subtypes,
        ylim=ylim,
        hatches=hatches,
    )

def calculate_cell_wound_vs_dermis_shift(cell_raw, value_col=CELL_VALUE_COL):
    """
    Figure 3 data:
    Calculate % shift from dermis to wound within each subtype.

        lower shift = lower wound vs lower dermis
        upper shift = upper wound vs upper dermis

    Uses replicate-level pairing by CellRepeat where possible.
    """

    d = prepare_cell_raw(cell_raw, value_col=value_col)

    # No control wound, so exclude control from this shift.
    d = d[d["SubtypeClean"].astype(str) != "control"].copy()

    rows = []

    shift_defs = {
        "lower_shift": ("dermis_sub", "wound_sub"),
        "upper_shift": ("dermis_epi", "wound_epi"),
    }

    for subtype, d_sub in d.groupby("SubtypeClean", observed=True):
        for shift_name, (baseline_region, target_region) in shift_defs.items():
            base = d_sub[d_sub["Region"] == baseline_region].copy()
            targ = d_sub[d_sub["Region"] == target_region].copy()

            base_vals = base[[value_col]].reset_index(drop=True)
            targ_vals = targ[[value_col]].reset_index(drop=True)

            n = min(len(base_vals), len(targ_vals))

            if n == 0:
                continue

            for i in range(n):
                baseline = float(base_vals[value_col].iloc[i])
                target = float(targ_vals[value_col].iloc[i])

                if not np.isfinite(baseline) or baseline == 0 or not np.isfinite(target):
                    continue

                shift = ((target - baseline) / baseline) * 100

                rows.append({
                    "SubtypeClean": str(subtype),
                    "ShiftType": shift_name,
                    "BaselineRegion": baseline_region,
                    "TargetRegion": target_region,
                    "PairIndex": i + 1,
                    "BaselineValue": baseline,
                    "TargetValue": target,
                    "PercentShift": shift,
                })

    shift_df = pd.DataFrame(rows)

    if shift_df.empty:
        return shift_df, pd.DataFrame()

    summary = (
        shift_df
        .groupby(["SubtypeClean", "ShiftType"], as_index=False)["PercentShift"]
        .agg(mean="mean", std="std", n="count")
    )

    return shift_df, summary

def plot_cell_wound_vs_dermis_shift(
    cell_raw,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=None,
    hatches=True,
):
    subtypes = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)
    subtypes = [st for st in subtypes if st != "control"]

    shift_df, summary = calculate_cell_wound_vs_dermis_shift(cell_raw)

    shift_df = shift_df[shift_df["SubtypeClean"].astype(str).isin(subtypes)].copy()
    summary = summary[summary["SubtypeClean"].astype(str).isin(subtypes)].copy()

    return _plot_cell_shift_bars(
        shift_df=shift_df,
        summary=summary,
        shift_order=["lower_shift", "upper_shift"],
        title="Fibroblast density % shift from dermis to wound",
        ylabel="% shift from dermis",
        subtype_order=subtypes,
        ylim=ylim,
        hatches=hatches,
    )

def calculate_cell_upper_vs_lower_shift(cell_raw, value_col=CELL_VALUE_COL):
    """
    Figure 4 data:
    Calculate % shift within region:

        dermis shift = upper dermis vs lower dermis
        wound shift = upper wound vs lower wound
    """

    d = prepare_cell_raw(cell_raw, value_col=value_col)

    rows = []

    shift_defs = {
        "dermis_epi_shift": ("dermis_sub", "dermis_epi"),
        "wound_epi_shift": ("wound_sub", "wound_epi"),
    }

    for subtype, d_sub in d.groupby("SubtypeClean", observed=True):
        for shift_name, (baseline_region, target_region) in shift_defs.items():
            base = d_sub[d_sub["Region"] == baseline_region].copy()
            targ = d_sub[d_sub["Region"] == target_region].copy()

            base_vals = base[[value_col]].reset_index(drop=True)
            targ_vals = targ[[value_col]].reset_index(drop=True)

            n = min(len(base_vals), len(targ_vals))

            if n == 0:
                continue

            for i in range(n):
                baseline = float(base_vals[value_col].iloc[i])
                target = float(targ_vals[value_col].iloc[i])

                if not np.isfinite(baseline) or baseline == 0 or not np.isfinite(target):
                    continue

                shift = ((target - baseline) / baseline) * 100

                rows.append({
                    "SubtypeClean": str(subtype),
                    "ShiftType": shift_name,
                    "BaselineRegion": baseline_region,
                    "TargetRegion": target_region,
                    "PairIndex": i + 1,
                    "BaselineValue": baseline,
                    "TargetValue": target,
                    "PercentShift": shift,
                })

    shift_df = pd.DataFrame(rows)

    if shift_df.empty:
        return shift_df, pd.DataFrame()

    summary = (
        shift_df
        .groupby(["SubtypeClean", "ShiftType"], as_index=False)["PercentShift"]
        .agg(mean="mean", std="std", n="count")
    )

    return shift_df, summary

def plot_cell_upper_vs_lower_shift(
    cell_raw,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=None,
    hatches=True,
):
    subtypes = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)

    shift_df, summary = calculate_cell_upper_vs_lower_shift(cell_raw)

    shift_df = shift_df[shift_df["SubtypeClean"].astype(str).isin(subtypes)].copy()
    summary = summary[summary["SubtypeClean"].astype(str).isin(subtypes)].copy()

    return _plot_cell_shift_bars(
        shift_df=shift_df,
        summary=summary,
        shift_order=["dermis_epi_shift", "wound_epi_shift"],
        title="Fibroblast density % shift from lower to upper regions",
        ylabel="% shift from lower region",
        subtype_order=subtypes,
        ylim=ylim,
        hatches=hatches,
    )

# =============================================================================
# saxs plotting
def resolve_subtypes_to_plot(subtype_order, subtypes_to_plot=None):
    """
    Resolve which subtypes should be plotted, preserving subtype_order.
    """
    if subtypes_to_plot is None:
        return [str(st).strip() for st in subtype_order]

    wanted = [str(st).strip() for st in subtypes_to_plot]
    wanted_set = set(wanted)

    ordered = [
        str(st).strip()
        for st in subtype_order
        if str(st).strip() in wanted_set
    ]

    ordered += [
        st for st in wanted
        if st not in ordered
    ]

    return ordered

def trim_by_std(
    df,
    *,
    value_col,
    trim_std_devs=None,
    group_cols=("subtype", "region"),
    print_cols=False,
):
    """
    Remove values outside mean ± N*SD within each group.
    """

    if df is None or df.empty or value_col not in df.columns or trim_std_devs is None:
        return df.copy()

    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    keep = pd.Series(True, index=d.index)
    removed_total = 0

    for _, idx in d.groupby(list(group_cols), dropna=False).groups.items():
        vals = pd.to_numeric(d.loc[idx, value_col], errors="coerce").to_numpy(float)

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
        removed_total += int((~sub_keep & finite).sum())

    out = d.loc[keep].copy()

    if print_cols:
        print(
            f"[trim_by_std] value_col={value_col} "
            f"trim_std_devs={trim_std_devs} "
            f"rows_before={len(d):,} rows_after={len(out):,} "
            f"removed={removed_total:,}"
        )

    return out

def prepare_saxs_sample_data(
    saxs_sample,
    parameter="curvearea",
    value_col=SAXS_VALUE_COL,
    normalise=False,
    normalised_col="SAXS_normalised_value",
):
    """
    Prepare per-sample SAXS data for plotting.

    If normalise=True:
        uses the raw value_col, applies subtype/sample affine normalisation,
        and stores the plotting value in normalised_col.
    """

    if saxs_sample is None or saxs_sample.empty:
        return pd.DataFrame(), value_col

    d = saxs_sample.copy()
    d = add_clean_subtype(d)

    if "parameter" not in d.columns:
        raise KeyError("saxs_sample must contain a 'parameter' column.")

    if "region" not in d.columns:
        raise KeyError("saxs_sample must contain a 'region' column.")

    if value_col not in d.columns:
        raise KeyError(f"saxs_sample must contain '{value_col}' column.")

    d["parameter"] = d["parameter"].astype(str).str.strip()
    d["region"] = d["region"].astype(str).str.strip()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    d = d[
        (d["parameter"] == parameter)
        & d[value_col].notna()
        & d["SubtypeClean"].notna()
    ].copy()

    if d.empty:
        return d, value_col

    if normalise:
        d = normalise_saxs_by_subtype_target_pull_sample_endpoints(
            d,
            value_col=value_col,
            out_col=normalised_col,
            clip=True,
            to_unit_interval=True,
            b_from="five_regions",
            T_from="mean_dermis_max",
        )
        return d, normalised_col

    return d.reset_index(drop=True), value_col

def prepare_saxs_point_data_for_plot(
    saxs_points,
    parameter="curvearea",
    raw_value_col="value",
    normalise=True,
    normalised_col="SAXS_normalised_value",
    curvearea_thresh=None,
    saxs_thresh=None,
    trim_std_devs=6,
):
    """
    Prepare SAXS point-level data for plotting.

    This matches the original SAXS workflow more closely:
        point data
        optional gates
        trim outliers
        normalise point values
        then later summarise to sample/region means
    """

    if saxs_points is None or saxs_points.empty:
        return pd.DataFrame(), raw_value_col

    d = saxs_points.copy()
    d = add_clean_subtype(d)

    required = ["subtype", "region", "Filenumber", "parameter", raw_value_col]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise KeyError(f"saxs_points missing columns: {missing}")

    d["parameter"] = d["parameter"].astype(str).str.strip()
    d["region"] = d["region"].astype(str).str.strip()
    d["subtype"] = d["subtype"].astype(str).str.strip()
    d["Filenumber"] = d["Filenumber"].astype(str).str.strip()
    d[raw_value_col] = pd.to_numeric(d[raw_value_col], errors="coerce")

    d = d[
        (d["parameter"] == parameter)
        & d[raw_value_col].notna()
        & d["region"].isin(SAXS_DERMIS_REGIONS + SAXS_WOUND_REGIONS)
    ].copy()

    if d.empty:
        return d, raw_value_col

    # Optional gates if those columns exist in exported point-level data.
    # Your current exported long table may only have one parameter/value at a time,
    # so these gates may not be possible here unless you export the raw columns too.
    if curvearea_thresh is not None and "collagen_third_norm_0_1" in d.columns:
        v = pd.to_numeric(d["collagen_third_norm_0_1"], errors="coerce")
        d = d[v >= curvearea_thresh].copy()

    if saxs_thresh is not None and "total_SAXS_norm_0_1" in d.columns:
        v = pd.to_numeric(d["total_SAXS_norm_0_1"], errors="coerce")
        d = d[v >= saxs_thresh].copy()

    if d.empty:
        return d, raw_value_col

    d = trim_by_std(
        d,
        value_col=raw_value_col,
        trim_std_devs=trim_std_devs,
        group_cols=("subtype", "region"),
        print_cols=False,
    )

    if normalise:
        d = normalise_saxs_by_subtype_target_pull_sample_endpoints(
            d,
            value_col=raw_value_col,
            out_col=normalised_col,
            clip=True,
            to_unit_interval=True,
            b_from="five_regions",
            T_from="mean_dermis_max",
        )
        return d, normalised_col

    return d, raw_value_col

def pool_saxs_dermis_wound_regions(saxs_df):
    """
    Pool split SAXS regions into:
        dermis = dermis_sub + dermis_epi
        wound = wound_sub + wound_epi

    If the data is already unsplit and has dermis/wound, it keeps those.
    """

    if saxs_df is None or saxs_df.empty:
        return pd.DataFrame()

    d = saxs_df.copy()

    region_map = {
        "dermis_sub": "dermis",
        "dermis_epi": "dermis",
        "wound_sub": "wound",
        "wound_epi": "wound",
        "dermis": "dermis",
        "wound": "wound",
    }

    d = d[d["region"].isin(region_map)].copy()
    d["RegionPooled"] = d["region"].map(region_map)

    return d

def summarise_saxs_for_bars(
    saxs_df,
    region_col="region",
    value_col=SAXS_VALUE_COL,
):
    """
    Summarise SAXS sample-level data for bar charts.

    Bars are mean across samples.
    Error bars are SD across samples.
    """

    if saxs_df is None or saxs_df.empty:
        return pd.DataFrame()

    d = saxs_df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    summary = (
        d.groupby(["SubtypeClean", region_col], observed=True)[value_col]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .rename(columns={region_col: "Region"})
    )

    return summary

def summarise_saxs_points_to_sample_bars(
    df,
    value_col,
    region_col="region",
):
    """
    Convert point-level SAXS data to:
        sample means
        subtype/region summary

    This matches pooled=False:
        first average within each sample/region,
        then average those sample means across samples.
    """

    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    sample_means = (
        d.groupby(
            ["experiment", "subtype", "SubtypeClean", "Filenumber", region_col],
            observed=True,
            as_index=False,
        )[value_col]
        .mean()
        .rename(columns={value_col: "SampleMean", region_col: "Region"})
    )

    summary = (
        sample_means
        .groupby(["SubtypeClean", "Region"], observed=True)["SampleMean"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )

    return sample_means, summary

def _plot_saxs_grouped_bars(
    summary,
    points,
    regions,
    title,
    ylabel,
    region_col_points="region",
    value_col=SAXS_VALUE_COL,
    subtype_order=SUBTYPE_ORDER,
    colours=NI_COLOURS,
    labels=NI_LABELS,
    region_labels=SAXS_REGION_LABELS,
    ylim=None,
    scatter=True,
    jitter=0.045,
    point_size=24,
    alpha_range=(0.95, 0.30),
    hatches=True,
    region_hatches=None,
    hatch_linewidth=0.7,
    show=True,
):
    """
    Generic grouped SAXS bar chart.

    x = subtype
    grouped bars = region
    bars = mean across samples
    error bars = SD across samples
    points = individual sample means

    hatches=True:
        Adds region-specific hatch patterns and shows them in the legend.

    hatches=False:
        Uses colour/alpha only.
    """

    if summary is None or summary.empty:
        print("No SAXS summary data to plot.")
        return None, None

    if region_hatches is None:
        region_hatches = {
            "dermis_sub": "///",
            "dermis_epi": "\\\\\\",
            "wound_sub": "...",
            "wound_epi": "xxx",
            "dermis": "///",
            "wound": "xxx",
        }

    present_subtypes = [
        st for st in subtype_order
        if st in set(summary["SubtypeClean"].astype(str))
    ]

    if not present_subtypes:
        print("No matching SAXS subtypes to plot.")
        return None, None

    x = np.arange(len(present_subtypes))
    width = 0.82 / max(len(regions), 1)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    rng = np.random.default_rng(0)

    region_alphas = np.linspace(alpha_range[0], alpha_range[1], len(regions))

    old_hatch_lw = plt.rcParams.get("hatch.linewidth", 1.0)
    plt.rcParams["hatch.linewidth"] = hatch_linewidth

    try:
        for j, region in enumerate(regions):
            offset = (j - (len(regions) - 1) / 2) * width

            means = []
            errs = []

            for subtype in present_subtypes:
                row = summary[
                    (summary["SubtypeClean"].astype(str) == subtype)
                    & (summary["Region"].astype(str) == region)
                ]

                if row.empty:
                    means.append(np.nan)
                    errs.append(np.nan)
                else:
                    means.append(float(row["mean"].iloc[0]))
                    errs.append(
                        float(row["std"].iloc[0])
                        if pd.notna(row["std"].iloc[0])
                        else np.nan
                    )

            means = np.asarray(means, dtype=float)
            errs = np.asarray(errs, dtype=float)
            bar_cols = [colours.get(st, "grey") for st in present_subtypes]

            bars = ax.bar(
                x + offset,
                means,
                width=width,
                color=bar_cols,
                alpha=region_alphas[j],
                edgecolor="black",
                linewidth=0.6,
                zorder=2,
            )

            if hatches:
                for bar, subtype in zip(bars, present_subtypes):
                    hatch_colour = "black" if subtype == "control" else "lightgrey"

                    ax.bar(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        width=bar.get_width(),
                        color="none",
                        edgecolor=hatch_colour,
                        linewidth=0.0,
                        hatch=region_hatches.get(region, ""),
                        alpha=1.0,
                        zorder=3,
                    )

            ax.errorbar(
                x + offset,
                means,
                yerr=errs,
                fmt="none",
                ecolor="black",
                capsize=3,
                linewidth=1.1,
                zorder=4,
            )

            if scatter and points is not None and not points.empty:
                for i, subtype in enumerate(present_subtypes):
                    vals = points.loc[
                        (points["SubtypeClean"].astype(str) == subtype)
                        & (points[region_col_points].astype(str) == region),
                        value_col,
                    ].to_numpy(dtype=float)

                    vals = vals[np.isfinite(vals)]

                    if vals.size == 0:
                        continue

                    xj = x[i] + offset + rng.normal(0, jitter, size=vals.size)

                    # ax.scatter(
                    #     xj,
                    #     vals,
                    #     color="black",
                    #     s=point_size,
                    #     alpha=0.65,
                    #     linewidths=0,
                    #     zorder=5,
                    # )

    finally:
        plt.rcParams["hatch.linewidth"] = old_hatch_lw

    ax.set_xticks(x)
    ax.set_xticklabels(
        [labels.get(st, st) for st in present_subtypes],
        rotation=0,
        ha="right",
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)

    legend_handles = []
    for j, region in enumerate(regions):
        if hatches:
            legend_handles.append(
                Patch(
                    facecolor="white",
                    edgecolor="black",
                    hatch=region_hatches.get(region, ""),
                    label=region_labels.get(region, region),
                )
            )
        else:
            legend_handles.append(
                Patch(
                    facecolor="grey",
                    edgecolor="black",
                    alpha=region_alphas[j],
                    label=region_labels.get(region, region),
                )
            )

    # ax.legend(handles=legend_handles, frameon=False)

    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    if show:
        plt.show()
    
    return fig, ax

def p_to_stars(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

def paired_region_t_tests(
    sample_means,
    *,
    value_col,
    subtype_order=SUBTYPE_ORDER,
    region_a="dermis",
    region_b="wound",
):
    """
    Paired t-test between two regions within each subtype.

    Uses sample-level means, paired by Filenumber.
    """

    if sample_means is None or sample_means.empty:
        return pd.DataFrame()

    d = sample_means.copy()

    required = ["SubtypeClean", "Filenumber", "Region", value_col]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise KeyError(f"Missing required paired-test columns: {missing}")

    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    rows = []

    for subtype in subtype_order:
        ds = d[d["SubtypeClean"].astype(str) == str(subtype)].copy()

        if ds.empty:
            continue

        wide = (
            ds.pivot_table(
                index="Filenumber",
                columns="Region",
                values=value_col,
                aggfunc="mean",
            )
            .reset_index()
        )

        if region_a not in wide.columns or region_b not in wide.columns:
            rows.append({
                "SubtypeClean": subtype,
                "Label": NI_LABELS.get(subtype, subtype),
                "RegionA": region_a,
                "RegionB": region_b,
                "n_pairs": 0,
                "MeanA": np.nan,
                "MeanB": np.nan,
                "MeanDifference_B_minus_A": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "stars": "n/a",
            })
            continue

        paired = wide[[region_a, region_b]].apply(pd.to_numeric, errors="coerce").dropna()

        n = len(paired)

        if n < 2:
            p = np.nan
            t_stat = np.nan
        else:
            from scipy.stats import ttest_rel
            t_stat, p = ttest_rel(paired[region_b], paired[region_a], nan_policy="omit")

        mean_a = paired[region_a].mean() if n else np.nan
        mean_b = paired[region_b].mean() if n else np.nan

        rows.append({
            "SubtypeClean": subtype,
            "Label": NI_LABELS.get(subtype, subtype),
            "RegionA": region_a,
            "RegionB": region_b,
            "n_pairs": n,
            "MeanA": mean_a,
            "MeanB": mean_b,
            "MeanDifference_B_minus_A": mean_b - mean_a if n else np.nan,
            "t_stat": t_stat,
            "p_value": p,
            "stars": p_to_stars(p),
        })

    return pd.DataFrame(rows)

def add_paired_region_annotations(
    ax,
    test_df,
    summary,
    *,
    subtype_order=SUBTYPE_ORDER,
    regions=("dermis", "wound"),
    region_a="dermis",
    region_b="wound",
    bar_group_width=0.82,
    y_pad_frac=0.05,
    bracket_height_frac=0.035,
    text_pad_frac=0.012,
):
    """
    Add paired-test brackets and stars above grouped bars.

    Assumes grouped-bar layout:
        x = subtype
        grouped bars = regions
    """

    if test_df is None or test_df.empty or summary is None or summary.empty:
        return ax

    present_subtypes = list(subtype_order)

    x_base = np.arange(len(present_subtypes))
    width = bar_group_width / max(len(regions), 1)

    region_offsets = {
        region: (j - (len(regions) - 1) / 2) * width
        for j, region in enumerate(regions)
    }

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin if ymax > ymin else 1.0

    y_pad = yrange * y_pad_frac
    bracket_h = yrange * bracket_height_frac
    text_pad = yrange * text_pad_frac

    max_y = ymax

    for i, subtype in enumerate(present_subtypes):
        stat = test_df[test_df["SubtypeClean"].astype(str) == str(subtype)]

        if stat.empty:
            continue

        stars = stat["stars"].iloc[0]

        if stars in {"n/a", "", None}:
            continue

        s = summary[summary["SubtypeClean"].astype(str) == str(subtype)].copy()

        vals = []

        for region in [region_a, region_b]:
            row = s[s["Region"].astype(str) == region]

            if row.empty:
                continue

            mean = float(row["mean"].iloc[0])
            std = float(row["std"].iloc[0]) if pd.notna(row["std"].iloc[0]) else 0.0

            if np.isfinite(mean):
                vals.append(mean + std)

        if not vals:
            continue

        x1 = x_base[i] + region_offsets[region_a]
        x2 = x_base[i] + region_offsets[region_b]

        y = max(vals) + y_pad

        ax.plot(
            [x1, x1, x2, x2],
            [y, y + bracket_h, y + bracket_h, y],
            color="black",
            linewidth=1.1,
            clip_on=False,
            zorder=10,
        )

        ax.text(
            (x1 + x2) / 2,
            y + bracket_h + text_pad,
            stars,
            ha="center",
            va="bottom",
            fontsize=11,
            color="black",
            clip_on=False,
            zorder=11,
        )

        max_y = max(max_y, y + bracket_h + text_pad * 4)

    ax.set_ylim(ymin, max_y + y_pad)

    return ax

def plot_saxs_total_collagen_dermis_vs_wound(
    saxs_points,
    parameter="curvearea",
    normalise=True,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=None,
    trim_std_devs=6,
    hatches=True,
    paired_test=True,
    print_stats=True,
):
    """
    SAXS total collagen intensity in dermis vs wound.

    Uses point-level data, normalises first, then computes sample-level means.

    Statistical testing is performed on sample-level means.

    Dynamic statistics:
        1. If both dermis and wound are present:
               paired dermis vs wound test within each subtype

               normal paired differences     -> paired t-test
               non-normal paired differences -> Wilcoxon signed-rank test

        2. If only one region is present:
               compare between subtypes within that region

               two groups:
                   normal groups     -> independent t-test
                   non-normal groups -> Mann-Whitney U test

               more than two groups:
                   all normal     -> one-way ANOVA
                   any non-normal -> Kruskal-Wallis test
    """

    subtype_order = resolve_subtypes_to_plot(
        subtype_order,
        subtypes_to_plot,
    )

    # ------------------------------------------------------------
    # Prepare SAXS point-level data
    # ------------------------------------------------------------
    d, plot_value_col = prepare_saxs_point_data_for_plot(
        saxs_points=saxs_points,
        parameter=parameter,
        normalise=normalise,
        trim_std_devs=trim_std_devs,
    )

    if d.empty:
        print(f"No SAXS point data available for parameter={parameter}")
        return None, None, pd.DataFrame()

    # ------------------------------------------------------------
    # Pool regions into dermis / wound
    # ------------------------------------------------------------
    d = pool_saxs_dermis_wound_regions(d)
    d = d.rename(
        columns={
            "RegionPooled": "RegionForPlot",
        }
    )

    # ------------------------------------------------------------
    # Convert point-level values into sample-level means
    # ------------------------------------------------------------
    sample_means, summary = summarise_saxs_points_to_sample_bars(
        d,
        value_col=plot_value_col,
        region_col="RegionForPlot",
    )

    bar_summary_print = print_saxs_bar_summary(
        summary,
        title="SAXS total collagen intensity bar means",
    )

    ylabel = (
        "Normalised total collagen intensity"
        if normalise
        else SAXS_PARAMETER_LABELS.get(parameter, parameter)
    )

    # ------------------------------------------------------------
    # Plot grouped bars
    # ------------------------------------------------------------
    fig, ax = _plot_saxs_grouped_bars(
        summary=summary,
        points=sample_means.rename(
            columns={
                "SampleMean": plot_value_col,
            }
        ),
        regions=["dermis", "wound"],
        title="SAXS total collagen intensity in dermis and wound",
        ylabel=ylabel,
        region_col_points="Region",
        value_col=plot_value_col,
        subtype_order=subtype_order,
        ylim=ylim,
        hatches=hatches,
        show=False,
    )

    # ------------------------------------------------------------
    # Automatic statistics
    # ------------------------------------------------------------
    stats_df = pd.DataFrame()

    if paired_test and fig is not None and ax is not None:

        if "Region" not in sample_means.columns:

            print("\n[SAXS total collagen stats] No Region column found in sample_means.")
            print("Available columns:")
            print(sample_means.columns.tolist())

        else:

            regions_present = set(
                sample_means["Region"]
                .astype(str)
                .str.lower()
                .str.strip()
                .unique()
            )

            # ------------------------------------------------------------
            # Case 1: paired dermis vs wound within each subtype
            # ------------------------------------------------------------
            if {"dermis", "wound"}.issubset(regions_present):

                stats_df = auto_saxs_paired_region_tests(
                    sample_means=sample_means,
                    value_col="SampleMean",
                    subtype_order=subtype_order,
                    region_col="Region",
                    sample_col=None,  # auto-detects Filenumber / SampleKey / etc.
                    region_a="dermis",
                    region_b="wound",
                )

                if not stats_df.empty:

                    add_paired_region_annotations(
                        ax,
                        stats_df,
                        summary,
                        subtype_order=subtype_order,
                        regions=("dermis", "wound"),
                        region_a="dermis",
                        region_b="wound",
                    )

                if print_stats and not stats_df.empty:

                    stats_print = stats_df.copy()

                    for c in [
                        "MeanA",
                        "MeanB",
                        "MeanDifference_B_minus_A",
                        "Normality_p",
                        "t_stat",
                        "statistic",
                        "p_value",
                    ]:
                        if c in stats_print.columns:
                            stats_print[c] = pd.to_numeric(
                                stats_print[c],
                                errors="coerce",
                            )

                    paired_cols = [
                        c for c in [
                            "Label",
                            "RegionA",
                            "RegionB",
                            "n_pairs",
                            "MeanA",
                            "MeanB",
                            "MeanDifference_B_minus_A",
                            "NormalityTest",
                            "Normality_p",
                            "NormalityInterpretation",
                            "SelectedTest",
                            "statistic",
                            "p_value",
                            "stars",
                        ]
                        if c in stats_print.columns
                    ]

                    print("\n[SAXS automatic paired tests: total collagen dermis vs wound]")
                    print(
                        stats_print[paired_cols]
                        .to_string(index=False)
                    )

            # ------------------------------------------------------------
            # Case 2: only one region present, compare between subtypes
            # ------------------------------------------------------------
            else:

                stats_df = auto_saxs_between_subtype_tests(
                    sample_means=sample_means,
                    value_col="SampleMean",
                    subtype_order=subtype_order,
                    region_col="Region",
                    sample_col=None,  # auto-detects Filenumber / SampleKey / etc.
                )

                if print_stats and not stats_df.empty:

                    stats_print = stats_df.copy()

                    for c in [
                        "statistic",
                        "p_value",
                    ]:
                        if c in stats_print.columns:
                            stats_print[c] = pd.to_numeric(
                                stats_print[c],
                                errors="coerce",
                            )

                    between_cols = [
                        c for c in [
                            "Region",
                            "Comparison",
                            "n_groups",
                            "GroupNs",
                            "NormalityTest",
                            "Normality_p_values",
                            "NormalityInterpretation",
                            "SelectedTest",
                            "statistic",
                            "p_value",
                            "stars",
                        ]
                        if c in stats_print.columns
                    ]

                    print("\n[SAXS automatic between-subtype tests: total collagen]")
                    print(
                        stats_print[between_cols]
                        .to_string(index=False)
                    )

    plt.tight_layout()
    plt.show()

    return fig, ax, stats_df

def print_saxs_bar_summary(summary, title="SAXS bar summary"):
    """
    Print mean, std, sem, n for each SAXS bar.
    Works for dermis/wound, 4-region, or any grouped SAXS bar summary.
    """

    if summary is None or summary.empty:
        print(f"\n{title}: no summary data.")
        return pd.DataFrame()

    s = summary.copy()

    print_cols = []

    for c in ["SubtypeClean", "Label", "Region", "RegionForPlot", "RegionClean"]:
        if c in s.columns:
            print_cols.append(c)

    for c in ["mean", "std", "sem", "n", "error"]:
        if c in s.columns:
            print_cols.append(c)

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(s[print_cols].to_string(index=False))

    return s[print_cols]

def plot_saxs_total_collagen_dermis_vs_wound_pull_norm(
    saxs_points,
    parameter="curvearea",
    raw_value_col="value",
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    colours=NI_COLOURS,
    labels=NI_LABELS,
    ylim=None,
    trim_std_devs=6,
    curvearea_thresh=0.1,
    saxs_thresh=0.0,
    param_thresh=0,
    clip=True,
    hatches=True,
    stats_test="welch",   # "welch", "mannwhitney", or "paired"
    add_stats=True,
    print_stats=True,
    title="",
    ylabel="Normalised total collagen intensity",
):
    """
    SAXS total collagen intensity in dermis vs wound using the previous
    pull-normalisation method.

    Workflow:
        1. Filter point-level SAXS data for parameter.
        2. Apply thresholds/gates.
        3. Trim outliers.
        4. Pull-normalise within subtype/sample using dermis+wound regions.
        5. Pool dermis_sub + dermis_epi -> dermis;
           pool wound_sub + wound_epi -> wound.
        6. Average point values to sample means.
        7. Plot subtype mean ± SD across sample means.
        8. Optionally test dermis vs wound within each subtype.
    """

    from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu

    def _p_to_stars_local(p):
        if not np.isfinite(p):
            return "NA"
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    def _norm_subtype_local(v):
        if "normalise_subtype" in globals():
            return normalise_subtype(v)
        return str(v).strip().lower()

    def _resolve_subtypes_local(subtype_order, subtypes_to_plot):
        order = [_norm_subtype_local(s) for s in subtype_order]

        if subtypes_to_plot is None:
            return order

        wanted = [_norm_subtype_local(s) for s in subtypes_to_plot]
        return [s for s in order if s in wanted]

    def _pull_normalise(df, value_col, out_col):
        """
        Previous SAXS pulling normalisation:
            T = subtype mean of per-sample dermis max
            a = per-sample min across dermis+wound
            b = per-sample max across dermis+wound
        """

        d = df.copy()

        for c in ["subtype", "region", "Filenumber"]:
            d[c] = d[c].astype(str).str.strip()

        d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

        scale_regions = set(SAXS_DERMIS_REGIONS + SAXS_WOUND_REGIONS)

        scale_mask = d["region"].isin(scale_regions) & d[value_col].notna()
        dermis_mask = d["region"].isin(SAXS_DERMIS_REGIONS) & d[value_col].notna()

        if not scale_mask.any():
            raise ValueError("No finite values in dermis+wound regions for normalisation.")

        if not dermis_mask.any():
            raise ValueError("No finite dermis values for normalisation.")

        dermis = d.loc[dermis_mask, ["subtype", "Filenumber", value_col]].copy()

        per_sample_dermis_max = (
            dermis
            .groupby(["subtype", "Filenumber"], as_index=False)[value_col]
            .max()
            .rename(columns={value_col: "dermis_max"})
        )

        subtype_T = (
            per_sample_dermis_max
            .groupby("subtype", as_index=False)["dermis_max"]
            .mean()
            .rename(columns={"dermis_max": "T"})
        )

        scale_vals = d.loc[scale_mask, ["subtype", "Filenumber", value_col]].copy()

        per_sample_a = (
            scale_vals
            .groupby(["subtype", "Filenumber"], as_index=False)[value_col]
            .min()
            .rename(columns={value_col: "a"})
        )

        per_sample_b = (
            scale_vals
            .groupby(["subtype", "Filenumber"], as_index=False)[value_col]
            .max()
            .rename(columns={value_col: "b"})
        )

        d = d.merge(subtype_T, on="subtype", how="left")
        d = d.merge(per_sample_a, on=["subtype", "Filenumber"], how="left")
        d = d.merge(per_sample_b, on=["subtype", "Filenumber"], how="left")

        a = pd.to_numeric(d["a"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(d["b"], errors="coerce").to_numpy(float)
        T = pd.to_numeric(d["T"], errors="coerce").to_numpy(float)
        x = pd.to_numeric(d[value_col], errors="coerce").to_numpy(float)

        ok = (
            d["region"].isin(scale_regions).to_numpy()
            & np.isfinite(a)
            & np.isfinite(b)
            & np.isfinite(T)
            & np.isfinite(x)
            & (b > a)
            & (T > 0)
            & (T > a)
        )

        d[out_col] = np.nan

        if ok.any():
            scale = (T[ok] - a[ok]) / (b[ok] - a[ok])
            xprime = a[ok] + (x[ok] - a[ok]) * scale
            xprime = xprime / T[ok]

            if clip:
                xprime = np.clip(xprime, 0.0, 1.0)

            d.loc[ok, out_col] = xprime

        return d.drop(columns=["T", "a", "b"], errors="ignore")

    def _make_stats(sample_means):
        rows = []

        for subtype in present_subtypes:
            ds = sample_means[sample_means["SubtypeClean"].astype(str) == subtype].copy()

            wide = (
                ds.pivot_table(
                    index="Filenumber",
                    columns="Region",
                    values="SampleMean",
                    aggfunc="mean",
                )
                .reset_index()
            )

            if "dermis" not in wide.columns or "wound" not in wide.columns:
                rows.append({
                    "SubtypeClean": subtype,
                    "Label": labels.get(subtype, subtype),
                    "n_dermis": 0,
                    "n_wound": 0,
                    "n_pairs": 0,
                    "mean_dermis": np.nan,
                    "mean_wound": np.nan,
                    "test": stats_test,
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "stars": "NA",
                })
                continue

            dermis = pd.to_numeric(wide["dermis"], errors="coerce").dropna().to_numpy(float)
            wound = pd.to_numeric(wide["wound"], errors="coerce").dropna().to_numpy(float)
            paired = wide[["dermis", "wound"]].apply(pd.to_numeric, errors="coerce").dropna()

            stat = np.nan
            p = np.nan

            if stats_test == "paired":
                if len(paired) >= 2:
                    stat, p = ttest_rel(
                        paired["wound"],
                        paired["dermis"],
                        nan_policy="omit",
                    )

            elif stats_test == "welch":
                if len(dermis) >= 2 and len(wound) >= 2:
                    stat, p = ttest_ind(
                        wound,
                        dermis,
                        equal_var=False,
                        nan_policy="omit",
                    )

            elif stats_test == "mannwhitney":
                if len(dermis) >= 1 and len(wound) >= 1:
                    res = mannwhitneyu(wound, dermis, alternative="two-sided")
                    stat = res.statistic
                    p = res.pvalue

            else:
                raise ValueError("stats_test must be 'welch', 'mannwhitney', or 'paired'.")

            rows.append({
                "SubtypeClean": subtype,
                "Label": labels.get(subtype, subtype),
                "n_dermis": len(dermis),
                "n_wound": len(wound),
                "n_pairs": len(paired),
                "mean_dermis": float(np.nanmean(dermis)) if len(dermis) else np.nan,
                "mean_wound": float(np.nanmean(wound)) if len(wound) else np.nan,
                "test": stats_test,
                "statistic": float(stat) if np.isfinite(stat) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
                "stars": _p_to_stars_local(p),
            })

        return pd.DataFrame(rows)

    def _add_sig_bar(ax, x1, x2, y, h, text):
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + h, y + h, y],
            color="black",
            linewidth=1.0,
            clip_on=False,
            zorder=10,
        )
        ax.text(
            (x1 + x2) / 2,
            y + h,
            text,
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            clip_on=False,
            zorder=11,
        )

    # ------------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------------
    subtype_order = _resolve_subtypes_local(subtype_order, subtypes_to_plot)

    if saxs_points is None or saxs_points.empty:
        print("No SAXS point data provided.")
        return None, None, pd.DataFrame()

    d = saxs_points.copy()
    d = add_clean_subtype(d)

    required = ["subtype", "SubtypeClean", "region", "Filenumber", "parameter", raw_value_col]
    missing = [c for c in required if c not in d.columns]

    if missing:
        raise KeyError(f"saxs_points missing required columns: {missing}")

    for c in ["subtype", "SubtypeClean", "region", "Filenumber", "parameter"]:
        d[c] = d[c].astype(str).str.strip()

    d[raw_value_col] = pd.to_numeric(d[raw_value_col], errors="coerce")

    d = d[
        (d["parameter"] == parameter)
        & d["region"].isin(SAXS_DERMIS_REGIONS + SAXS_WOUND_REGIONS)
        & d[raw_value_col].notna()
    ].copy()

    if d.empty:
        print(f"No SAXS point data available for parameter={parameter}")
        return None, None, pd.DataFrame()

    d = d[d["SubtypeClean"].astype(str).isin(subtype_order)].copy()

    if d.empty:
        print("No matching SAXS subtypes after filtering.")
        return None, None, pd.DataFrame()

    # ------------------------------------------------------------------
    # Thresholds / gates
    # ------------------------------------------------------------------
    if curvearea_thresh is not None and parameter == "curvearea":
        d = d[d[raw_value_col] >= float(curvearea_thresh)].copy()

    if param_thresh is not None:
        d = d[d[raw_value_col] >= float(param_thresh)].copy()

    if saxs_thresh is not None and "total_SAXS_norm_0_1" in d.columns:
        saxs_vals = pd.to_numeric(d["total_SAXS_norm_0_1"], errors="coerce")
        d = d[saxs_vals >= float(saxs_thresh)].copy()

    if d.empty:
        print("No SAXS rows left after threshold filtering.")
        return None, None, pd.DataFrame()

    # ------------------------------------------------------------------
    # Trim before normalisation
    # ------------------------------------------------------------------
    d = trim_by_std(
        d,
        value_col=raw_value_col,
        trim_std_devs=trim_std_devs,
        group_cols=("subtype", "region"),
        print_cols=False,
    )

    if d.empty:
        print("No SAXS rows left after trimming.")
        return None, None, pd.DataFrame()

    # ------------------------------------------------------------------
    # Pull normalisation
    # ------------------------------------------------------------------
    norm_col = f"{raw_value_col}__pull_norm"

    d = _pull_normalise(
        d,
        value_col=raw_value_col,
        out_col=norm_col,
    )

    d = d.dropna(subset=[norm_col]).copy()

    if d.empty:
        print("No SAXS rows left after pull normalisation.")
        return None, None, pd.DataFrame()

    # ------------------------------------------------------------------
    # Pool dermis/wound
    # ------------------------------------------------------------------
    region_map = {
        "dermis_sub": "dermis",
        "dermis_epi": "dermis",
        "wound_sub": "wound",
        "wound_epi": "wound",
        "dermis": "dermis",
        "wound": "wound",
    }

    d = d[d["region"].isin(region_map)].copy()
    d["Region"] = d["region"].map(region_map)

    sample_means = (
        d.groupby(
            ["subtype", "SubtypeClean", "Filenumber", "Region"],
            as_index=False,
            observed=True,
        )[norm_col]
        .mean()
        .rename(columns={norm_col: "SampleMean"})
    )

    summary = (
        sample_means
        .groupby(["SubtypeClean", "Region"], as_index=False, observed=True)["SampleMean"]
        .agg(mean="mean", std="std", n="count")
    )

    present_subtypes = [
        st for st in subtype_order
        if st in set(summary["SubtypeClean"].astype(str))
    ]

    if not present_subtypes:
        print("No matching subtypes to plot.")
        return None, None, pd.DataFrame()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    stats_df = _make_stats(sample_means) if add_stats else pd.DataFrame()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    regions = ["dermis", "wound"]
    region_labels = {
        "dermis": "Dermis",
        "wound": "Wound",
    }

    region_hatches = {
        "dermis": "",
        "wound": "///",
    }

    region_alphas = {
        "dermis": 0.95,
        "wound": 0.45,
    }

    x = np.arange(len(present_subtypes))
    width = 0.82 / len(regions)

    fig, ax = plt.subplots(figsize=(10, 5.8))

    xpos_map = {}

    old_hatch_lw = plt.rcParams.get("hatch.linewidth", 1.0)
    plt.rcParams["hatch.linewidth"] = 0.7

    try:
        for j, region in enumerate(regions):
            offset = (j - (len(regions) - 1) / 2) * width
            xpos = x + offset

            means = []
            errs = []

            for subtype in present_subtypes:
                row = summary[
                    (summary["SubtypeClean"].astype(str) == subtype)
                    & (summary["Region"].astype(str) == region)
                ]

                if row.empty:
                    means.append(np.nan)
                    errs.append(np.nan)
                else:
                    means.append(float(row["mean"].iloc[0]))
                    errs.append(float(row["std"].iloc[0]) if pd.notna(row["std"].iloc[0]) else np.nan)

            means = np.asarray(means, dtype=float)
            errs = np.asarray(errs, dtype=float)

            bar_cols = [colours.get(st, "grey") for st in present_subtypes]

            bars = ax.bar(
                xpos,
                means,
                width=width,
                color=bar_cols,
                alpha=region_alphas.get(region, 0.75),
                edgecolor="black",
                linewidth=0.6,
                label=region_labels.get(region, region),
                zorder=2,
            )

            if hatches:
                for bar, subtype in zip(bars, present_subtypes):
                    hatch_colour = "black" if subtype == "control" else "lightgrey"

                    ax.bar(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        width=bar.get_width(),
                        color="none",
                        edgecolor=hatch_colour,
                        linewidth=0.0,
                        hatch=region_hatches.get(region, ""),
                        alpha=1.0,
                        zorder=3,
                    )

            ax.errorbar(
                xpos,
                means,
                yerr=errs,
                fmt="none",
                ecolor="black",
                capsize=3,
                linewidth=1.1,
                zorder=4,
            )

            for i, subtype in enumerate(present_subtypes):
                xpos_map[(subtype, region)] = xpos[i]

    finally:
        plt.rcParams["hatch.linewidth"] = old_hatch_lw

    # ------------------------------------------------------------------
    # Significance brackets
    # ------------------------------------------------------------------
    if add_stats and not stats_df.empty:
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin if ymax > ymin else 1.0

        y_pad = 0.05 * yrange
        h = 0.035 * yrange
        top_y = ymax

        for subtype in present_subtypes:
            row = stats_df[stats_df["SubtypeClean"].astype(str) == subtype]

            if row.empty:
                continue

            stars = row["stars"].iloc[0]

            if stars in {"NA", "", None}:
                continue

            s = summary[summary["SubtypeClean"].astype(str) == subtype].copy()

            y_vals = []

            for region in regions:
                sr = s[s["Region"].astype(str) == region]

                if sr.empty:
                    continue

                mean = float(sr["mean"].iloc[0])
                std = float(sr["std"].iloc[0]) if pd.notna(sr["std"].iloc[0]) else 0.0

                if np.isfinite(mean):
                    y_vals.append(mean + std)

            if not y_vals:
                continue

            x1 = xpos_map.get((subtype, "dermis"))
            x2 = xpos_map.get((subtype, "wound"))

            if x1 is None or x2 is None:
                continue

            y = max(y_vals) + y_pad

            _add_sig_bar(
                ax,
                x1,
                x2,
                y,
                h,
                stars,
            )

            top_y = max(top_y, y + h + y_pad)

        ax.set_ylim(ymin, top_y)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [labels.get(st, st) for st in present_subtypes],
        rotation=0,
        ha="center",
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)

    if ylim is not None:
        ax.set_ylim(*ylim)

    legend_handles = []

    for region in regions:
        if hatches:
            legend_handles.append(
                Patch(
                    facecolor="white",
                    edgecolor="black",
                    hatch=region_hatches.get(region, ""),
                    label=region_labels.get(region, region),
                )
            )
        else:
            legend_handles.append(
                Patch(
                    facecolor="grey",
                    edgecolor="black",
                    alpha=region_alphas.get(region, 0.75),
                    label=region_labels.get(region, region),
                )
            )

    ax.legend(handles=legend_handles, frameon=False)

    plt.tight_layout()
    plt.show()

    if print_stats and not stats_df.empty:
        print(f"\n[SAXS dermis vs wound tests | test={stats_test}]")
        print(
            stats_df[
                [
                    "Label",
                    "n_dermis",
                    "n_wound",
                    "n_pairs",
                    "mean_dermis",
                    "mean_wound",
                    "test",
                    "statistic",
                    "p_value",
                    "stars",
                ]
            ].to_string(index=False)
        )

    return fig, ax, stats_df

def plot_saxs_total_collagen_4region(
    saxs_points,
    parameter="curvearea",
    normalise=True,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=None,
    trim_std_devs=6,
    hatches=True,
    print_summary=True,
    run_stats=True,
    print_stats=True,
):
    """
    SAXS total collagen intensity split into:
        dermis_sub
        dermis_epi
        wound_sub
        wound_epi

    Uses point-level SAXS data to calculate sample-level means.

    Statistical testing is performed on sample-level means.

    Tests performed:
        1. Within each subtype:
               dermis_sub vs dermis_epi
               wound_sub  vs wound_epi, if present

           normal paired differences     -> paired t-test
           non-normal paired differences -> Wilcoxon signed-rank test

        2. Between subtypes within each region:
               dermis_sub: subtype A vs subtype B
               dermis_epi: subtype A vs subtype B
               wound_sub:  subtype A vs subtype B, if present
               wound_epi:  subtype A vs subtype B, if present

           two groups:
               normal groups     -> independent t-test
               non-normal groups -> Mann-Whitney U test

           more than two groups:
               all normal     -> one-way ANOVA
               any non-normal -> Kruskal-Wallis test
    """

    subtype_order = resolve_subtypes_to_plot(
        subtype_order,
        subtypes_to_plot,
    )

    # ------------------------------------------------------------
    # Prepare SAXS point-level data
    # ------------------------------------------------------------
    d, plot_value_col = prepare_saxs_point_data_for_plot(
        saxs_points=saxs_points,
        parameter=parameter,
        normalise=normalise,
        trim_std_devs=trim_std_devs,
    )

    d = d[
        d["region"].isin(SAXS_REGION_ORDER_4)
    ].copy()

    if d.empty:
        print(f"No split-region SAXS point data available for parameter={parameter}")
        return None, None, pd.DataFrame()

    # ------------------------------------------------------------
    # Convert point-level values into sample-level means
    # ------------------------------------------------------------
    sample_means, summary = summarise_saxs_points_to_sample_bars(
        d,
        value_col=plot_value_col,
        region_col="region",
    )

    if print_summary:
        print_saxs_bar_summary(
            summary,
            title="SAXS total collagen 4-region bar means",
        )

    ylabel = (
        "Normalised total collagen intensity"
        if normalise
        else SAXS_PARAMETER_LABELS.get(parameter, parameter)
    )

    # ------------------------------------------------------------
    # Plot grouped bars
    # ------------------------------------------------------------
    fig, ax = _plot_saxs_grouped_bars(
        summary=summary,
        points=sample_means.rename(
            columns={
                "SampleMean": plot_value_col,
            }
        ),
        regions=SAXS_REGION_ORDER_4,
        title=" ",
        ylabel=ylabel,
        region_col_points="Region",
        value_col=plot_value_col,
        subtype_order=subtype_order,
        ylim=ylim,
        hatches=hatches,
        show=False,
    )

    # ------------------------------------------------------------
    # Automatic statistics
    # ------------------------------------------------------------
    stats_df = pd.DataFrame()

    if run_stats and fig is not None and ax is not None:

        all_stats = []

        if "Region" not in sample_means.columns:

            print("\n[SAXS total collagen 4-region stats] No Region column found in sample_means.")
            print("Available columns:")
            print(sample_means.columns.tolist())

        else:

            # ------------------------------------------------------------
            # 1. Within-subtype paired spatial tests
            # ------------------------------------------------------------
            paired_region_pairs = [
                ("dermis_sub", "dermis_epi"),
                ("wound_sub", "wound_epi"),
            ]

            paired_stats = auto_saxs_multiple_paired_region_tests(
                sample_means=sample_means,
                region_pairs=paired_region_pairs,
                value_col="SampleMean",
                subtype_order=subtype_order,
                region_col="Region",
                sample_col=None,  # auto-detects Filenumber / SampleKey / etc.
            )

            if not paired_stats.empty:

                all_stats.append(paired_stats)

                if print_stats:

                    paired_print = paired_stats.copy()

                    for c in [
                        "MeanA",
                        "MeanB",
                        "MeanDifference_B_minus_A",
                        "Normality_p",
                        "t_stat",
                        "statistic",
                        "p_value",
                    ]:
                        if c in paired_print.columns:
                            paired_print[c] = pd.to_numeric(
                                paired_print[c],
                                errors="coerce",
                            )

                    paired_cols = [
                        c for c in [
                            "ComparisonType",
                            "Label",
                            "RegionA",
                            "RegionB",
                            "n_pairs",
                            "MeanA",
                            "MeanB",
                            "MeanDifference_B_minus_A",
                            "NormalityTest",
                            "Normality_p",
                            "NormalityInterpretation",
                            "SelectedTest",
                            "statistic",
                            "p_value",
                            "stars",
                        ]
                        if c in paired_print.columns
                    ]

                    print("\n[SAXS automatic paired spatial tests: total collagen lower/sub vs upper/epi regions]")
                    print(
                        paired_print[paired_cols]
                        .to_string(index=False)
                    )

            # ------------------------------------------------------------
            # 2. Between-subtype tests within each region
            # ------------------------------------------------------------
            between_stats = auto_saxs_between_subtype_tests(
                sample_means=sample_means,
                value_col="SampleMean",
                subtype_order=subtype_order,
                region_col="Region",
                sample_col=None,  # auto-detects Filenumber / SampleKey / etc.
            )

            if not between_stats.empty:

                between_stats["ComparisonType"] = "between-subtype test within region"
                all_stats.append(between_stats)

                if print_stats:

                    between_print = between_stats.copy()

                    for c in [
                        "statistic",
                        "p_value",
                    ]:
                        if c in between_print.columns:
                            between_print[c] = pd.to_numeric(
                                between_print[c],
                                errors="coerce",
                            )

                    between_cols = [
                        c for c in [
                            "ComparisonType",
                            "Region",
                            "Comparison",
                            "n_groups",
                            "GroupNs",
                            "NormalityTest",
                            "Normality_p_values",
                            "NormalityInterpretation",
                            "SelectedTest",
                            "statistic",
                            "p_value",
                            "stars",
                        ]
                        if c in between_print.columns
                    ]

                    print("\n[SAXS automatic between-subtype tests within each region: total collagen]")
                    print(
                        between_print[between_cols]
                        .to_string(index=False)
                    )

            if all_stats:
                stats_df = pd.concat(
                    all_stats,
                    ignore_index=True,
                    sort=False,
                )

    plt.tight_layout()
    plt.show()

    return fig, ax, stats_df

def normalise_saxs_by_subtype_target_pull_sample_endpoints(
    df,
    value_col,
    out_col=None,
    clip=True,
    to_unit_interval=True,
    b_from="five_regions",
    T_from="mean_dermis_max",
):
    """
    Normalise raw SAXS values per subtype.

    Per subtype:
        T = mean of per-sample dermis max

    Per sample:
        a = min across dermis + wound regions
        b = max across dermis + wound regions, or dermis only

    Transform:
        x' = a + (x - a) * (T - a) / (b - a)

    If to_unit_interval=True:
        x' = x' / T
    """

    if df is None or df.empty:
        return pd.DataFrame()

    if value_col not in df.columns:
        raise KeyError(f"Missing value column: {value_col}")

    out_col = out_col or f"{value_col}__subtype_affine"

    d = df.copy()

    for c in ["subtype", "region", "Filenumber"]:
        if c not in d.columns:
            raise KeyError(f"Missing required SAXS column: {c}")
        d[c] = d[c].astype(str).str.strip()

    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    scale_regions = set(SAXS_DERMIS_REGIONS + SAXS_WOUND_REGIONS)

    five_mask = d["region"].isin(scale_regions) & d[value_col].notna()

    if not five_mask.any():
        raise ValueError("No finite SAXS values in dermis/wound regions.")

    derm_mask = d["region"].isin(SAXS_DERMIS_REGIONS) & d[value_col].notna()
    derm = d.loc[derm_mask, ["subtype", "Filenumber", value_col]].copy()

    if derm.empty:
        raise ValueError("No finite SAXS dermis values found for normalisation.")

    per_sample_dermis_max = (
        derm.groupby(["subtype", "Filenumber"], as_index=False)[value_col]
        .max()
        .rename(columns={value_col: "dermis_max"})
    )

    if T_from == "mean_dermis_max":
        subtype_T = (
            per_sample_dermis_max.groupby("subtype", as_index=False)["dermis_max"]
            .mean()
            .rename(columns={"dermis_max": "T"})
        )
    else:
        raise ValueError("Unsupported T_from")

    five = d.loc[five_mask, ["subtype", "Filenumber", value_col]].copy()

    per_sample_a = (
        five.groupby(["subtype", "Filenumber"], as_index=False)[value_col]
        .min()
        .rename(columns={value_col: "a"})
    )

    if b_from == "five_regions":
        per_sample_b = (
            five.groupby(["subtype", "Filenumber"], as_index=False)[value_col]
            .max()
            .rename(columns={value_col: "b"})
        )
    elif b_from == "dermis":
        per_sample_b = per_sample_dermis_max.rename(columns={"dermis_max": "b"})
    else:
        raise ValueError("b_from must be 'five_regions' or 'dermis'")

    d = d.merge(subtype_T, on="subtype", how="left")
    d = d.merge(per_sample_a, on=["subtype", "Filenumber"], how="left")
    d = d.merge(per_sample_b, on=["subtype", "Filenumber"], how="left")

    five_mask = d["region"].isin(scale_regions) & d[value_col].notna()

    a = pd.to_numeric(d["a"], errors="coerce").to_numpy(float)
    b = pd.to_numeric(d["b"], errors="coerce").to_numpy(float)
    T = pd.to_numeric(d["T"], errors="coerce").to_numpy(float)
    x = pd.to_numeric(d[value_col], errors="coerce").to_numpy(float)

    ok = (
        five_mask.to_numpy()
        & np.isfinite(a)
        & np.isfinite(b)
        & np.isfinite(T)
        & np.isfinite(x)
        & (b > a)
        & (T > 0)
        & (T > a)
    )

    d[out_col] = np.nan

    if ok.any():
        scale = (T[ok] - a[ok]) / (b[ok] - a[ok])
        xprime = a[ok] + (x[ok] - a[ok]) * scale

        if to_unit_interval:
            xprime = xprime / T[ok]

        if clip and to_unit_interval:
            xprime = np.clip(xprime, 0.0, 1.0)

        d.loc[ok, out_col] = xprime

    return d.drop(columns=["T", "a", "b"], errors="ignore")

def prepare_saxs_points_parameter(
    saxs_points,
    parameter,
    raw_value_col="value",
    trim_std_devs=6,
    require_regions=None,
):
    """
    Prepare SAXS point-level data for a parameter without normalisation.

    Used for:
        D_period
        peak_width_q / FWHM
        wa_moment etc.
    """

    if saxs_points is None or saxs_points.empty:
        return pd.DataFrame(), raw_value_col

    d = saxs_points.copy()
    d = add_clean_subtype(d)

    required = ["subtype", "region", "Filenumber", "parameter", raw_value_col]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise KeyError(f"saxs_points missing columns: {missing}")

    d["parameter"] = d["parameter"].astype(str).str.strip()
    d["region"] = d["region"].astype(str).str.strip()
    d["subtype"] = d["subtype"].astype(str).str.strip()
    d["Filenumber"] = d["Filenumber"].astype(str).str.strip()
    d[raw_value_col] = pd.to_numeric(d[raw_value_col], errors="coerce")

    if require_regions is None:
        require_regions = SAXS_DERMIS_REGIONS + SAXS_WOUND_REGIONS

    d = d[
        (d["parameter"] == parameter)
        & d["region"].isin(require_regions)
        & d[raw_value_col].notna()
    ].copy()

    if d.empty:
        return d, raw_value_col

    d = trim_by_std(
        d,
        value_col=raw_value_col,
        trim_std_devs=trim_std_devs,
        group_cols=("subtype", "region"),
        print_cols=False,
    )

    return d.reset_index(drop=True), raw_value_col

def prepare_saxs_region_summary_from_points(
    saxs_points,
    parameter,
    regions,
    pool_to_dermis_wound=False,
    trim_std_devs=6,
):
    """
    Prepare point-level SAXS parameter into:
        sample_means
        summary

    If pool_to_dermis_wound=True:
        dermis_sub + dermis_epi -> dermis
        wound_sub + wound_epi -> wound
    """

    d, value_col = prepare_saxs_points_parameter(
        saxs_points=saxs_points,
        parameter=parameter,
        trim_std_devs=trim_std_devs,
    )

    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), value_col

    if pool_to_dermis_wound:
        d = pool_saxs_dermis_wound_regions(d)
        d = d.rename(columns={"RegionPooled": "RegionForPlot"})
        region_col = "RegionForPlot"
    else:
        d = d[d["region"].isin(regions)].copy()
        region_col = "region"

    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), value_col

    sample_means, summary = summarise_saxs_points_to_sample_bars(
        d,
        value_col=value_col,
        region_col=region_col,
    )

    return sample_means, summary, value_col

def plot_saxs_dperiod_dermis_vs_wound(
    saxs_points,
    parameter=SAXS_DPERIOD_PARAM,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=SAXS_DPERIOD_YLIM,
    trim_std_devs=6,
    hatches=True,
    print_summary=True,
    run_stats=True,
    print_stats=True,
):
    """
    Bar plot of D-period in dermis vs wound.

    No normalisation.

    Uses point-level SAXS data to calculate sample-level means.

    Statistical testing is performed on sample-level means.

    Dynamic statistics:
        1. If both dermis and wound are present:
               paired dermis vs wound test within each subtype

               normal paired differences     -> paired t-test
               non-normal paired differences -> Wilcoxon signed-rank test

        2. If only one region is present:
               compare between subtypes within that region

               two groups:
                   normal groups     -> independent t-test
                   non-normal groups -> Mann-Whitney U test

               more than two groups:
                   all normal     -> one-way ANOVA
                   any non-normal -> Kruskal-Wallis test
    """

    subtype_order = resolve_subtypes_to_plot(
        subtype_order,
        subtypes_to_plot,
    )

    # ------------------------------------------------------------
    # Prepare sample-level D-period means
    # ------------------------------------------------------------
    sample_means, summary, value_col = prepare_saxs_region_summary_from_points(
        saxs_points=saxs_points,
        parameter=parameter,
        regions=["dermis", "wound"],
        pool_to_dermis_wound=True,
        trim_std_devs=trim_std_devs,
    )

    if summary.empty:
        print(f"No SAXS D-period data available for parameter={parameter}")
        return None, None, pd.DataFrame()

    if print_summary:
        print_saxs_bar_summary(
            summary,
            title="SAXS D-period dermis vs wound bar means",
        )

    # ------------------------------------------------------------
    # Plot grouped bars
    # ------------------------------------------------------------
    fig, ax = _plot_saxs_grouped_bars(
        summary=summary,
        points=sample_means.rename(
            columns={
                "SampleMean": value_col,
            }
        ),
        regions=["dermis", "wound"],
        title="SAXS D-period in dermis and wound",
        ylabel="D-period (nm)",
        region_col_points="Region",
        value_col=value_col,
        subtype_order=subtype_order,
        ylim=ylim,
        hatches=hatches,
        show=False,
    )

    # ------------------------------------------------------------
    # Automatic statistics
    # ------------------------------------------------------------
    stats_df = pd.DataFrame()

    if run_stats and fig is not None and ax is not None:

        if "Region" not in sample_means.columns:

            print("\n[SAXS D-period stats] No Region column found in sample_means.")
            print("Available columns:")
            print(sample_means.columns.tolist())

        else:

            regions_present = set(
                sample_means["Region"]
                .astype(str)
                .str.lower()
                .str.strip()
                .unique()
            )

            # ------------------------------------------------------------
            # Case 1: paired dermis vs wound within each subtype
            # ------------------------------------------------------------
            if {"dermis", "wound"}.issubset(regions_present):

                stats_df = auto_saxs_paired_region_tests(
                    sample_means=sample_means,
                    value_col="SampleMean",
                    subtype_order=subtype_order,
                    region_col="Region",
                    sample_col=None,
                    region_a="dermis",
                    region_b="wound",
                )

                if not stats_df.empty:

                    add_paired_region_annotations(
                        ax,
                        stats_df,
                        summary,
                        subtype_order=subtype_order,
                        regions=("dermis", "wound"),
                        region_a="dermis",
                        region_b="wound",
                    )

                if print_stats and not stats_df.empty:

                    stats_print = stats_df.copy()

                    for c in [
                        "MeanA",
                        "MeanB",
                        "MeanDifference_B_minus_A",
                        "Normality_p",
                        "t_stat",
                        "statistic",
                        "p_value",
                    ]:
                        if c in stats_print.columns:
                            stats_print[c] = pd.to_numeric(
                                stats_print[c],
                                errors="coerce",
                            )

                    paired_cols = [
                        c for c in [
                            "Label",
                            "RegionA",
                            "RegionB",
                            "n_pairs",
                            "MeanA",
                            "MeanB",
                            "MeanDifference_B_minus_A",
                            "NormalityTest",
                            "Normality_p",
                            "NormalityInterpretation",
                            "SelectedTest",
                            "statistic",
                            "p_value",
                            "stars",
                        ]
                        if c in stats_print.columns
                    ]

                    print("\n[SAXS automatic paired tests: D-period dermis vs wound]")
                    print(
                        stats_print[paired_cols]
                        .to_string(index=False)
                    )

            # ------------------------------------------------------------
            # Case 2: only one region present, compare between subtypes
            # ------------------------------------------------------------
            else:

                stats_df = auto_saxs_between_subtype_tests(
                    sample_means=sample_means,
                    value_col="SampleMean",
                    subtype_order=subtype_order,
                    region_col="Region",
                    sample_col=None,
                )

                if print_stats and not stats_df.empty:

                    stats_print = stats_df.copy()

                    for c in [
                        "statistic",
                        "p_value",
                    ]:
                        if c in stats_print.columns:
                            stats_print[c] = pd.to_numeric(
                                stats_print[c],
                                errors="coerce",
                            )

                    between_cols = [
                        c for c in [
                            "Region",
                            "Comparison",
                            "n_groups",
                            "GroupNs",
                            "NormalityTest",
                            "Normality_p_values",
                            "NormalityInterpretation",
                            "SelectedTest",
                            "statistic",
                            "p_value",
                            "stars",
                        ]
                        if c in stats_print.columns
                    ]

                    print("\n[SAXS automatic between-subtype tests: D-period]")
                    print(
                        stats_print[between_cols]
                        .to_string(index=False)
                    )

    plt.tight_layout()
    plt.show()

    return fig, ax, stats_df

def plot_saxs_dperiod_4region(
    saxs_points,
    parameter=SAXS_DPERIOD_PARAM,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=SAXS_DPERIOD_YLIM,
    trim_std_devs=6,
    hatches=True,
    print_summary=True,
    run_stats=True,
    print_stats=True,
):
    """
    Bar plot of D-period split into:
        dermis_sub
        dermis_epi
        wound_sub
        wound_epi

    Uses point-level SAXS data to calculate sample-level means.

    Statistical testing is performed on sample-level means.

    Tests performed:
        1. Within each subtype:
               dermis_sub vs dermis_epi
               wound_sub  vs wound_epi, if present

           normal paired differences     -> paired t-test
           non-normal paired differences -> Wilcoxon signed-rank test

        2. Between subtypes within each region:
               dermis_sub: subtype A vs subtype B
               dermis_epi: subtype A vs subtype B
               wound_sub:  subtype A vs subtype B, if present
               wound_epi:  subtype A vs subtype B, if present

           two groups:
               normal groups     -> independent t-test
               non-normal groups -> Mann-Whitney U test

           more than two groups:
               all normal     -> one-way ANOVA
               any non-normal -> Kruskal-Wallis test
    """

    subtype_order = resolve_subtypes_to_plot(
        subtype_order,
        subtypes_to_plot,
    )

    # ------------------------------------------------------------
    # Prepare sample-level D-period means
    # ------------------------------------------------------------
    sample_means, summary, value_col = prepare_saxs_region_summary_from_points(
        saxs_points=saxs_points,
        parameter=parameter,
        regions=SAXS_REGION_ORDER_4,
        pool_to_dermis_wound=False,
        trim_std_devs=trim_std_devs,
    )

    if summary.empty:
        print(f"No split-region SAXS D-period data available for parameter={parameter}")
        return None, None, pd.DataFrame()

    if print_summary:
        print_saxs_bar_summary(
            summary,
            title="SAXS D-period 4-region bar means",
        )

    # ------------------------------------------------------------
    # Plot grouped bars
    # ------------------------------------------------------------
    fig, ax = _plot_saxs_grouped_bars(
        summary=summary,
        points=sample_means.rename(
            columns={
                "SampleMean": value_col,
            }
        ),
        regions=SAXS_REGION_ORDER_4,
        title="SAXS D-period by spatial region",
        ylabel="D-period (nm)",
        region_col_points="Region",
        value_col=value_col,
        subtype_order=subtype_order,
        ylim=ylim,
        hatches=hatches,
        show=False,
    )

    # ------------------------------------------------------------
    # Automatic statistics
    # ------------------------------------------------------------
    stats_df = pd.DataFrame()

    if run_stats and fig is not None and ax is not None:

        all_stats = []

        if "Region" not in sample_means.columns:

            print("\n[SAXS D-period 4-region stats] No Region column found in sample_means.")
            print("Available columns:")
            print(sample_means.columns.tolist())

        else:

            # ------------------------------------------------------------
            # 1. Within-subtype paired spatial tests
            # ------------------------------------------------------------
            paired_region_pairs = [
                ("dermis_sub", "dermis_epi"),
                ("wound_sub", "wound_epi"),
            ]

            paired_stats = auto_saxs_multiple_paired_region_tests(
                sample_means=sample_means,
                region_pairs=paired_region_pairs,
                value_col="SampleMean",
                subtype_order=subtype_order,
                region_col="Region",
                sample_col=None,
            )

            if not paired_stats.empty:

                all_stats.append(paired_stats)

                if print_stats:

                    paired_print = paired_stats.copy()

                    for c in [
                        "MeanA",
                        "MeanB",
                        "MeanDifference_B_minus_A",
                        "Normality_p",
                        "t_stat",
                        "statistic",
                        "p_value",
                    ]:
                        if c in paired_print.columns:
                            paired_print[c] = pd.to_numeric(
                                paired_print[c],
                                errors="coerce",
                            )

                    paired_cols = [
                        c for c in [
                            "ComparisonType",
                            "Label",
                            "RegionA",
                            "RegionB",
                            "n_pairs",
                            "MeanA",
                            "MeanB",
                            "MeanDifference_B_minus_A",
                            "NormalityTest",
                            "Normality_p",
                            "NormalityInterpretation",
                            "SelectedTest",
                            "statistic",
                            "p_value",
                            "stars",
                        ]
                        if c in paired_print.columns
                    ]

                    print("\n[SAXS automatic paired spatial tests: D-period lower/sub vs upper/epi regions]")
                    print(
                        paired_print[paired_cols]
                        .to_string(index=False)
                    )

            # ------------------------------------------------------------
            # 2. Between-subtype tests within each region
            # ------------------------------------------------------------
            between_stats = auto_saxs_between_subtype_tests(
                sample_means=sample_means,
                value_col="SampleMean",
                subtype_order=subtype_order,
                region_col="Region",
                sample_col=None,
            )

            if not between_stats.empty:

                between_stats["ComparisonType"] = "between-subtype test within region"
                all_stats.append(between_stats)

                if print_stats:

                    between_print = between_stats.copy()

                    for c in [
                        "statistic",
                        "p_value",
                    ]:
                        if c in between_print.columns:
                            between_print[c] = pd.to_numeric(
                                between_print[c],
                                errors="coerce",
                            )

                    between_cols = [
                        c for c in [
                            "ComparisonType",
                            "Region",
                            "Comparison",
                            "n_groups",
                            "GroupNs",
                            "NormalityTest",
                            "Normality_p_values",
                            "NormalityInterpretation",
                            "SelectedTest",
                            "statistic",
                            "p_value",
                            "stars",
                        ]
                        if c in between_print.columns
                    ]

                    print("\n[SAXS automatic between-subtype tests within each region: D-period]")
                    print(
                        between_print[between_cols]
                        .to_string(index=False)
                    )

            if all_stats:
                stats_df = pd.concat(
                    all_stats,
                    ignore_index=True,
                    sort=False,
                )

    plt.tight_layout()
    plt.show()

    return fig, ax, stats_df

def calculate_saxs_dperiod_shift_dermis_to_wound(
    saxs_points,
    parameter=SAXS_DPERIOD_PARAM,
    trim_std_devs=6,
):
    """
    Calculate D-period % shift from dermis to wound.

    Uses pooled dermis and pooled wound sample means.
    """

    sample_means, _, _ = prepare_saxs_region_summary_from_points(
        saxs_points=saxs_points,
        parameter=parameter,
        regions=["dermis", "wound"],
        pool_to_dermis_wound=True,
        trim_std_devs=trim_std_devs,
    )

    if sample_means.empty:
        return pd.DataFrame(), pd.DataFrame()

    wide = (
        sample_means
        .pivot_table(
            index=["experiment", "subtype", "SubtypeClean", "Filenumber"],
            columns="Region",
            values="SampleMean",
            aggfunc="first",
        )
        .reset_index()
    )

    if "dermis" not in wide.columns or "wound" not in wide.columns:
        return pd.DataFrame(), pd.DataFrame()

    wide["PercentShift"] = ((wide["wound"] - wide["dermis"]) / wide["dermis"]) * 100
    wide["ShiftType"] = "wound_vs_dermis"

    shift_df = wide[
        [
            "experiment",
            "subtype",
            "SubtypeClean",
            "Filenumber",
            "ShiftType",
            "dermis",
            "wound",
            "PercentShift",
        ]
    ].dropna(subset=["PercentShift"]).copy()

    summary = (
        shift_df
        .groupby(["SubtypeClean", "ShiftType"], as_index=False)["PercentShift"]
        .agg(mean="mean", std="std", n="count")
    )

    return shift_df, summary

def _plot_saxs_shift_bars(
    shift_df,
    summary,
    shift_order,
    title,
    ylabel,
    subtype_order=SUBTYPE_ORDER,
    colours=NI_COLOURS,
    labels=NI_LABELS,
    shift_labels=None,
    ylim=None,
    jitter=0.045,
    point_size=24,
    alpha_range=(0.95, 0.45),
    hatches=True,
    shift_hatches=None,
    hatch_linewidth=0.7,
    keep_empty_subtypes=True,
    bar_group_width=0.78,
):
    """
    Generic SAXS % shift bar plot.

    keep_empty_subtypes=True:
        Keeps empty x-axis spaces for subtypes with no shift value,
        e.g. Unwounded/control, but draws no bar there.

    bar_group_width:
        Total width allocated to the grouped bars at each subtype tick.
    """

    if summary is None or summary.empty:
        print("No SAXS shift data to plot.")
        return None, None

    shift_labels = shift_labels or {
        "wound_vs_dermis": "Wound vs dermis",
        "lower_wound_vs_dermis": "Lower wound vs lower dermis",
        "upper_wound_vs_dermis": "Upper wound vs upper dermis",
        "lower_wound_vs_lower_dermis": "Lower wound vs lower dermis",
        "upper_wound_vs_upper_dermis": "Upper wound vs upper dermis",
        "upper_dermis_vs_lower_dermis": "Upper dermis vs lower dermis",
        "upper_wound_vs_lower_wound": "Upper wound vs lower wound",
    }

    if shift_hatches is None:
        shift_hatches = {
            "wound_vs_dermis": "xxx",
            "lower_wound_vs_dermis": "///",
            "upper_wound_vs_dermis": "\\\\\\",
            "lower_wound_vs_lower_dermis": "///",
            "upper_wound_vs_upper_dermis": "\\\\\\",
            "upper_dermis_vs_lower_dermis": "///",
            "upper_wound_vs_lower_wound": "xxx",
        }

    available_subtypes = set(summary["SubtypeClean"].astype(str))

    if keep_empty_subtypes:
        present_subtypes = list(subtype_order)
    else:
        present_subtypes = [
            st for st in subtype_order
            if st in available_subtypes
        ]

    if not present_subtypes:
        print("No matching SAXS shift subtypes to plot.")
        return None, None

    x = np.arange(len(present_subtypes))
    width = bar_group_width / max(len(shift_order), 1)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    rng = np.random.default_rng(0)

    shift_alphas = np.linspace(alpha_range[0], alpha_range[1], len(shift_order))

    old_hatch_lw = plt.rcParams.get("hatch.linewidth", 1.0)
    plt.rcParams["hatch.linewidth"] = hatch_linewidth

    try:
        for j, shift_type in enumerate(shift_order):
            offset = (j - (len(shift_order) - 1) / 2) * width

            means = []
            errs = []

            for subtype in present_subtypes:
                row = summary[
                    (summary["SubtypeClean"].astype(str) == subtype)
                    & (summary["ShiftType"] == shift_type)
                ]

                if row.empty:
                    means.append(np.nan)
                    errs.append(np.nan)
                else:
                    means.append(float(row["mean"].iloc[0]))
                    errs.append(
                        float(row["std"].iloc[0])
                        if pd.notna(row["std"].iloc[0])
                        else np.nan
                    )

            means = np.asarray(means, dtype=float)
            errs = np.asarray(errs, dtype=float)
            
            # Missing subtype/shift combinations, e.g. Unwounded, are plotted as exactly zero
            # with zero error, so the x-axis space is preserved and aligned with other plots.
            missing = ~np.isfinite(means)
            means[missing] = 0.0
            errs[missing] = 0.0
            
            xpos = x + offset
            
            bars = ax.bar(
                xpos,
                means,
                width=width,
                color=[colours.get(st, "grey") for st in present_subtypes],
                alpha=shift_alphas[j],
                edgecolor="black",
                linewidth=0.6,
                zorder=2,
            )

            if hatches:
                for bar, subtype in zip(bars, present_subtypes):
                    hatch_colour = "black" if subtype == "control" else "lightgrey"
            
                    ax.bar(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        width=bar.get_width(),
                        color="none",
                        edgecolor=hatch_colour,
                        linewidth=0.0,
                        hatch=shift_hatches.get(shift_type, ""),
                        alpha=1.0,
                        zorder=3,
                    )

            ax.errorbar(
                xpos,
                means,
                yerr=errs,
                fmt="none",
                ecolor="black",
                capsize=3,
                linewidth=1.1,
                zorder=4,
            )

            if shift_df is not None and not shift_df.empty:
                for i, subtype in enumerate(present_subtypes):
                    vals = shift_df.loc[
                        (shift_df["SubtypeClean"].astype(str) == subtype)
                        & (shift_df["ShiftType"] == shift_type),
                        "PercentShift",
                    ].to_numpy(dtype=float)

                    vals = vals[np.isfinite(vals)]

                    if vals.size == 0:
                        continue

                    xj = xpos[i] + rng.normal(0, jitter, size=vals.size)

                    # ax.scatter(
                    #     xj,
                    #     vals,
                    #     color="black",
                    #     s=point_size,
                    #     alpha=0.65,
                    #     linewidths=0,
                    #     zorder=5,
                    # )

    finally:
        plt.rcParams["hatch.linewidth"] = old_hatch_lw

    ax.axhline(0, color="black", linewidth=1, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [labels.get(st, st) for st in present_subtypes],
        rotation=0,
        ha="center",
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)

    legend_handles = []

    for j, shift_type in enumerate(shift_order):
        if hatches:
            legend_handles.append(
                Patch(
                    facecolor="white",
                    edgecolor="black",
                    hatch=shift_hatches.get(shift_type, ""),
                    label=shift_labels.get(shift_type, shift_type),
                )
            )
        else:
            legend_handles.append(
                Patch(
                    facecolor="grey",
                    edgecolor="black",
                    alpha=shift_alphas[j],
                    label=shift_labels.get(shift_type, shift_type),
                )
            )

    ax.legend(handles=legend_handles, frameon=False)

    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()

    return fig, ax

def plot_saxs_dperiod_shift_dermis_to_wound(
    saxs_points,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=None,
    trim_std_devs=6,
    hatches=True,
    keep_empty_subtypes=True,
    bar_group_width=0.78,
):
    """
    D-period % shift from pooled dermis to pooled wound.
    """

    subtype_order = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)

    shift_df, summary = calculate_saxs_dperiod_shift_dermis_to_wound(
        saxs_points=saxs_points,
        trim_std_devs=trim_std_devs,
    )

    return _plot_saxs_shift_bars(
        shift_df=shift_df,
        summary=summary,
        shift_order=["wound_vs_dermis"],
        title="",
        ylabel="D-period shift from dermis (%)",
        subtype_order=subtype_order,
        ylim=ylim,
        hatches=hatches,
        keep_empty_subtypes=keep_empty_subtypes,
        bar_group_width=bar_group_width,
    )

def prepare_saxs_dperiod_fwhm_from_spread(
    saxs_points,
    regions,
    pool_to_dermis_wound=False,
    trim_std_devs=6,
):
    """
    Calculate D-period FWHM from the within-sample spread of raw D-period values.

    This matches the old SAXS code:
        sample std = std of raw D-period points within sample/region
        FWHM = 2.354820045 * sample std
    """

    d, value_col = prepare_saxs_points_parameter(
        saxs_points=saxs_points,
        parameter="D_period",
        trim_std_devs=trim_std_devs,
    )

    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), "FWHM"

    if pool_to_dermis_wound:
        d = pool_saxs_dermis_wound_regions(d)
        d = d.rename(columns={"RegionPooled": "RegionForPlot"})
        region_col = "RegionForPlot"
    else:
        d = d[d["region"].isin(regions)].copy()
        region_col = "region"

    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), "FWHM"

    spread_samples = (
        d.groupby(
            ["experiment", "subtype", "SubtypeClean", "Filenumber", region_col],
            observed=True,
            as_index=False,
        )[value_col]
        .agg(std="std", n_points="count")
        .rename(columns={region_col: "Region"})
    )

    spread_samples["FWHM"] = 2.354820045 * pd.to_numeric(
        spread_samples["std"],
        errors="coerce",
    )

    summary = (
        spread_samples
        .groupby(["SubtypeClean", "Region"], observed=True)["FWHM"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )

    return spread_samples, summary, "FWHM"

def plot_saxs_dperiod_fwhm_dermis_vs_wound(
    saxs_points,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=None,
    trim_std_devs=6,
    hatches=True,
    print_summary=True,
    run_stats=True,
    print_stats=True,
):
    """
    D-period FWHM in dermis vs wound.

    FWHM is calculated as:
        2.3548 * within-sample SD of raw D-period values

    Statistical testing is performed on sample-level FWHM values.

    Dynamic statistics:
        1. If both dermis and wound are present:
               paired dermis vs wound test within each subtype

               normal paired differences     -> paired t-test
               non-normal paired differences -> Wilcoxon signed-rank test

        2. If only one region is present:
               compare between subtypes within that region

               two groups:
                   normal groups     -> independent t-test
                   non-normal groups -> Mann-Whitney U test

               more than two groups:
                   all normal     -> one-way ANOVA
                   any non-normal -> Kruskal-Wallis test
    """

    subtype_order = resolve_subtypes_to_plot(
        subtype_order,
        subtypes_to_plot,
    )

    # ------------------------------------------------------------
    # Prepare sample-level D-period FWHM values
    # ------------------------------------------------------------
    spread_samples, summary, value_col = prepare_saxs_dperiod_fwhm_from_spread(
        saxs_points=saxs_points,
        regions=["dermis", "wound"],
        pool_to_dermis_wound=True,
        trim_std_devs=trim_std_devs,
    )

    if summary.empty:
        print("No SAXS D-period FWHM data available.")
        return None, None, pd.DataFrame()

    if print_summary:
        print_saxs_bar_summary(
            summary,
            title="SAXS D-period FWHM dermis vs wound bar means",
        )

    # ------------------------------------------------------------
    # Plot grouped bars
    # ------------------------------------------------------------
    fig, ax = _plot_saxs_grouped_bars(
        summary=summary,
        points=spread_samples.rename(
            columns={
                "FWHM": value_col,
            }
        ),
        regions=["dermis", "wound"],
        title="",
        ylabel="FWHM of D-period (nm)",
        region_col_points="Region",
        value_col=value_col,
        subtype_order=subtype_order,
        ylim=ylim,
        hatches=hatches,
        show=False,
    )

    # ------------------------------------------------------------
    # Automatic statistics
    # ------------------------------------------------------------
    stats_df = pd.DataFrame()

    if run_stats and fig is not None and ax is not None:

        if spread_samples is None or spread_samples.empty:

            print("\n[SAXS D-period FWHM stats] No spread_samples data available.")

        elif "Region" not in spread_samples.columns:

            print("\n[SAXS D-period FWHM stats] No Region column found in spread_samples.")
            print("Available columns:")
            print(spread_samples.columns.tolist())

        else:

            # ------------------------------------------------------------
            # Prepare stats input
            # ------------------------------------------------------------
            stats_input = spread_samples.copy()

            if "FWHM" in stats_input.columns:

                stats_input = stats_input.rename(
                    columns={
                        "FWHM": "SampleMean",
                    }
                )

            elif value_col in stats_input.columns:

                stats_input = stats_input.rename(
                    columns={
                        value_col: "SampleMean",
                    }
                )

            elif "SampleMean" not in stats_input.columns:

                print("\n[SAXS D-period FWHM stats] Could not find FWHM value column.")
                print("Expected one of: 'FWHM', value_col, or 'SampleMean'")
                print("value_col =", value_col)
                print("Available columns:")
                print(stats_input.columns.tolist())
                stats_input = pd.DataFrame()

            if stats_input is not None and not stats_input.empty:

                regions_present = set(
                    stats_input["Region"]
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    .unique()
                )

                # ------------------------------------------------------------
                # Case 1: paired dermis vs wound within each subtype
                # ------------------------------------------------------------
                if {"dermis", "wound"}.issubset(regions_present):

                    stats_df = auto_saxs_paired_region_tests(
                        sample_means=stats_input,
                        value_col="SampleMean",
                        subtype_order=subtype_order,
                        region_col="Region",
                        sample_col=None,
                        region_a="dermis",
                        region_b="wound",
                    )

                    if not stats_df.empty:

                        add_paired_region_annotations(
                            ax,
                            stats_df,
                            summary,
                            subtype_order=subtype_order,
                            regions=("dermis", "wound"),
                            region_a="dermis",
                            region_b="wound",
                        )

                    if print_stats and not stats_df.empty:

                        stats_print = stats_df.copy()

                        for c in [
                            "MeanA",
                            "MeanB",
                            "MeanDifference_B_minus_A",
                            "Normality_p",
                            "t_stat",
                            "statistic",
                            "p_value",
                        ]:
                            if c in stats_print.columns:
                                stats_print[c] = pd.to_numeric(
                                    stats_print[c],
                                    errors="coerce",
                                )

                        paired_cols = [
                            c for c in [
                                "Label",
                                "RegionA",
                                "RegionB",
                                "n_pairs",
                                "MeanA",
                                "MeanB",
                                "MeanDifference_B_minus_A",
                                "NormalityTest",
                                "Normality_p",
                                "NormalityInterpretation",
                                "SelectedTest",
                                "statistic",
                                "p_value",
                                "stars",
                            ]
                            if c in stats_print.columns
                        ]

                        print("\n[SAXS automatic paired tests: D-period FWHM dermis vs wound]")
                        print(
                            stats_print[paired_cols]
                            .to_string(index=False)
                        )

                # ------------------------------------------------------------
                # Case 2: only one region present, compare between subtypes
                # ------------------------------------------------------------
                else:

                    stats_df = auto_saxs_between_subtype_tests(
                        sample_means=stats_input,
                        value_col="SampleMean",
                        subtype_order=subtype_order,
                        region_col="Region",
                        sample_col=None,
                    )

                    if print_stats and not stats_df.empty:

                        stats_print = stats_df.copy()

                        for c in [
                            "statistic",
                            "p_value",
                        ]:
                            if c in stats_print.columns:
                                stats_print[c] = pd.to_numeric(
                                    stats_print[c],
                                    errors="coerce",
                                )

                        between_cols = [
                            c for c in [
                                "Region",
                                "Comparison",
                                "n_groups",
                                "GroupNs",
                                "NormalityTest",
                                "Normality_p_values",
                                "NormalityInterpretation",
                                "SelectedTest",
                                "statistic",
                                "p_value",
                                "stars",
                            ]
                            if c in stats_print.columns
                        ]

                        print("\n[SAXS automatic between-subtype tests: D-period FWHM]")
                        print(
                            stats_print[between_cols]
                            .to_string(index=False)
                        )

    plt.tight_layout()
    plt.show()

    return fig, ax, stats_df

def calculate_saxs_dperiod_shift_4region(
    saxs_points,
    parameter=SAXS_DPERIOD_PARAM,
    trim_std_devs=6,
):
    """
    Calculate D-period % shift:
        lower dermis -> lower wound
        upper dermis -> upper wound
    """

    sample_means, _, _ = prepare_saxs_region_summary_from_points(
        saxs_points=saxs_points,
        parameter=parameter,
        regions=SAXS_REGION_ORDER_4,
        pool_to_dermis_wound=False,
        trim_std_devs=trim_std_devs,
    )

    if sample_means.empty:
        return pd.DataFrame(), pd.DataFrame()

    wide = (
        sample_means
        .pivot_table(
            index=["experiment", "subtype", "SubtypeClean", "Filenumber"],
            columns="Region",
            values="SampleMean",
            aggfunc="first",
        )
        .reset_index()
    )

    rows = []

    shift_defs = {
        "lower_wound_vs_dermis": ("dermis_sub", "wound_sub"),
        "upper_wound_vs_dermis": ("dermis_epi", "wound_epi"),
    }

    for _, r in wide.iterrows():
        for shift_type, (base_region, target_region) in shift_defs.items():
            if base_region not in wide.columns or target_region not in wide.columns:
                continue

            base = r.get(base_region, np.nan)
            target = r.get(target_region, np.nan)

            if not np.isfinite(base) or base == 0 or not np.isfinite(target):
                continue

            rows.append({
                "experiment": r["experiment"],
                "subtype": r["subtype"],
                "SubtypeClean": r["SubtypeClean"],
                "Filenumber": r["Filenumber"],
                "ShiftType": shift_type,
                "BaselineRegion": base_region,
                "TargetRegion": target_region,
                "BaselineValue": base,
                "TargetValue": target,
                "PercentShift": ((target - base) / base) * 100,
            })

    shift_df = pd.DataFrame(rows)

    if shift_df.empty:
        return shift_df, pd.DataFrame()

    summary = (
        shift_df
        .groupby(["SubtypeClean", "ShiftType"], as_index=False)["PercentShift"]
        .agg(mean="mean", std="std", n="count")
    )

    return shift_df, summary

def plot_saxs_dperiod_shift_4region(
    saxs_points,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=None,
    trim_std_devs=6,
    hatches=True,
    print_summary=True,
):
    """
    D-period % shift:
        lower wound vs lower dermis
        upper wound vs upper dermis
    """

    subtype_order = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)

    shift_df, summary = calculate_saxs_dperiod_shift_4region(
        saxs_points=saxs_points,
        trim_std_devs=trim_std_devs,
    )

    if summary is None or summary.empty:
        print("No SAXS D-period 4-region shift data available.")
        return None, None

    if print_summary:
        print_saxs_shift_summary(
            summary,
            title="SAXS D-period 4-region shift bar means"
        )

    return _plot_saxs_shift_bars(
        shift_df=shift_df,
        summary=summary,
        shift_order=["lower_wound_vs_dermis", "upper_wound_vs_dermis"],
        title="D-period % shift from matched dermis to wound regions",
        ylabel="D-period shift from matched dermis (%)",
        subtype_order=subtype_order,
        ylim=ylim,
        hatches=hatches,
        shift_labels={
            "lower_wound_vs_dermis": "Lower wound vs lower dermis",
            "upper_wound_vs_dermis": "Upper wound vs upper dermis",
        },
    )

def print_saxs_shift_summary(summary, title="SAXS shift summary"):
    """
    Print mean, std, sem, n for SAXS shift bar plots.
    """

    if summary is None or summary.empty:
        print(f"\n{title}: no summary data.")
        return pd.DataFrame()

    s = summary.copy()

    print_cols = []

    for c in ["SubtypeClean", "Label", "ShiftType", "shift_type", "Comparison"]:
        if c in s.columns:
            print_cols.append(c)

    for c in ["mean", "std", "sem", "n", "error"]:
        if c in s.columns:
            print_cols.append(c)

    # fallback: print all columns if expected names differ
    if not print_cols:
        print_cols = list(s.columns)

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(s[print_cols].to_string(index=False))

    return s[print_cols]

def plot_saxs_dperiod_fwhm_4region(
    saxs_points,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=None,
    trim_std_devs=6,
    hatches=True,
    print_summary=True,
    run_stats=True,
    print_stats=True,
):
    """
    D-period FWHM split into:
        dermis_sub
        dermis_epi
        wound_sub
        wound_epi

    FWHM is calculated as:
        2.3548 * within-sample SD of raw D-period values

    Statistical testing is performed on sample-level FWHM values.

    Tests performed:
        1. Within each subtype:
               dermis_sub vs dermis_epi
               wound_sub  vs wound_epi, if present

           normal paired differences     -> paired t-test
           non-normal paired differences -> Wilcoxon signed-rank test

        2. Between subtypes within each region:
               dermis_sub: subtype A vs subtype B
               dermis_epi: subtype A vs subtype B
               wound_sub:  subtype A vs subtype B, if present
               wound_epi:  subtype A vs subtype B, if present

           two groups:
               normal groups     -> independent t-test
               non-normal groups -> Mann-Whitney U test

           more than two groups:
               all normal     -> one-way ANOVA
               any non-normal -> Kruskal-Wallis test
    """

    subtype_order = resolve_subtypes_to_plot(
        subtype_order,
        subtypes_to_plot,
    )

    # ------------------------------------------------------------
    # Prepare sample-level D-period FWHM values
    # ------------------------------------------------------------
    spread_samples, summary, value_col = prepare_saxs_dperiod_fwhm_from_spread(
        saxs_points=saxs_points,
        regions=SAXS_REGION_ORDER_4,
        pool_to_dermis_wound=False,
        trim_std_devs=trim_std_devs,
    )

    if summary.empty:
        print("No split-region SAXS D-period FWHM data available.")
        return None, None, pd.DataFrame()

    if print_summary:
        print_saxs_bar_summary(
            summary,
            title="SAXS D-period FWHM 4-region bar means",
        )

    # ------------------------------------------------------------
    # Plot grouped bars
    # ------------------------------------------------------------
    fig, ax = _plot_saxs_grouped_bars(
        summary=summary,
        points=spread_samples.rename(
            columns={
                "FWHM": value_col,
            }
        ),
        regions=SAXS_REGION_ORDER_4,
        title="D-period spread (FWHM) by spatial region",
        ylabel="Sample FWHM of D-period (nm)",
        region_col_points="Region",
        value_col=value_col,
        subtype_order=subtype_order,
        ylim=ylim,
        hatches=hatches,
        show=False,
    )

    # ------------------------------------------------------------
    # Automatic statistics
    # ------------------------------------------------------------
    stats_df = pd.DataFrame()

    if run_stats and fig is not None and ax is not None:

        if spread_samples is None or spread_samples.empty:

            print("\n[SAXS D-period FWHM 4-region stats] No spread_samples data available.")

        elif "Region" not in spread_samples.columns:

            print("\n[SAXS D-period FWHM 4-region stats] No Region column found in spread_samples.")
            print("Available columns:")
            print(spread_samples.columns.tolist())

        else:

            # ------------------------------------------------------------
            # Prepare stats input
            # ------------------------------------------------------------
            stats_input = spread_samples.copy()

            if "FWHM" in stats_input.columns:

                stats_input = stats_input.rename(
                    columns={
                        "FWHM": "SampleMean",
                    }
                )

            elif value_col in stats_input.columns:

                stats_input = stats_input.rename(
                    columns={
                        value_col: "SampleMean",
                    }
                )

            elif "SampleMean" not in stats_input.columns:

                print("\n[SAXS D-period FWHM 4-region stats] Could not find FWHM value column.")
                print("Expected one of: 'FWHM', value_col, or 'SampleMean'")
                print("value_col =", value_col)
                print("Available columns:")
                print(stats_input.columns.tolist())
                stats_input = pd.DataFrame()

            if stats_input is not None and not stats_input.empty:

                all_stats = []

                # ------------------------------------------------------------
                # 1. Within-subtype paired spatial tests
                # ------------------------------------------------------------
                paired_region_pairs = [
                    ("dermis_sub", "dermis_epi"),
                    ("wound_sub", "wound_epi"),
                ]

                paired_stats = auto_saxs_multiple_paired_region_tests(
                    sample_means=stats_input,
                    region_pairs=paired_region_pairs,
                    value_col="SampleMean",
                    subtype_order=subtype_order,
                    region_col="Region",
                    sample_col=None,
                )

                if not paired_stats.empty:

                    all_stats.append(paired_stats)

                    if print_stats:

                        paired_print = paired_stats.copy()

                        for c in [
                            "MeanA",
                            "MeanB",
                            "MeanDifference_B_minus_A",
                            "Normality_p",
                            "t_stat",
                            "statistic",
                            "p_value",
                        ]:
                            if c in paired_print.columns:
                                paired_print[c] = pd.to_numeric(
                                    paired_print[c],
                                    errors="coerce",
                                )

                        paired_cols = [
                            c for c in [
                                "ComparisonType",
                                "Label",
                                "RegionA",
                                "RegionB",
                                "n_pairs",
                                "MeanA",
                                "MeanB",
                                "MeanDifference_B_minus_A",
                                "NormalityTest",
                                "Normality_p",
                                "NormalityInterpretation",
                                "SelectedTest",
                                "statistic",
                                "p_value",
                                "stars",
                            ]
                            if c in paired_print.columns
                        ]

                        print("\n[SAXS automatic paired spatial tests: D-period FWHM lower/sub vs upper/epi regions]")
                        print(
                            paired_print[paired_cols]
                            .to_string(index=False)
                        )

                # ------------------------------------------------------------
                # 2. Between-subtype tests within each region
                # ------------------------------------------------------------
                between_stats = auto_saxs_between_subtype_tests(
                    sample_means=stats_input,
                    value_col="SampleMean",
                    subtype_order=subtype_order,
                    region_col="Region",
                    sample_col=None,
                )

                if not between_stats.empty:

                    between_stats["ComparisonType"] = "between-subtype test within region"
                    all_stats.append(between_stats)

                    if print_stats:

                        between_print = between_stats.copy()

                        for c in [
                            "statistic",
                            "p_value",
                        ]:
                            if c in between_print.columns:
                                between_print[c] = pd.to_numeric(
                                    between_print[c],
                                    errors="coerce",
                                )

                        between_cols = [
                            c for c in [
                                "ComparisonType",
                                "Region",
                                "Comparison",
                                "n_groups",
                                "GroupNs",
                                "NormalityTest",
                                "Normality_p_values",
                                "NormalityInterpretation",
                                "SelectedTest",
                                "statistic",
                                "p_value",
                                "stars",
                            ]
                            if c in between_print.columns
                        ]

                        print("\n[SAXS automatic between-subtype tests within each region: D-period FWHM]")
                        print(
                            between_print[between_cols]
                            .to_string(index=False)
                        )

                if all_stats:
                    stats_df = pd.concat(
                        all_stats,
                        ignore_index=True,
                        sort=False,
                    )

    plt.tight_layout()
    plt.show()

    return fig, ax, stats_df

def plot_saxs_wa_dermis_vs_wound(
    saxs_points,
    parameter=SAXS_WA_PARAM,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=None,
    trim_std_devs=6,
    hatches=True,
):
    """
    Average wa parameter in dermis vs wound.

    No normalisation.
    Uses point-level values -> sample means -> subtype mean ± SD.
    """

    subtype_order = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)

    sample_means, summary, value_col = prepare_saxs_region_summary_from_points(
        saxs_points=saxs_points,
        parameter=parameter,
        regions=["dermis", "wound"],
        pool_to_dermis_wound=True,
        trim_std_devs=trim_std_devs,
    )

    if summary.empty:
        print(f"No SAXS wa data available for parameter={parameter}")
        return None, None

    return _plot_saxs_grouped_bars(
        summary=summary,
        points=sample_means.rename(columns={"SampleMean": value_col}),
        regions=["dermis", "wound"],
        title="SAXS average wa parameter in dermis and wound",
        ylabel="wa parameter",
        region_col_points="Region",
        value_col=value_col,
        subtype_order=subtype_order,
        ylim=ylim,
        hatches=hatches,
    )

def plot_saxs_wa_4region(
    saxs_points,
    parameter=SAXS_WA_PARAM,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=None,
    trim_std_devs=6,
    hatches=True,
):
    """
    Average wa parameter split into:
        lower dermis
        upper dermis
        lower wound
        upper wound
    """

    subtype_order = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)

    sample_means, summary, value_col = prepare_saxs_region_summary_from_points(
        saxs_points=saxs_points,
        parameter=parameter,
        regions=SAXS_REGION_ORDER_4,
        pool_to_dermis_wound=False,
        trim_std_devs=trim_std_devs,
    )

    if summary.empty:
        print(f"No split-region SAXS wa data available for parameter={parameter}")
        return None, None

    return _plot_saxs_grouped_bars(
        summary=summary,
        points=sample_means.rename(columns={"SampleMean": value_col}),
        regions=SAXS_REGION_ORDER_4,
        title="SAXS average wa parameter by spatial region",
        ylabel="wa parameter",
        region_col_points="Region",
        value_col=value_col,
        subtype_order=subtype_order,
        ylim=ylim,
        hatches=hatches,
    )

def prepare_saxs_peak_position_points(
    saxs_points,
    peak_param="peak_position_folded",
    rsq_param="rsq",
    value_col="value",
    min_rsq=0.3,
    regions=None,
):
    """
    Prepare folded peak-position data from long-format SAXS point table.

    Converts:
        parameter | value

    into a wide table with:
        peak_position_folded
        rsq

    Keeps point-level x/y/Filenumber/region structure.
    """

    if saxs_points is None or saxs_points.empty:
        return pd.DataFrame()

    d = saxs_points.copy()
    d = add_clean_subtype(d)

    required = [
        "experiment",
        "subtype",
        "SubtypeClean",
        "Filenumber",
        "region",
        "x",
        "y",
        "parameter",
        value_col,
    ]

    missing = [c for c in required if c not in d.columns]
    if missing:
        raise KeyError(f"saxs_points missing columns: {missing}")

    d["parameter"] = d["parameter"].astype(str).str.strip()
    d["region"] = d["region"].astype(str).str.strip()
    d["subtype"] = d["subtype"].astype(str).str.strip()
    d["Filenumber"] = d["Filenumber"].astype(str).str.strip()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    if regions is not None:
        regions = [str(r).strip() for r in regions]
        d = d[d["region"].isin(regions)].copy()

    keep_params = [peak_param, rsq_param]

    d = d[d["parameter"].isin(keep_params)].copy()

    if d.empty:
        print(f"No SAXS peak-position rows found for {keep_params}")
        return pd.DataFrame()

    keys = [
        "experiment",
        "subtype",
        "SubtypeClean",
        "Filenumber",
        "region",
        "x",
        "y",
    ]

    wide = (
        d.pivot_table(
            index=keys,
            columns="parameter",
            values=value_col,
            aggfunc="first",
        )
        .reset_index()
    )

    wide.columns.name = None

    if peak_param not in wide.columns:
        print(f"Missing peak parameter after pivot: {peak_param}")
        return pd.DataFrame()

    if rsq_param not in wide.columns:
        wide[rsq_param] = np.nan

    wide[peak_param] = pd.to_numeric(wide[peak_param], errors="coerce")
    wide[rsq_param] = pd.to_numeric(wide[rsq_param], errors="coerce")

    wide = wide[np.isfinite(wide[peak_param])].copy()

    if min_rsq is not None:
        wide = wide[np.isfinite(wide[rsq_param]) & (wide[rsq_param] >= float(min_rsq))].copy()

    wide[peak_param] = np.mod(wide[peak_param], 180.0)

    return wide.reset_index(drop=True)

def plot_saxs_peak_position_circle_by_subtype(
    saxs_points,
    peak_param=SAXS_PEAK_POSITION_PARAM,
    rsq_param=SAXS_PEAK_RSQ_PARAM,
    min_rsq=0.3,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    subtype_colors=NI_COLOURS,
    labels=NI_LABELS,
    dermis_region="dermis",
    wound_region="wound",
    ncols=3,
    max_rays_per_region=900,
    ray_alpha=0.22,
    ray_width=0.65,
    circle_radius=1.0,
):
    """
    Circle/ray plot of folded peak position by subtype.

    Top half = dermis
    Bottom half = wound
    """

    subtype_order = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)

    regions_needed = SAXS_REGION_ORDER_4 + ["dermis", "wound"]

    d = prepare_saxs_peak_position_points(
        saxs_points=saxs_points,
        peak_param=peak_param,
        rsq_param=rsq_param,
        min_rsq=min_rsq,
        regions=regions_needed,
    )

    if d.empty:
        print("No folded peak-position data to plot.")
        return None, None

    d = pool_saxs_dermis_wound_regions(d)
    d = d.rename(columns={"RegionPooled": "RegionForPlot"})
    d = d[d["RegionForPlot"].isin([dermis_region, wound_region])].copy()

    # print("\n[Peak-position circle plot ray counts]")
    # print(f"peak_param = {peak_param}")
    # print(f"rsq_param  = {rsq_param}")
    # print(f"min_rsq    = {min_rsq}")

    count_rows = []

    for subtype in subtype_order:
        d_sub = d[d["SubtypeClean"].astype(str) == str(subtype)].copy()

        if d_sub.empty:
            continue

        for region_name in [dermis_region, wound_region]:
            d_reg = d_sub[d_sub["RegionForPlot"] == region_name].copy()
            vals = pd.to_numeric(d_reg[peak_param], errors="coerce")

            count_rows.append({
                "Subtype": subtype,
                "Region": region_name,
                "N_samples": d_reg["Filenumber"].nunique() if "Filenumber" in d_reg.columns else np.nan,
                "N_rows": len(d_reg),
                "N_finite_peak_values_before_downsample": int(np.isfinite(vals).sum()),
                "N_rays_plotted_after_downsample": min(int(np.isfinite(vals).sum()), int(max_rays_per_region)),
            })

    count_df = pd.DataFrame(count_rows)

    if count_df.empty:
        print("No peak-position rows found after filtering/pooling.")
    # else:
    #     print(count_df.to_string(index=False))

    if d.empty:
        print("No dermis/wound peak-position rows after pooling.")
        return None, None

    present_subtypes = [
        st for st in subtype_order
        if st in set(d["SubtypeClean"].astype(str))
    ]

    if not present_subtypes:
        print("No matching subtypes to plot.")
        return None, None

    def _angle_to_xy(angle_deg):
        theta = np.deg2rad(90.0 - angle_deg)
        return circle_radius * np.cos(theta), circle_radius * np.sin(theta)

    def _draw_ray(ax, angle_deg, color):
        x, y = _angle_to_xy(float(angle_deg))
        ax.plot(
            [0, x],
            [0, y],
            color=color,
            alpha=ray_alpha,
            lw=ray_width,
            zorder=3,
        )

    def _downsample(vals, max_n):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]

        if len(vals) <= max_n:
            return vals

        idx = np.linspace(0, len(vals) - 1, max_n, dtype=int)
        return vals[idx]

    def _map_dermis_top(vals):
        vals = np.mod(np.asarray(vals, dtype=float), 180.0)
        vals = vals[np.isfinite(vals)]

        out = vals - 90.0
        out = np.mod(out, 360.0)

        out = np.where((out >= 60.0) & (out <= 120.0), out - 90.0, out)
        out = np.where((out >= 240.0) & (out <= 300.0), out + 90.0, out)

        return np.mod(out, 360.0)

    def _map_wound_bottom(vals):
        vals = np.mod(np.asarray(vals, dtype=float), 180.0)
        vals = vals[np.isfinite(vals)]

        top = vals - 90.0
        top = np.mod(top, 360.0)

        top = np.where((top >= 60.0) & (top <= 120.0), top - 90.0, top)
        top = np.where((top >= 240.0) & (top <= 300.0), top + 90.0, top)
        top = np.mod(top, 360.0)

        return np.mod(top + 180.0, 360.0)

    nrows = int(np.ceil(len(present_subtypes) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.4 * ncols, 4.6 * nrows),
    )

    axes = np.atleast_1d(axes).ravel()

    for i, subtype in enumerate(present_subtypes):
        ax = axes[i]

        d_sub = d[d["SubtypeClean"].astype(str) == subtype].copy()
        base_color = subtype_colors.get(subtype, "grey")

        top = Wedge(
            center=(0, 0),
            r=circle_radius,
            theta1=0,
            theta2=180,
            facecolor=base_color,
            edgecolor="none",
            alpha=0.18,
            zorder=0,
        )

        bottom = Wedge(
            center=(0, 0),
            r=circle_radius,
            theta1=180,
            theta2=360,
            facecolor="white",
            edgecolor="none",
            alpha=1.0,
            zorder=0,
        )

        ax.add_patch(top)
        ax.add_patch(bottom)

        circ = plt.Circle(
            (0, 0),
            circle_radius,
            fill=False,
            color="black",
            lw=1.0,
            zorder=2,
        )
        ax.add_patch(circ)

        ax.plot([0, 0], [-circle_radius, circle_radius], color="black", lw=0.7, alpha=0.4)
        ax.plot([-circle_radius, circle_radius], [0, 0], color="black", lw=0.7, alpha=0.4)

        derm = d_sub.loc[d_sub["RegionForPlot"] == dermis_region, peak_param].to_numpy(float)
        wound = d_sub.loc[d_sub["RegionForPlot"] == wound_region, peak_param].to_numpy(float)

        derm = _downsample(_map_dermis_top(derm), max_rays_per_region)
        wound = _downsample(_map_wound_bottom(wound), max_rays_per_region)

        for angle in derm:
            _draw_ray(ax, angle, "grey")

        for angle in wound:
            _draw_ray(ax, angle, base_color)

        ax.set_title(labels.get(subtype, subtype))
        ax.set_aspect("equal")
        ax.set_xlim(-1.15 * circle_radius, 1.15 * circle_radius)
        ax.set_ylim(-1.15 * circle_radius, 1.15 * circle_radius)
        ax.axis("off")

    for ax in axes[len(present_subtypes):]:
        ax.axis("off")

    handles = [
        Patch(facecolor="lightgrey", edgecolor="black", label="Dermis"),
        Patch(facecolor="dimgray", edgecolor="black", label="Wound"),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
    )

    fig.suptitle(
        f"Folded peak-position ray plots by subtype (rsq ≥ {min_rsq:g})",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.show()

    return fig, axes

def plot_saxs_peak_position_bar(
    saxs_points,
    peak_param=SAXS_PEAK_POSITION_PARAM,
    rsq_param=SAXS_PEAK_RSQ_PARAM,
    min_rsq=0.3,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    subtype_colors=NI_COLOURS,
    labels=NI_LABELS,
    regions=("dermis", "wound"),
    ylim=(0, 180),
    hatches=True,
    print_summary=True,
    run_stats=True,
    print_stats=True,
):
    """
    Bar chart of folded peak position by subtype and region.

    Uses point-level folded peak positions:
        point values -> sample-level means -> subtype mean ± SD

    Statistical testing is performed on sample-level means.

    Dynamic statistics:
        1. If both dermis and wound are present:
               paired dermis vs wound test within each subtype

               normal paired differences     -> paired t-test
               non-normal paired differences -> Wilcoxon signed-rank test

        2. If only one region is present:
               compare between subtypes within that region

               two groups:
                   normal groups     -> independent t-test
                   non-normal groups -> Mann-Whitney U test

               more than two groups:
                   all normal     -> one-way ANOVA
                   any non-normal -> Kruskal-Wallis test
    """

    subtype_order = resolve_subtypes_to_plot(
        subtype_order,
        subtypes_to_plot,
    )

    # ------------------------------------------------------------
    # Prepare folded peak-position point-level data
    # ------------------------------------------------------------
    d = prepare_saxs_peak_position_points(
        saxs_points=saxs_points,
        peak_param=peak_param,
        rsq_param=rsq_param,
        min_rsq=min_rsq,
        regions=SAXS_REGION_ORDER_4 + ["dermis", "wound"],
    )

    if d.empty:
        print("No folded peak-position data to plot.")
        return None, None, pd.DataFrame()

    # ------------------------------------------------------------
    # Pool into dermis / wound
    # ------------------------------------------------------------
    d = pool_saxs_dermis_wound_regions(d)
    d = d.rename(
        columns={
            "RegionPooled": "RegionForPlot",
        }
    )

    d = d[
        d["RegionForPlot"].isin(regions)
    ].copy()

    if d.empty:
        print("No dermis/wound folded peak-position data.")
        return None, None, pd.DataFrame()

    # ------------------------------------------------------------
    # Convert point-level values into sample-level means
    # ------------------------------------------------------------
    sample_means = (
        d.groupby(
            [
                "experiment",
                "subtype",
                "SubtypeClean",
                "Filenumber",
                "RegionForPlot",
            ],
            observed=True,
            as_index=False,
        )[peak_param]
        .mean()
        .rename(
            columns={
                peak_param: "SampleMean",
                "RegionForPlot": "Region",
            }
        )
    )

    summary = (
        sample_means
        .groupby(
            [
                "SubtypeClean",
                "Region",
            ],
            observed=True,
        )["SampleMean"]
        .agg(
            mean="mean",
            std="std",
            n="count",
        )
        .reset_index()
    )

    summary["sem"] = summary["std"] / np.sqrt(summary["n"])
    summary["error"] = summary["std"]

    if print_summary:
        print_saxs_bar_summary(
            summary,
            title="SAXS folded peak-position bar means",
        )

    # ------------------------------------------------------------
    # Plot grouped bars
    # ------------------------------------------------------------
    fig, ax = _plot_saxs_grouped_bars(
        summary=summary,
        points=sample_means.rename(
            columns={
                "SampleMean": peak_param,
            }
        ),
        regions=list(regions),
        title="Folded peak position by subtype",
        ylabel="Folded peak position (°)",
        region_col_points="Region",
        value_col=peak_param,
        subtype_order=subtype_order,
        colours=subtype_colors,
        labels=labels,
        ylim=ylim,
        hatches=hatches,
        show=False,
    )

    # ------------------------------------------------------------
    # Automatic statistics
    # ------------------------------------------------------------
    stats_df = pd.DataFrame()

    if run_stats and fig is not None and ax is not None:

        if "Region" not in sample_means.columns:

            print("\n[SAXS folded peak-position stats] No Region column found in sample_means.")
            print("Available columns:")
            print(sample_means.columns.tolist())

        else:

            regions_present = set(
                sample_means["Region"]
                .astype(str)
                .str.lower()
                .str.strip()
                .unique()
            )

            requested_regions = set(
                str(r).lower().strip()
                for r in regions
            )

            # ------------------------------------------------------------
            # Case 1: paired dermis vs wound within each subtype
            # ------------------------------------------------------------
            if (
                {"dermis", "wound"}.issubset(regions_present)
                and {"dermis", "wound"}.issubset(requested_regions)
            ):

                stats_df = auto_saxs_paired_region_tests(
                    sample_means=sample_means,
                    value_col="SampleMean",
                    subtype_order=subtype_order,
                    region_col="Region",
                    sample_col=None,
                    region_a="dermis",
                    region_b="wound",
                )

                if not stats_df.empty:

                    add_paired_region_annotations(
                        ax,
                        stats_df,
                        summary,
                        subtype_order=subtype_order,
                        regions=("dermis", "wound"),
                        region_a="dermis",
                        region_b="wound",
                    )

                if print_stats and not stats_df.empty:

                    stats_print = stats_df.copy()

                    for c in [
                        "MeanA",
                        "MeanB",
                        "MeanDifference_B_minus_A",
                        "Normality_p",
                        "t_stat",
                        "statistic",
                        "p_value",
                    ]:
                        if c in stats_print.columns:
                            stats_print[c] = pd.to_numeric(
                                stats_print[c],
                                errors="coerce",
                            )

                    paired_cols = [
                        c for c in [
                            "Label",
                            "RegionA",
                            "RegionB",
                            "n_pairs",
                            "MeanA",
                            "MeanB",
                            "MeanDifference_B_minus_A",
                            "NormalityTest",
                            "Normality_p",
                            "NormalityInterpretation",
                            "SelectedTest",
                            "statistic",
                            "p_value",
                            "stars",
                        ]
                        if c in stats_print.columns
                    ]

                    print("\n[SAXS automatic paired tests: folded peak position dermis vs wound]")
                    print(
                        stats_print[paired_cols]
                        .to_string(index=False)
                    )

            # ------------------------------------------------------------
            # Case 2: only one region present, or non-paired region structure
            # ------------------------------------------------------------
            else:

                stats_df = auto_saxs_between_subtype_tests(
                    sample_means=sample_means,
                    value_col="SampleMean",
                    subtype_order=subtype_order,
                    region_col="Region",
                    sample_col=None,
                )

                if print_stats and not stats_df.empty:

                    stats_print = stats_df.copy()

                    for c in [
                        "statistic",
                        "p_value",
                    ]:
                        if c in stats_print.columns:
                            stats_print[c] = pd.to_numeric(
                                stats_print[c],
                                errors="coerce",
                            )

                    between_cols = [
                        c for c in [
                            "Region",
                            "Comparison",
                            "n_groups",
                            "GroupNs",
                            "NormalityTest",
                            "Normality_p_values",
                            "NormalityInterpretation",
                            "SelectedTest",
                            "statistic",
                            "p_value",
                            "stars",
                        ]
                        if c in stats_print.columns
                    ]

                    print("\n[SAXS automatic between-subtype tests: folded peak position]")
                    print(
                        stats_print[between_cols]
                        .to_string(index=False)
                    )

    plt.tight_layout()
    plt.show()

    return fig, ax, stats_df

def calculate_saxs_peak_position_shift_from_dermis(
    saxs_points,
    peak_param=SAXS_PEAK_POSITION_PARAM,
    rsq_param=SAXS_PEAK_RSQ_PARAM,
    min_rsq=0.3,
):
    """
    Calculate folded peak-position % shift from dermis to wound.

    Uses:
        point-level folded peak position
        -> sample mean dermis/wound
        -> % shift per sample
    """

    d = prepare_saxs_peak_position_points(
        saxs_points=saxs_points,
        peak_param=peak_param,
        rsq_param=rsq_param,
        min_rsq=min_rsq,
        regions=SAXS_REGION_ORDER_4 + ["dermis", "wound"],
    )

    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    d = pool_saxs_dermis_wound_regions(d)
    d = d.rename(columns={"RegionPooled": "RegionForPlot"})

    d = d[d["RegionForPlot"].isin(["dermis", "wound"])].copy()

    sample_means = (
        d.groupby(
            ["experiment", "subtype", "SubtypeClean", "Filenumber", "RegionForPlot"],
            observed=True,
            as_index=False,
        )[peak_param]
        .mean()
        .rename(columns={peak_param: "SampleMean", "RegionForPlot": "Region"})
    )

    wide = (
        sample_means
        .pivot_table(
            index=["experiment", "subtype", "SubtypeClean", "Filenumber"],
            columns="Region",
            values="SampleMean",
            aggfunc="first",
        )
        .reset_index()
    )

    if "dermis" not in wide.columns or "wound" not in wide.columns:
        return pd.DataFrame(), pd.DataFrame()

    ok = (
        np.isfinite(wide["dermis"])
        & (wide["dermis"] != 0)
        & np.isfinite(wide["wound"])
    )

    shift_df = wide.loc[ok].copy()
    shift_df["PercentShift"] = ((shift_df["wound"] - shift_df["dermis"]) / shift_df["dermis"]) * 100
    shift_df["ShiftType"] = "wound_vs_dermis"

    summary = (
        shift_df
        .groupby(["SubtypeClean", "ShiftType"], as_index=False)["PercentShift"]
        .agg(mean="mean", std="std", n="count")
    )

    return shift_df, summary

def plot_saxs_peak_position_shift_from_dermis(
    saxs_points,
    peak_param=SAXS_PEAK_POSITION_PARAM,
    rsq_param=SAXS_PEAK_RSQ_PARAM,
    min_rsq=0.3,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=None,
    hatches=True,
):
    """
    Bar chart of folded peak-position % shift from dermis to wound.
    """

    subtype_order = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)

    shift_df, summary = calculate_saxs_peak_position_shift_from_dermis(
        saxs_points=saxs_points,
        peak_param=peak_param,
        rsq_param=rsq_param,
        min_rsq=min_rsq,
    )

    return _plot_saxs_shift_bars(
        shift_df=shift_df,
        summary=summary,
        shift_order=["wound_vs_dermis"],
        title="Folded peak-position % shift from dermis to wound",
        ylabel="Folded peak-position shift from dermis (%)",
        subtype_order=subtype_order,
        ylim=ylim,
        hatches=hatches,
        shift_labels={
            "wound_vs_dermis": "Wound vs dermis",
        },
    )

def _saxs_peak_spread_from_angles(
    angles,
    *,
    centre_deg,
    arc_width=180.0,
    gap_thresh_deg=20.0,
):
    """
    Calculate angular spread on a 180° semicircle using the gap-threshold method.

    Steps:
        1. Map angles into a continuous semicircle centred on centre_deg.
        2. Sort angles.
        3. Calculate neighbouring gaps, including the wrap gap.
        4. Ignore/suppress empty gaps larger than gap_thresh_deg.
        5. Spread = 180 - sum(large gaps)

    This matches the logic used in the original SAXS code:
        - captures the occupied angular range
        - avoids artificially inflating spread due to large empty gaps
    """

    vals = np.asarray(angles, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return np.nan

    if vals.size == 1:
        return 0.0

    half_width = arc_width / 2.0

    # Convert to coordinates centred around centre_deg.
    # Result lies approximately in [-90, +90] around the chosen axis.
    centred = ((vals - centre_deg + half_width) % arc_width) - half_width

    # Shift to 0–180 for easier gap calculation.
    vals180 = np.sort(centred + half_width)

    gaps = np.diff(vals180)
    wrap_gap = (vals180[0] + arc_width) - vals180[-1]

    all_gaps = np.concatenate([gaps, [wrap_gap]])
    big_gaps = all_gaps[all_gaps > gap_thresh_deg]

    spread = float(arc_width - np.sum(big_gaps))
    spread = max(0.0, min(float(arc_width), spread))

    return spread

def _map_folded_to_dermis_top(vals):
    """
    Map folded 0–180 peak positions to the dermis/top semicircle.

    Result clusters around 0 degrees:
        0–90   -> 0–90
        90–180 -> 270–360
    """

    vals = np.mod(np.asarray(vals, dtype=float), 180.0)
    vals = vals[np.isfinite(vals)]

    mapped = np.where(vals > 90.0, vals - 180.0, vals)
    mapped = np.mod(mapped, 360.0)

    return mapped

def _map_folded_to_wound_bottom(vals):
    """
    Map folded 0–180 peak positions to the wound/bottom semicircle.

    Result clusters around 180 degrees:
        dermis/top mapped values + 180
    """

    return np.mod(_map_folded_to_dermis_top(vals) + 180.0, 360.0)

def _opposite_rays(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return np.mod(vals + 180.0, 360.0)

def _unwrap_for_interval(vals, start, end):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return vals

    if start <= end:
        out = vals[(vals >= start) & (vals <= end)]
    else:
        out = vals.copy()
        out[out < start] += 360.0
        out = out[(out >= start) & (out <= end + 360.0)]

    return np.sort(out)

def _spread_gap_threshold_oldstyle(vals, arc_width=180.0, gap_thresh_deg=20.0):
    """
    Old SAXS-style angular spread.

    Spread = 180 - sum(gaps larger than gap_thresh_deg)
    """

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return np.nan

    if vals.size == 1:
        return 0.0

    vals = np.sort(vals)

    gaps = np.diff(vals)
    wrap_gap = (vals[0] + arc_width) - vals[-1]

    all_gaps = np.concatenate([gaps, [wrap_gap]])
    big_gaps = all_gaps[all_gaps > gap_thresh_deg]

    spread = float(arc_width - np.sum(big_gaps))
    spread = max(0.0, min(float(arc_width), spread))

    return spread

def _collect_peak_values(df, peak_param=SAXS_PEAK_SPREAD_PARAM):
    vals = pd.to_numeric(df[peak_param], errors="coerce").to_numpy(float)
    vals = vals[np.isfinite(vals)]
    return vals

def calculate_saxs_peak_spread_samples(
    saxs_points,
    peak_param=SAXS_PEAK_SPREAD_PARAM,
    rsq_param=SAXS_PEAK_SPREAD_RSQ_PARAM,
    min_rsq=0.3,
    region_mode="pooled",
    gap_thresh_deg=20.0,
    max_points_per_sample_region=300,
):
    """
    Calculate peak-position spread per sample and region.

    Parameters
    ----------
    region_mode:
        "pooled"
            dermis_sub + dermis_epi -> dermis
            wound_sub + wound_epi -> wound

        "split"
            keep dermis_sub, dermis_epi, wound_sub, wound_epi

    Returns
    -------
    spread_samples : pd.DataFrame
        One row per sample/region.

    spread_summary : pd.DataFrame
        Mean ± SD across samples per subtype/region.
    """

    if region_mode not in {"pooled", "split"}:
        raise ValueError("region_mode must be 'pooled' or 'split'.")

    regions = SAXS_REGION_ORDER_4 + ["dermis", "wound"]

    d = prepare_saxs_peak_position_points(
        saxs_points=saxs_points,
        peak_param=peak_param,
        rsq_param=rsq_param,
        min_rsq=min_rsq,
        regions=regions,
    )

    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    if region_mode == "pooled":
        d = pool_saxs_dermis_wound_regions(d)
        d = d.rename(columns={"RegionPooled": "RegionForSpread"})
        region_order = ["dermis", "wound"]
    else:
        d = d[d["region"].isin(SAXS_REGION_ORDER_4)].copy()
        d = d.rename(columns={"region": "RegionForSpread"})
        region_order = SAXS_REGION_ORDER_4

    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []

    group_cols = [
        "experiment",
        "subtype",
        "SubtypeClean",
        "Filenumber",
        "RegionForSpread",
    ]

    for keys, g in d.groupby(group_cols, observed=True):
        experiment, subtype, subtype_clean, filenumber, region = keys

        vals = _collect_peak_values(g, peak_param=peak_param)

        if max_points_per_sample_region is not None and len(vals) > max_points_per_sample_region:
            idx = np.linspace(0, len(vals) - 1, int(max_points_per_sample_region), dtype=int)
            vals = vals[idx]

        if str(region).startswith("dermis"):
            mapped = _map_folded_to_dermis_top(vals)
            centre = 0.0
        
        elif str(region).startswith("wound"):
            mapped = _map_folded_to_wound_bottom(vals)
            centre = 180.0
        
        else:
            mapped = vals
            centre = 0.0
        
        spread = _saxs_peak_spread_from_angles(
            mapped,
            centre_deg=centre,
            arc_width=180.0,
            gap_thresh_deg=gap_thresh_deg,
        )

        rows.append({
            "experiment": experiment,
            "subtype": subtype,
            "SubtypeClean": subtype_clean,
            "Filenumber": filenumber,
            "Region": region,
            "SpreadDegrees": spread,
            "N_points": int(len(vals)),
        })

    spread_samples = pd.DataFrame(rows)

    if spread_samples.empty:
        return spread_samples, pd.DataFrame()

    spread_samples["Region"] = pd.Categorical(
        spread_samples["Region"],
        categories=region_order,
        ordered=True,
    )

    spread_summary = (
        spread_samples
        .groupby(["SubtypeClean", "Region"], observed=True)["SpreadDegrees"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )

    return spread_samples, spread_summary

def _plot_saxs_peak_spread_grouped_bars(
    spread_samples,
    spread_summary,
    regions,
    title,
    ylabel="Spread (degrees)",
    subtype_order=SUBTYPE_ORDER,
    colours=NI_COLOURS,
    labels=NI_LABELS,
    region_labels=SAXS_REGION_LABELS,
    ylim=(0, 180),
    jitter=0.045,
    point_size=24,
    alpha_range=(0.95, 0.30),
    hatches=True,
    region_hatches=None,
    hatch_linewidth=0.7,
):
    """
    Generic grouped bar chart for peak-position spread.
    """

    if spread_summary is None or spread_summary.empty:
        print("No peak-position spread summary to plot.")
        return None, None

    points = spread_samples.rename(columns={"SpreadDegrees": "value"})
    summary = spread_summary.rename(columns={"SpreadDegrees": "value"})

    return _plot_saxs_grouped_bars(
        summary=summary,
        points=points,
        regions=regions,
        title=title,
        ylabel=ylabel,
        region_col_points="Region",
        value_col="value",
        subtype_order=subtype_order,
        colours=colours,
        labels=labels,
        region_labels=region_labels,
        ylim=ylim,
        scatter=True,
        jitter=jitter,
        point_size=point_size,
        alpha_range=alpha_range,
        hatches=hatches,
        region_hatches=region_hatches,
        hatch_linewidth=hatch_linewidth,
    )

def plot_saxs_peak_spread_dermis_vs_wound(
    saxs_points,
    peak_param=SAXS_PEAK_SPREAD_PARAM,
    rsq_param=SAXS_PEAK_SPREAD_RSQ_PARAM,
    min_rsq=0.3,
    gap_thresh_deg=20.0,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=(0, 90),
    hatches=True,
    print_summary=True,
    run_stats=True,
    print_stats=True,
):
    """
    Spread (degrees) by subtype in dermis and wound.

    Uses point-level SAXS peak-position data to calculate sample-level
    peak-position spread for each sample/region.

    Statistical testing is performed on sample-level spread values.

    Dynamic statistics:
        1. If both dermis and wound are present:
               paired dermis vs wound test within each subtype

               normal paired differences     -> paired t-test
               non-normal paired differences -> Wilcoxon signed-rank test

        2. If only one region is present:
               compare between subtypes within that region

               two groups:
                   normal groups     -> independent t-test
                   non-normal groups -> Mann-Whitney U test

               more than two groups:
                   all normal     -> one-way ANOVA
                   any non-normal -> Kruskal-Wallis test
    """

    subtype_order = resolve_subtypes_to_plot(
        subtype_order,
        subtypes_to_plot,
    )

    # ------------------------------------------------------------
    # Calculate sample-level peak-position spread
    # ------------------------------------------------------------
    spread_samples, spread_summary = calculate_saxs_peak_spread_samples(
        saxs_points=saxs_points,
        peak_param=peak_param,
        rsq_param=rsq_param,
        min_rsq=min_rsq,
        region_mode="pooled",
        gap_thresh_deg=gap_thresh_deg,
    )

    if spread_summary is None or spread_summary.empty:
        print("No SAXS peak-spread dermis/wound data available.")
        return None, None, pd.DataFrame()

    # Make sure print_saxs_bar_summary can print the usual fields
    if "sem" not in spread_summary.columns and {"std", "n"}.issubset(spread_summary.columns):
        spread_summary = spread_summary.copy()
        spread_summary["sem"] = spread_summary["std"] / np.sqrt(spread_summary["n"])

    if "error" not in spread_summary.columns and "std" in spread_summary.columns:
        spread_summary = spread_summary.copy()
        spread_summary["error"] = spread_summary["std"]

    if print_summary:
        print_saxs_bar_summary(
            spread_summary,
            title="SAXS peak-position spread dermis vs wound bar means",
        )

    # ------------------------------------------------------------
    # Plot grouped bars
    # ------------------------------------------------------------
    fig, ax = _plot_saxs_peak_spread_grouped_bars(
        spread_samples=spread_samples,
        spread_summary=spread_summary,
        regions=["dermis", "wound"],
        title="Peak-position spread in dermis and wound",
        ylabel="Spread (degrees)",
        subtype_order=subtype_order,
        ylim=ylim,
        hatches=hatches,
    )

    # ------------------------------------------------------------
    # Automatic statistics
    # ------------------------------------------------------------
    stats_df = pd.DataFrame()

    if run_stats and spread_samples is not None and not spread_samples.empty:

        stats_input = spread_samples.copy()

        possible_value_cols = [
            "SpreadDegrees",
            "Spread",
            "spread",
            "PeakSpread",
            "peak_spread",
            "SampleMean",
            peak_param,
        ]

        value_col = None

        for c in possible_value_cols:
            if c in stats_input.columns:
                value_col = c
                break

        if value_col is None:

            print("\n[SAXS peak-spread stats] Could not find spread value column.")
            print("Available columns:")
            print(stats_input.columns.tolist())

        elif "Region" not in stats_input.columns:

            print("\n[SAXS peak-spread stats] No Region column found in spread_samples.")
            print("Available columns:")
            print(stats_input.columns.tolist())

        else:

            stats_input = stats_input.rename(
                columns={
                    value_col: "SampleMean",
                }
            )

            regions_present = set(
                stats_input["Region"]
                .astype(str)
                .str.lower()
                .str.strip()
                .unique()
            )

            # ------------------------------------------------------------
            # Case 1: paired dermis vs wound within each subtype
            # ------------------------------------------------------------
            if {"dermis", "wound"}.issubset(regions_present):

                stats_df = auto_saxs_paired_region_tests(
                    sample_means=stats_input,
                    value_col="SampleMean",
                    subtype_order=subtype_order,
                    region_col="Region",
                    sample_col=None,
                    region_a="dermis",
                    region_b="wound",
                )

                if print_stats and not stats_df.empty:

                    stats_print = stats_df.copy()

                    for c in [
                        "MeanA",
                        "MeanB",
                        "MeanDifference_B_minus_A",
                        "Normality_p",
                        "t_stat",
                        "statistic",
                        "p_value",
                    ]:
                        if c in stats_print.columns:
                            stats_print[c] = pd.to_numeric(
                                stats_print[c],
                                errors="coerce",
                            )

                    paired_cols = [
                        c for c in [
                            "Label",
                            "RegionA",
                            "RegionB",
                            "n_pairs",
                            "MeanA",
                            "MeanB",
                            "MeanDifference_B_minus_A",
                            "NormalityTest",
                            "Normality_p",
                            "NormalityInterpretation",
                            "SelectedTest",
                            "statistic",
                            "p_value",
                            "stars",
                        ]
                        if c in stats_print.columns
                    ]

                    print("\n[SAXS automatic paired tests: peak-position spread dermis vs wound]")
                    print(
                        stats_print[paired_cols]
                        .to_string(index=False)
                    )

            # ------------------------------------------------------------
            # Case 2: only one region present, compare between subtypes
            # ------------------------------------------------------------
            else:

                stats_df = auto_saxs_between_subtype_tests(
                    sample_means=stats_input,
                    value_col="SampleMean",
                    subtype_order=subtype_order,
                    region_col="Region",
                    sample_col=None,
                )

                if print_stats and not stats_df.empty:

                    stats_print = stats_df.copy()

                    for c in [
                        "statistic",
                        "p_value",
                    ]:
                        if c in stats_print.columns:
                            stats_print[c] = pd.to_numeric(
                                stats_print[c],
                                errors="coerce",
                            )

                    between_cols = [
                        c for c in [
                            "Region",
                            "Comparison",
                            "n_groups",
                            "GroupNs",
                            "NormalityTest",
                            "Normality_p_values",
                            "NormalityInterpretation",
                            "SelectedTest",
                            "statistic",
                            "p_value",
                            "stars",
                        ]
                        if c in stats_print.columns
                    ]

                    print("\n[SAXS automatic between-subtype tests: peak-position spread]")
                    print(
                        stats_print[between_cols]
                        .to_string(index=False)
                    )

    return fig, ax, stats_df

def plot_saxs_peak_spread_4region(
    saxs_points,
    peak_param=SAXS_PEAK_SPREAD_PARAM,
    rsq_param=SAXS_PEAK_SPREAD_RSQ_PARAM,
    min_rsq=0.3,
    gap_thresh_deg=20.0,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=(0, 90),
    hatches=True,
    print_summary=True,
    run_stats=True,
    print_stats=True,
):
    """
    Spread (degrees) by subtype in:
        dermis_sub
        dermis_epi
        wound_sub
        wound_epi

    Uses point-level SAXS peak-position data to calculate sample-level
    peak-position spread for each sample/region.

    Statistical testing is performed on sample-level spread values.

    Tests performed:
        1. Within each subtype:
               dermis_sub vs dermis_epi
               wound_sub  vs wound_epi, if present

           normal paired differences     -> paired t-test
           non-normal paired differences -> Wilcoxon signed-rank test

        2. Between subtypes within each region:
               dermis_sub: subtype A vs subtype B
               dermis_epi: subtype A vs subtype B
               wound_sub:  subtype A vs subtype B, if present
               wound_epi:  subtype A vs subtype B, if present

           two groups:
               normal groups     -> independent t-test
               non-normal groups -> Mann-Whitney U test

           more than two groups:
               all normal     -> one-way ANOVA
               any non-normal -> Kruskal-Wallis test
    """

    subtype_order = resolve_subtypes_to_plot(
        subtype_order,
        subtypes_to_plot,
    )

    # ------------------------------------------------------------
    # Calculate sample-level peak-position spread
    # ------------------------------------------------------------
    spread_samples, spread_summary = calculate_saxs_peak_spread_samples(
        saxs_points=saxs_points,
        peak_param=peak_param,
        rsq_param=rsq_param,
        min_rsq=min_rsq,
        region_mode="split",
        gap_thresh_deg=gap_thresh_deg,
    )

    if spread_summary is None or spread_summary.empty:
        print("No SAXS peak-spread data available.")
        return None, None, pd.DataFrame()

    # Make sure print_saxs_bar_summary can print the usual fields
    if "sem" not in spread_summary.columns and {"std", "n"}.issubset(spread_summary.columns):
        spread_summary = spread_summary.copy()
        spread_summary["sem"] = spread_summary["std"] / np.sqrt(spread_summary["n"])

    if "error" not in spread_summary.columns and "std" in spread_summary.columns:
        spread_summary = spread_summary.copy()
        spread_summary["error"] = spread_summary["std"]

    if print_summary:
        print_saxs_bar_summary(
            spread_summary,
            title="SAXS peak-position spread 4-region bar means",
        )

    # ------------------------------------------------------------
    # Plot grouped bars
    # ------------------------------------------------------------
    fig, ax = _plot_saxs_peak_spread_grouped_bars(
        spread_samples=spread_samples,
        spread_summary=spread_summary,
        regions=SAXS_REGION_ORDER_4,
        title="Peak-position spread by spatial region",
        ylabel="Spread (degrees)",
        subtype_order=subtype_order,
        ylim=ylim,
        hatches=hatches,
    )

    # ------------------------------------------------------------
    # Automatic statistics
    # ------------------------------------------------------------
    stats_df = pd.DataFrame()

    if run_stats and spread_samples is not None and not spread_samples.empty:

        stats_input = spread_samples.copy()

        possible_value_cols = [
            "SpreadDegrees",
            "Spread",
            "spread",
            "PeakSpread",
            "peak_spread",
            "SampleMean",
            peak_param,
        ]

        value_col = None

        for c in possible_value_cols:
            if c in stats_input.columns:
                value_col = c
                break

        if value_col is None:

            print("\n[SAXS peak-spread 4-region stats] Could not find spread value column.")
            print("Available columns:")
            print(stats_input.columns.tolist())

        elif "Region" not in stats_input.columns:

            print("\n[SAXS peak-spread 4-region stats] No Region column found in spread_samples.")
            print("Available columns:")
            print(stats_input.columns.tolist())

        else:

            stats_input = stats_input.rename(
                columns={
                    value_col: "SampleMean",
                }
            )

            all_stats = []

            # ------------------------------------------------------------
            # 1. Within-subtype paired spatial tests
            # ------------------------------------------------------------
            paired_region_pairs = [
                ("dermis_sub", "dermis_epi"),
                ("wound_sub", "wound_epi"),
            ]

            paired_stats = auto_saxs_multiple_paired_region_tests(
                sample_means=stats_input,
                region_pairs=paired_region_pairs,
                value_col="SampleMean",
                subtype_order=subtype_order,
                region_col="Region",
                sample_col=None,
            )

            if not paired_stats.empty:

                all_stats.append(paired_stats)

                if print_stats:

                    paired_print = paired_stats.copy()

                    for c in [
                        "MeanA",
                        "MeanB",
                        "MeanDifference_B_minus_A",
                        "Normality_p",
                        "t_stat",
                        "statistic",
                        "p_value",
                    ]:
                        if c in paired_print.columns:
                            paired_print[c] = pd.to_numeric(
                                paired_print[c],
                                errors="coerce",
                            )

                    paired_cols = [
                        c for c in [
                            "ComparisonType",
                            "Label",
                            "RegionA",
                            "RegionB",
                            "n_pairs",
                            "MeanA",
                            "MeanB",
                            "MeanDifference_B_minus_A",
                            "NormalityTest",
                            "Normality_p",
                            "NormalityInterpretation",
                            "SelectedTest",
                            "statistic",
                            "p_value",
                            "stars",
                        ]
                        if c in paired_print.columns
                    ]

                    print("\n[SAXS automatic paired spatial tests: peak spread lower/sub vs upper/epi regions]")
                    print(
                        paired_print[paired_cols]
                        .to_string(index=False)
                    )

            # ------------------------------------------------------------
            # 2. Between-subtype tests within each region
            # ------------------------------------------------------------
            between_stats = auto_saxs_between_subtype_tests(
                sample_means=stats_input,
                value_col="SampleMean",
                subtype_order=subtype_order,
                region_col="Region",
                sample_col=None,
            )

            if not between_stats.empty:

                between_stats["ComparisonType"] = "between-subtype test within region"
                all_stats.append(between_stats)

                if print_stats:

                    between_print = between_stats.copy()

                    for c in [
                        "statistic",
                        "p_value",
                    ]:
                        if c in between_print.columns:
                            between_print[c] = pd.to_numeric(
                                between_print[c],
                                errors="coerce",
                            )

                    between_cols = [
                        c for c in [
                            "ComparisonType",
                            "Region",
                            "Comparison",
                            "n_groups",
                            "GroupNs",
                            "NormalityTest",
                            "Normality_p_values",
                            "NormalityInterpretation",
                            "SelectedTest",
                            "statistic",
                            "p_value",
                            "stars",
                        ]
                        if c in between_print.columns
                    ]

                    print("\n[SAXS automatic between-subtype tests within each region: peak spread]")
                    print(
                        between_print[between_cols]
                        .to_string(index=False)
                    )

            if all_stats:
                stats_df = pd.concat(
                    all_stats,
                    ignore_index=True,
                    sort=False,
                )

    return fig, ax, stats_df

def _summarise_peak_spread_percent_shift(shift_df):
    if shift_df is None or shift_df.empty:
        return pd.DataFrame()

    return (
        shift_df
        .groupby(["SubtypeClean", "ShiftType"], as_index=False)["PercentShift"]
        .agg(mean="mean", std="std", n="count")
    )

def _plot_saxs_peak_spread_shift_bars(
    shift_df,
    shift_summary,
    shift_order,
    title,
    ylabel="% shift",
    subtype_order=SUBTYPE_ORDER,
    colours=NI_COLOURS,
    labels=NI_LABELS,
    shift_labels=None,
    ylim=None,
    jitter=0.045,
    point_size=24,
    alpha_range=(0.95, 0.45),
    hatches=True,
    shift_hatches=None,
    hatch_linewidth=0.7,
):
    """
    Generic % shift bar chart for peak-position spread.

    x = subtype
    grouped bars = shift type
    bars = mean % shift
    error bars = SD
    points = per-sample shifts

    hatches=True:
        Adds shift-specific hatch patterns and shows them in the legend.

    hatches=False:
        Uses colour/alpha only.
    """

    if shift_summary is None or shift_summary.empty:
        print("No peak-position spread shift summary to plot.")
        return None, None

    shift_labels = shift_labels or {}

    if shift_hatches is None:
        shift_hatches = {
            "wound_vs_dermis": "xxx",
            "lower_wound_vs_lower_dermis": "///",
            "upper_wound_vs_upper_dermis": "\\\\\\",
            "upper_dermis_vs_lower_dermis": "///",
            "upper_wound_vs_lower_wound": "xxx",
        }

    present_subtypes = [
        st for st in subtype_order
        if st in set(shift_summary["SubtypeClean"].astype(str))
    ]

    if not present_subtypes:
        print("No matching subtypes to plot.")
        return None, None

    x = np.arange(len(present_subtypes))
    width = 0.78 / max(len(shift_order), 1)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    rng = np.random.default_rng(0)

    shift_alphas = np.linspace(alpha_range[0], alpha_range[1], len(shift_order))

    old_hatch_lw = plt.rcParams.get("hatch.linewidth", 1.0)
    plt.rcParams["hatch.linewidth"] = hatch_linewidth

    try:
        for j, shift_type in enumerate(shift_order):
            offset = (j - (len(shift_order) - 1) / 2) * width

            means = []
            errs = []

            for subtype in present_subtypes:
                row = shift_summary[
                    (shift_summary["SubtypeClean"].astype(str) == subtype)
                    & (shift_summary["ShiftType"] == shift_type)
                ]

                if row.empty:
                    means.append(np.nan)
                    errs.append(np.nan)
                else:
                    means.append(float(row["mean"].iloc[0]))
                    errs.append(
                        float(row["std"].iloc[0])
                        if pd.notna(row["std"].iloc[0])
                        else np.nan
                    )

            means = np.asarray(means, dtype=float)
            errs = np.asarray(errs, dtype=float)
            bar_cols = [colours.get(st, "grey") for st in present_subtypes]

            bars = ax.bar(
                x + offset,
                means,
                width=width,
                color=bar_cols,
                alpha=shift_alphas[j],
                edgecolor="black",
                linewidth=0.6,
                zorder=2,
            )

            if hatches:
                for bar, subtype in zip(bars, present_subtypes):
                    hatch_colour = "black" if subtype == "control" else "lightgrey"

                    ax.bar(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        width=bar.get_width(),
                        color="none",
                        edgecolor=hatch_colour,
                        linewidth=0.0,
                        hatch=shift_hatches.get(shift_type, ""),
                        alpha=1.0,
                        zorder=3,
                    )

            ax.errorbar(
                x + offset,
                means,
                yerr=errs,
                fmt="none",
                ecolor="black",
                capsize=3,
                linewidth=1.1,
                zorder=4,
            )

            if shift_df is not None and not shift_df.empty:
                for i, subtype in enumerate(present_subtypes):
                    vals = shift_df.loc[
                        (shift_df["SubtypeClean"].astype(str) == subtype)
                        & (shift_df["ShiftType"] == shift_type),
                        "PercentShift",
                    ].to_numpy(float)

                    vals = vals[np.isfinite(vals)]

                    if vals.size == 0:
                        continue

                    xj = x[i] + offset + rng.normal(0, jitter, size=vals.size)

                    # ax.scatter(
                    #     xj,
                    #     vals,
                    #     color="black",
                    #     s=point_size,
                    #     alpha=0.65,
                    #     linewidths=0,
                    #     zorder=5,
                    # )

    finally:
        plt.rcParams["hatch.linewidth"] = old_hatch_lw

    ax.axhline(0, color="black", linewidth=1, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [labels.get(st, st) for st in present_subtypes],
        rotation=0,
        ha="right",
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)

    legend_handles = []
    for j, shift_type in enumerate(shift_order):
        if hatches:
            legend_handles.append(
                Patch(
                    facecolor="white",
                    edgecolor="black",
                    hatch=shift_hatches.get(shift_type, ""),
                    label=shift_labels.get(shift_type, shift_type),
                )
            )
        else:
            legend_handles.append(
                Patch(
                    facecolor="grey",
                    edgecolor="black",
                    alpha=shift_alphas[j],
                    label=shift_labels.get(shift_type, shift_type),
                )
            )

    ax.legend(handles=legend_handles, frameon=False)

    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()

    return fig, ax

def calculate_saxs_peak_spread_shift_dermis_to_wound(
    saxs_points,
    peak_param=SAXS_PEAK_SPREAD_PARAM,
    rsq_param=SAXS_PEAK_SPREAD_RSQ_PARAM,
    min_rsq=0.3,
    gap_thresh_deg=20.0,
):
    """
    Calculate % shift in spread:
        pooled dermis -> pooled wound
    """

    spread_samples, _ = calculate_saxs_peak_spread_samples(
        saxs_points=saxs_points,
        peak_param=peak_param,
        rsq_param=rsq_param,
        min_rsq=min_rsq,
        region_mode="pooled",
        gap_thresh_deg=gap_thresh_deg,
    )

    if spread_samples.empty:
        return pd.DataFrame(), pd.DataFrame()

    wide = (
        spread_samples
        .pivot_table(
            index=["experiment", "subtype", "SubtypeClean", "Filenumber"],
            columns="Region",
            values="SpreadDegrees",
            aggfunc="first",
        )
        .reset_index()
    )

    if "dermis" not in wide.columns or "wound" not in wide.columns:
        return pd.DataFrame(), pd.DataFrame()

    rows = []

    for _, r in wide.iterrows():
        dermis = r.get("dermis", np.nan)
        wound = r.get("wound", np.nan)

        if not np.isfinite(dermis) or dermis == 0 or not np.isfinite(wound):
            continue

        rows.append({
            "experiment": r["experiment"],
            "subtype": r["subtype"],
            "SubtypeClean": r["SubtypeClean"],
            "Filenumber": r["Filenumber"],
            "ShiftType": "wound_vs_dermis",
            "BaselineRegion": "dermis",
            "TargetRegion": "wound",
            "BaselineValue": dermis,
            "TargetValue": wound,
            "PercentShift": ((wound - dermis) / dermis) * 100,
        })

    shift_df = pd.DataFrame(rows)
    shift_summary = _summarise_peak_spread_percent_shift(shift_df)

    return shift_df, shift_summary

def plot_saxs_peak_spread_shift_dermis_to_wound(
    saxs_points,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=None,
    min_rsq=0.3,
    gap_thresh_deg=20.0,
    hatches=True,
):
    """
    % shift from dermis spread to wound spread.
    """

    subtype_order = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)

    shift_df, shift_summary = calculate_saxs_peak_spread_shift_dermis_to_wound(
        saxs_points=saxs_points,
        min_rsq=min_rsq,
        gap_thresh_deg=gap_thresh_deg,
    )

    return _plot_saxs_peak_spread_shift_bars(
        shift_df=shift_df,
        shift_summary=shift_summary,
        shift_order=["wound_vs_dermis"],
        title="",
        # title="Peak-position spread % shift from dermis to wound",
        ylabel="% shift from dermis spread",
        subtype_order=subtype_order,
        ylim=ylim,
        hatches=hatches,
        shift_labels={
            "wound_vs_dermis": "Wound vs dermis",
        },
    )

def calculate_saxs_peak_spread_shift_matched_dermis_to_wound(
    saxs_points,
    peak_param=SAXS_PEAK_SPREAD_PARAM,
    rsq_param=SAXS_PEAK_SPREAD_RSQ_PARAM,
    min_rsq=0.3,
    gap_thresh_deg=20.0,
):
    """
    Calculate % shift in spread:
        lower dermis -> lower wound
        upper dermis -> upper wound
    """

    spread_samples, _ = calculate_saxs_peak_spread_samples(
        saxs_points=saxs_points,
        peak_param=peak_param,
        rsq_param=rsq_param,
        min_rsq=min_rsq,
        region_mode="split",
        gap_thresh_deg=gap_thresh_deg,
    )

    if spread_samples.empty:
        return pd.DataFrame(), pd.DataFrame()

    wide = (
        spread_samples
        .pivot_table(
            index=["experiment", "subtype", "SubtypeClean", "Filenumber"],
            columns="Region",
            values="SpreadDegrees",
            aggfunc="first",
        )
        .reset_index()
    )

    rows = []

    shift_defs = {
        "lower_wound_vs_lower_dermis": ("dermis_sub", "wound_sub"),
        "upper_wound_vs_upper_dermis": ("dermis_epi", "wound_epi"),
    }

    for _, r in wide.iterrows():
        for shift_type, (base_region, target_region) in shift_defs.items():
            base = r.get(base_region, np.nan)
            target = r.get(target_region, np.nan)

            if not np.isfinite(base) or base == 0 or not np.isfinite(target):
                continue

            rows.append({
                "experiment": r["experiment"],
                "subtype": r["subtype"],
                "SubtypeClean": r["SubtypeClean"],
                "Filenumber": r["Filenumber"],
                "ShiftType": shift_type,
                "BaselineRegion": base_region,
                "TargetRegion": target_region,
                "BaselineValue": base,
                "TargetValue": target,
                "PercentShift": ((target - base) / base) * 100,
            })

    shift_df = pd.DataFrame(rows)
    shift_summary = _summarise_peak_spread_percent_shift(shift_df)

    return shift_df, shift_summary

def plot_saxs_peak_spread_shift_matched_dermis_to_wound(
    saxs_points,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=None,
    min_rsq=0.3,
    gap_thresh_deg=20.0,
    hatches=True,
):
    """
    % shift in spread:
        lower dermis -> lower wound
        upper dermis -> upper wound
    """

    subtype_order = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)

    shift_df, shift_summary = calculate_saxs_peak_spread_shift_matched_dermis_to_wound(
        saxs_points=saxs_points,
        min_rsq=min_rsq,
        gap_thresh_deg=gap_thresh_deg,
    )

    return _plot_saxs_peak_spread_shift_bars(
        shift_df=shift_df,
        shift_summary=shift_summary,
        shift_order=[
            "lower_wound_vs_lower_dermis",
            "upper_wound_vs_upper_dermis",
        ],
        title="Peak-position spread % shift from matched dermis to wound",
        ylabel="% shift from matched dermis spread",
        subtype_order=subtype_order,
        ylim=ylim,
        hatches=hatches,
        shift_labels={
            "lower_wound_vs_lower_dermis": "Lower wound vs lower dermis",
            "upper_wound_vs_upper_dermis": "Upper wound vs upper dermis",
        },
    )

def calculate_saxs_peak_spread_shift_lower_to_upper(
    saxs_points,
    peak_param=SAXS_PEAK_SPREAD_PARAM,
    rsq_param=SAXS_PEAK_SPREAD_RSQ_PARAM,
    min_rsq=0.3,
    gap_thresh_deg=20.0,
):
    """
    Calculate % shift in spread:
        lower dermis -> upper dermis
        lower wound -> upper wound
    """

    spread_samples, _ = calculate_saxs_peak_spread_samples(
        saxs_points=saxs_points,
        peak_param=peak_param,
        rsq_param=rsq_param,
        min_rsq=min_rsq,
        region_mode="split",
        gap_thresh_deg=gap_thresh_deg,
    )

    if spread_samples.empty:
        return pd.DataFrame(), pd.DataFrame()

    wide = (
        spread_samples
        .pivot_table(
            index=["experiment", "subtype", "SubtypeClean", "Filenumber"],
            columns="Region",
            values="SpreadDegrees",
            aggfunc="first",
        )
        .reset_index()
    )

    rows = []

    shift_defs = {
        "upper_dermis_vs_lower_dermis": ("dermis_sub", "dermis_epi"),
        "upper_wound_vs_lower_wound": ("wound_sub", "wound_epi"),
    }

    for _, r in wide.iterrows():
        for shift_type, (base_region, target_region) in shift_defs.items():
            base = r.get(base_region, np.nan)
            target = r.get(target_region, np.nan)

            if not np.isfinite(base) or base == 0 or not np.isfinite(target):
                continue

            rows.append({
                "experiment": r["experiment"],
                "subtype": r["subtype"],
                "SubtypeClean": r["SubtypeClean"],
                "Filenumber": r["Filenumber"],
                "ShiftType": shift_type,
                "BaselineRegion": base_region,
                "TargetRegion": target_region,
                "BaselineValue": base,
                "TargetValue": target,
                "PercentShift": ((target - base) / base) * 100,
            })

    shift_df = pd.DataFrame(rows)
    shift_summary = _summarise_peak_spread_percent_shift(shift_df)

    return shift_df, shift_summary

def plot_saxs_peak_spread_shift_lower_to_upper(
    saxs_points,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    ylim=None,
    min_rsq=0.3,
    gap_thresh_deg=20.0,
    hatches=True,
):
    """
    % shift in spread:
        lower dermis -> upper dermis
        lower wound -> upper wound
    """

    subtype_order = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)

    shift_df, shift_summary = calculate_saxs_peak_spread_shift_lower_to_upper(
        saxs_points=saxs_points,
        min_rsq=min_rsq,
        gap_thresh_deg=gap_thresh_deg,
    )

    return _plot_saxs_peak_spread_shift_bars(
        shift_df=shift_df,
        shift_summary=shift_summary,
        shift_order=[
            "upper_dermis_vs_lower_dermis",
            "upper_wound_vs_lower_wound",
        ],
        title="Peak-position spread % shift from lower to upper regions",
        ylabel="% shift from lower-region spread",
        subtype_order=subtype_order,
        ylim=ylim,
        hatches=hatches,
        shift_labels={
            "upper_dermis_vs_lower_dermis": "Upper dermis vs lower dermis",
            "upper_wound_vs_lower_wound": "Upper wound vs lower wound",
        },
    )

# ====SAXS STATS=========================================================================



def auto_saxs_paired_region_tests(
    sample_means,
    value_col="SampleMean",
    subtype_order=None,
    subtype_col="SubtypeClean",
    region_col="Region",
    sample_col=None,
    region_a="dermis",
    region_b="wound",
    normality_alpha=0.05,
):
    """
    Automatically perform paired SAXS statistics between two matched regions.

    Designed for comparisons such as:
        dermis vs wound
        dermis_sub vs dermis_epi
        wound_sub vs wound_epi

    Uses sample-level means, not point-level SAXS data.

    Test selection:
        Always uses paired t-test.

    Shapiro-Wilk normality is still calculated on paired differences
    and printed in the output table, but it does not control the test choice.
    """

    if sample_means is None or sample_means.empty:
        return pd.DataFrame()

    d = sample_means.copy()

    if value_col not in d.columns:
        raise KeyError(f"value_col '{value_col}' not found in sample_means")

    if subtype_col not in d.columns:
        raise KeyError(f"subtype_col '{subtype_col}' not found in sample_means")

    if region_col not in d.columns:
        if "RegionForPlot" in d.columns:
            region_col = "RegionForPlot"
        elif "Region" in d.columns:
            region_col = "Region"
        else:
            raise KeyError("No usable region column found in sample_means")

    if sample_col is None:
        possible_sample_cols = [
            "SampleKey",
            "Sample",
            "SampleID",
            "sample",
            "Filenumber",
            "FileNumber",
            "File number",
            "Sample Number",
            "SampleNumber",
            "experiment",
            "Experiment",
        ]

        for c in possible_sample_cols:
            if c in d.columns:
                sample_col = c
                break

    if sample_col is None:
        print("\n[auto_saxs_paired_region_tests] Could not find sample column.")
        print("Available columns:")
        print(d.columns.tolist())

        raise KeyError(
            "No sample column found. Add the correct column name using "
            "sample_col='your_sample_column_name'."
        )

    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    d = d.dropna(
        subset=[
            value_col,
            subtype_col,
            region_col,
            sample_col,
        ]
    ).copy()

    if subtype_order is None:
        subtype_order = list(d[subtype_col].astype(str).unique())
    else:
        subtype_order = [str(s) for s in subtype_order]

    rows = []

    for subtype in subtype_order:

        d_sub = d[
            d[subtype_col].astype(str) == str(subtype)
        ].copy()

        if d_sub.empty:
            continue

        d_pair = d_sub[
            d_sub[region_col].astype(str).isin(
                [
                    region_a,
                    region_b,
                ]
            )
        ].copy()

        if d_pair.empty:
            continue

        wide = d_pair.pivot_table(
            index=sample_col,
            columns=region_col,
            values=value_col,
            aggfunc="mean",
        )

        if region_a not in wide.columns or region_b not in wide.columns:
            continue

        wide = wide[
            [
                region_a,
                region_b,
            ]
        ].dropna()

        n_pairs = len(wide)

        if n_pairs < 2:
            rows.append(
                {
                    "SubtypeClean": subtype,
                    "Label": str(subtype),
                    "RegionA": region_a,
                    "RegionB": region_b,
                    "n_pairs": n_pairs,
                    "MeanA": wide[region_a].mean() if region_a in wide else np.nan,
                    "MeanB": wide[region_b].mean() if region_b in wide else np.nan,
                    "MeanDifference_B_minus_A": np.nan,
                    "NormalityTest": "not enough pairs",
                    "Normality_p": np.nan,
                    "NormalityInterpretation": "not tested",
                    "SelectedTest": "not enough pairs",
                    "t_stat": np.nan,
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "stars": "ns",
                }
            )
            continue

        a = wide[region_a].astype(float)
        b = wide[region_b].astype(float)

        differences = b - a

        mean_a = a.mean()
        mean_b = b.mean()
        mean_diff = differences.mean()

        # ------------------------------------------------------------
        # Shapiro-Wilk on paired differences
        # Still printed, but does not control test selection
        # ------------------------------------------------------------
        if n_pairs >= 3:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(differences)
                normality_test = "Shapiro-Wilk on paired differences"
                normality_interpretation = (
                    "normal"
                    if shapiro_p >= normality_alpha
                    else "non-normal"
                )
            except Exception:
                shapiro_p = np.nan
                normality_test = "Shapiro-Wilk failed"
                normality_interpretation = "not tested"
        else:
            shapiro_p = np.nan
            normality_test = "not enough pairs for Shapiro-Wilk"
            normality_interpretation = "not tested"

        # ------------------------------------------------------------
        # Always use paired t-test
        # ------------------------------------------------------------
        selected_test = "paired t-test"

        try:
            stat, p_value = stats.ttest_rel(
                a,
                b,
                nan_policy="omit",
            )
        except Exception:
            stat, p_value = np.nan, np.nan

        t_stat = stat

        rows.append(
            {
                "SubtypeClean": subtype,
                "Label": str(subtype),
                "RegionA": region_a,
                "RegionB": region_b,
                "n_pairs": n_pairs,
                "MeanA": mean_a,
                "MeanB": mean_b,
                "MeanDifference_B_minus_A": mean_diff,
                "NormalityTest": normality_test,
                "Normality_p": shapiro_p,
                "NormalityInterpretation": normality_interpretation,
                "SelectedTest": selected_test,
                "t_stat": t_stat,
                "statistic": stat,
                "p_value": p_value,
                "stars": p_to_stars(p_value),
            }
        )

    return pd.DataFrame(rows)


def auto_saxs_multiple_paired_region_tests(
    sample_means,
    region_pairs,
    value_col="SampleMean",
    subtype_order=None,
    subtype_col="SubtypeClean",
    region_col="Region",
    sample_col=None,
    normality_alpha=0.05,
):
    """
    Run paired t-tests for multiple region pairs.

    Example region pairs:
        dermis_sub vs dermis_epi
        wound_sub vs wound_epi

    Uses sample-level means.

    This function simply loops through region pairs and calls
    auto_saxs_paired_region_tests(), which always uses paired t-test.
    """

    if sample_means is None or sample_means.empty:
        return pd.DataFrame()

    if region_col not in sample_means.columns:
        print("\n[auto_saxs_multiple_paired_region_tests] No region column found.")
        print("Available columns:")
        print(sample_means.columns.tolist())
        return pd.DataFrame()

    all_stats = []

    regions_present = set(
        sample_means[region_col]
        .astype(str)
        .str.strip()
        .unique()
    )

    for region_a, region_b in region_pairs:

        if region_a not in regions_present or region_b not in regions_present:
            continue

        stats_df = auto_saxs_paired_region_tests(
            sample_means=sample_means,
            value_col=value_col,
            subtype_order=subtype_order,
            subtype_col=subtype_col,
            region_col=region_col,
            sample_col=sample_col,
            region_a=region_a,
            region_b=region_b,
            normality_alpha=normality_alpha,
        )

        if not stats_df.empty:
            stats_df["ComparisonType"] = "within-subtype paired region test"
            all_stats.append(stats_df)

    if not all_stats:
        return pd.DataFrame()

    return pd.concat(
        all_stats,
        ignore_index=True,
        sort=False,
    )


def auto_saxs_between_subtype_tests(
    sample_means,
    value_col="SampleMean",
    subtype_col="SubtypeClean",
    region_col="Region",
    sample_col=None,
    subtype_order=None,
    normality_alpha=0.05,
):
    """
    Compare SAXS sample-level means between subtypes within each plotted region.

    Uses sample-level means, not point-level SAXS data.

    Test selection:
        Always uses one-way ANOVA when two or more subtype groups are present.

    Shapiro-Wilk normality is still calculated per subtype and printed
    in the output table, but it does not control the test choice.

    Note:
        For two subtypes, this still uses one-way ANOVA.
        A one-way ANOVA with two groups is mathematically equivalent
        to the standard independent two-sample t-test in terms of p-value.
    """

    if sample_means is None or sample_means.empty:
        return pd.DataFrame()

    d = sample_means.copy()

    if value_col not in d.columns:
        raise KeyError(f"value_col '{value_col}' not found in sample_means")

    if subtype_col not in d.columns:
        raise KeyError(f"subtype_col '{subtype_col}' not found in sample_means")

    if region_col not in d.columns:
        raise KeyError(f"region_col '{region_col}' not found in sample_means")

    if sample_col is None:
        possible_sample_cols = [
            "SampleKey",
            "Sample",
            "SampleID",
            "sample",
            "Filenumber",
            "FileNumber",
            "File number",
            "Sample Number",
            "SampleNumber",
            "experiment",
            "Experiment",
        ]

        for c in possible_sample_cols:
            if c in d.columns:
                sample_col = c
                break

    if sample_col is None:
        print("\n[auto_saxs_between_subtype_tests] Could not find sample column.")
        print("Available columns:")
        print(d.columns.tolist())

        raise KeyError(
            "No sample column found. Add the correct column name using "
            "sample_col='your_sample_column_name'."
        )

    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    d = d.dropna(
        subset=[
            value_col,
            subtype_col,
            region_col,
            sample_col,
        ]
    ).copy()

    if subtype_order is None:
        subtype_order = list(d[subtype_col].astype(str).unique())
    else:
        subtype_order = [str(s) for s in subtype_order]

    rows = []

    for region in d[region_col].astype(str).unique():

        d_region = d[
            d[region_col].astype(str) == str(region)
        ].copy()

        groups = []
        group_names = []

        for subtype in subtype_order:

            vals = (
                d_region.loc[
                    d_region[subtype_col].astype(str) == str(subtype),
                    value_col,
                ]
                .dropna()
                .astype(float)
            )

            if len(vals) > 0:
                groups.append(vals.to_numpy())
                group_names.append(str(subtype))

        if len(groups) < 2:
            continue

        normality_results = []

        for name, vals in zip(group_names, groups):

            if len(vals) >= 3:
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(vals)
                except Exception:
                    shapiro_p = np.nan
            else:
                shapiro_p = np.nan

            normality_results.append(
                f"{name}: p={shapiro_p:.4g}"
                if not pd.isna(shapiro_p)
                else f"{name}: n<3"
            )

        # ------------------------------------------------------------
        # Always use one-way ANOVA
        # ------------------------------------------------------------
        selected_test = "one-way ANOVA"

        try:
            stat, p_value = stats.f_oneway(*groups)
        except Exception:
            stat, p_value = np.nan, np.nan

        comparison = " vs ".join(group_names)

        rows.append(
            {
                "Region": region,
                "Comparison": comparison,
                "n_groups": len(groups),
                "GroupNs": ", ".join(
                    f"{name}: n={len(vals)}"
                    for name, vals in zip(group_names, groups)
                ),
                "NormalityTest": "Shapiro-Wilk per subtype",
                "Normality_p_values": "; ".join(normality_results),
                "NormalityInterpretation": "reported only; ANOVA used regardless",
                "SelectedTest": selected_test,
                "statistic": stat,
                "p_value": p_value,
                "stars": p_to_stars(p_value),
            }
        )

    return pd.DataFrame(rows)


def auto_saxs_long_shift_between_subtype_tests(
    shift_df,
    shift_type_col="ShiftType",
    value_col="PercentShift",
    subtype_col="SubtypeClean",
    sample_col=None,
    subtype_order=None,
    shift_order=None,
    shift_labels=None,
    normality_alpha=0.05,
):
    """
    Compare sample-level shift values between subtypes.

    Designed for long-format shift data:

        SubtypeClean | Filenumber | ShiftType | PercentShift

    Example comparisons:
        PBS vs 4W for wound_vs_dermis
        4W vs 4W+Met for upper_dermis_vs_lower_dermis

    Test selection:
        Always uses one-way ANOVA when two or more subtype groups are present.

    Shapiro-Wilk normality is still calculated per subtype and printed
    in the output table, but it does not control the test choice.

    Note:
        For two subtypes, this still uses one-way ANOVA.
        A one-way ANOVA with two groups is mathematically equivalent
        to the standard independent two-sample t-test in terms of p-value.
    """

    if shift_df is None or shift_df.empty:
        return pd.DataFrame()

    d = shift_df.copy()

    required_cols = [
        shift_type_col,
        value_col,
        subtype_col,
    ]

    missing = [
        c for c in required_cols
        if c not in d.columns
    ]

    if missing:
        print("\n[auto_saxs_long_shift_between_subtype_tests] Missing columns:")
        print(missing)
        print("Available columns:")
        print(d.columns.tolist())
        return pd.DataFrame()

    if sample_col is None:
        possible_sample_cols = [
            "SampleKey",
            "Sample",
            "SampleID",
            "sample",
            "Filenumber",
            "FileNumber",
            "File number",
            "Sample Number",
            "SampleNumber",
            "experiment",
            "Experiment",
        ]

        for c in possible_sample_cols:
            if c in d.columns:
                sample_col = c
                break

    d[value_col] = pd.to_numeric(
        d[value_col],
        errors="coerce",
    )

    d = d.dropna(
        subset=[
            value_col,
            subtype_col,
            shift_type_col,
        ]
    ).copy()

    if d.empty:
        return pd.DataFrame()

    if subtype_order is None:
        subtype_order = list(d[subtype_col].astype(str).unique())
    else:
        subtype_order = [str(s) for s in subtype_order]

    if shift_order is None:
        shift_order = list(d[shift_type_col].astype(str).unique())

    if shift_labels is None:
        shift_labels = {
            s: s
            for s in shift_order
        }

    rows = []

    for shift_type in shift_order:

        d_shift = d[
            d[shift_type_col].astype(str) == str(shift_type)
        ].copy()

        if d_shift.empty:
            continue

        groups = []
        group_names = []

        for subtype in subtype_order:

            vals = (
                d_shift.loc[
                    d_shift[subtype_col].astype(str) == str(subtype),
                    value_col,
                ]
                .dropna()
                .astype(float)
            )

            if len(vals) > 0:
                groups.append(vals.to_numpy())
                group_names.append(str(subtype))

        if len(groups) < 2:
            continue

        normality_results = []

        for name, vals in zip(group_names, groups):

            if len(vals) >= 3:
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(vals)
                except Exception:
                    shapiro_p = np.nan
            else:
                shapiro_p = np.nan

            normality_results.append(
                f"{name}: p={shapiro_p:.4g}"
                if not pd.isna(shapiro_p)
                else f"{name}: n<3"
            )

        # ------------------------------------------------------------
        # Always use one-way ANOVA
        # ------------------------------------------------------------
        selected_test = "one-way ANOVA"

        try:
            stat, p_value = stats.f_oneway(*groups)
        except Exception:
            stat, p_value = np.nan, np.nan

        comparison = " vs ".join(group_names)

        rows.append(
            {
                "ShiftType": shift_type,
                "ShiftLabel": shift_labels.get(
                    shift_type,
                    shift_type,
                ),
                "Comparison": comparison,
                "n_groups": len(groups),
                "GroupNs": ", ".join(
                    f"{name}: n={len(vals)}"
                    for name, vals in zip(group_names, groups)
                ),
                "NormalityTest": "Shapiro-Wilk per subtype",
                "Normality_p_values": "; ".join(normality_results),
                "NormalityInterpretation": "reported only; ANOVA used regardless",
                "SelectedTest": selected_test,
                "statistic": stat,
                "p_value": p_value,
                "stars": p_to_stars(p_value),
            }
        )

    return pd.DataFrame(rows)

# =============================================================================

# raman plotting

def prepare_raman_raw_for_plot(
    raw_raman_df,
    *,
    wave_col="Wave",
    intensity_col="Intensity",
):
    """
    Prepare already-loaded RawRaman data for plotting.

    Adds:
        WaveRound = rounded whole-number wavenumber
    """

    d = raw_raman_df.copy()
    d.columns = d.columns.astype(str).str.strip()

    required = [
        "Sample",
        "Subtype",
        "SpectralRegion",
        "AnatomicalRegion",
        "PointIndex",
        "x",
        "y",
        wave_col,
        intensity_col,
    ]

    missing = [c for c in required if c not in d.columns]
    if missing:
        raise KeyError(f"Missing required Raman columns: {missing}")

    for col in ["Sample", "PointIndex", "x", "y", wave_col, intensity_col]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.dropna(subset=["Sample", "x", "y", wave_col, intensity_col]).copy()

    d["WaveRound"] = d[wave_col].round().astype(int)

    if "SubtypeClean" not in d.columns:
        d["SubtypeClean"] = d["Subtype"].astype(str).str.strip().str.lower()

    return d

def prepare_raman_point_spectra(
    raman_raw,
    *,
    spectral_region="FP",
    anatomical_region=None,
    wave_min=None,
    wave_max=None,
):
    """
    Convert raw Raman rows into one spectrum per Sample/Subtype/x/y/PointIndex.

    Rounds Wave to whole number and averages duplicate rounded wavenumbers
    within the same point spectrum.
    """

    d = prepare_raman_raw_for_plot(raman_raw)

    if spectral_region is not None and "SpectralRegion" in d.columns:
        d = d[d["SpectralRegion"].astype(str).str.strip() == str(spectral_region)].copy()

    if anatomical_region is not None and "AnatomicalRegion" in d.columns:
        d = d[
            d["AnatomicalRegion"].astype(str).str.lower().str.strip()
            == str(anatomical_region).lower()
        ].copy()

    if wave_min is not None:
        d = d[d["WaveRound"] >= wave_min].copy()

    if wave_max is not None:
        d = d[d["WaveRound"] <= wave_max].copy()

    if d.empty:
        print("No Raman rows left after filtering.")
        return pd.DataFrame()

    group_cols = [
        "Sample",
        "Subtype",
        "SpectralRegion",
        "AnatomicalRegion",
        "PointIndex",
        "x",
        "y",
        "WaveRound",
    ]

    spectra = (
        d.groupby(group_cols, as_index=False, observed=True)["Intensity"]
        .mean()
    )

    spectra = add_raman_normalised_position(spectra)

    return spectra

def add_raman_normalised_position(spectra):
    """
    Add NormPos from 0 to 100 within each Sample/Subtype scan.

    Uses PointIndex order where available.
    """

    d = spectra.copy()

    point_cols = ["Sample", "Subtype", "PointIndex", "x", "y"]

    points = (
        d[point_cols]
        .drop_duplicates()
        .sort_values(["Sample", "Subtype", "PointIndex", "x", "y"])
        .copy()
    )

    norm_rows = []

    for (sample, subtype), g in points.groupby(["Sample", "Subtype"], observed=True):
        g = g.sort_values(["PointIndex", "x", "y"]).copy()
        n = len(g)

        if n <= 1:
            g["NormPos"] = 0.0
        else:
            g["NormPos"] = np.linspace(0, 100, n)

        norm_rows.append(g)

    pos_df = pd.concat(norm_rows, ignore_index=True)

    d = d.merge(
        pos_df[point_cols + ["NormPos"]],
        on=point_cols,
        how="left",
    )

    return d

def plot_raman_point_spectra_by_subtype(
    spectra,
    *,
    subtype_order=None,
    wave_col="WaveRound",
    intensity_col="Intensity",
    norm_col="NormPos",
    cmap="jet",
    alpha=0.35,
    linewidth=0.8,
    xlim=(1200, 1800),
    ylim=None,
    ncols=3,
    title="Raw Raman point spectra by normalised position",
    plot_every_nth=1,
    panel_by="subtype",   # "subtype" or "sample"
):
    """
    Plot raw Raman point spectra.

    panel_by="subtype":
        one panel per subtype
        each line = one x/y point spectrum

    panel_by="sample":
        one panel per sample
        panel title = subtype + sample
        each line = one x/y point spectrum

    plot_every_nth:
        1 = plot every spectrum
        2 = plot every second point spectrum
        5 = plot every fifth point spectrum
    """

    if spectra is None or spectra.empty:
        print("No Raman spectra to plot.")
        return None, None

    if panel_by not in {"subtype", "sample"}:
        raise ValueError("panel_by must be 'subtype' or 'sample'.")

    plot_every_nth = max(int(plot_every_nth), 1)

    d = spectra.copy()
    d[wave_col] = pd.to_numeric(d[wave_col], errors="coerce")
    d[intensity_col] = pd.to_numeric(d[intensity_col], errors="coerce")
    d[norm_col] = pd.to_numeric(d[norm_col], errors="coerce")
    d["Subtype"] = d["Subtype"].astype(str).str.strip()

    d = d.dropna(subset=[wave_col, intensity_col, norm_col, "Subtype"]).copy()

    if d.empty:
        print("No Raman spectra left after cleaning.")
        return None, None

    present = list(pd.unique(d["Subtype"]))

    if subtype_order is None:
        subtypes = present
    else:
        subtypes = [st for st in subtype_order if st in present]

    if not subtypes:
        print("No matching subtypes to plot.")
        return None, None

    d = d[d["Subtype"].isin(subtypes)].copy()

    spectrum_id_cols = ["Sample", "Subtype", "PointIndex", "x", "y"]
    spectrum_id_cols = [c for c in spectrum_id_cols if c in d.columns]

    point_table = (
        d[spectrum_id_cols + [norm_col]]
        .drop_duplicates()
        .sort_values(["Subtype", "Sample", "PointIndex", "x", "y"])
        .copy()
    )

    # Keep only every nth point spectrum within each sample/subtype.
    keep_rows = []

    for _, g in point_table.groupby(["Subtype", "Sample"], observed=True):
        g = g.sort_values(["PointIndex", "x", "y"]).copy()
        keep_rows.append(g.iloc[::plot_every_nth])

    keep_points = pd.concat(keep_rows, ignore_index=True) if keep_rows else pd.DataFrame()

    d = d.merge(
        keep_points[spectrum_id_cols],
        on=spectrum_id_cols,
        how="inner",
    )

    if d.empty:
        print("No Raman spectra left after plot_every_nth filtering.")
        return None, None

    norm = plt.Normalize(0, 100)
    cm = plt.get_cmap(cmap)

    # ------------------------------------------------------------------
    # Define panels
    # ------------------------------------------------------------------
    if panel_by == "subtype":
        panel_rows = [
            {
                "PanelKey": subtype,
                "Subtype": subtype,
                "Sample": None,
                "Title": subtype_label(subtype) if "subtype_label" in globals() else str(subtype),
            }
            for subtype in subtypes
        ]

    else:
        samples = (
            d[["Subtype", "Sample"]]
            .drop_duplicates()
            .sort_values(["Subtype", "Sample"])
        )

        panel_rows = []

        for _, r in samples.iterrows():
            subtype = str(r["Subtype"])
            sample = r["Sample"]

            label = subtype_label(subtype) if "subtype_label" in globals() else subtype

            panel_rows.append({
                "PanelKey": f"{subtype}_{sample}",
                "Subtype": subtype,
                "Sample": sample,
                "Title": f"{label} | Sample {sample}",
            })

    if not panel_rows:
        print("No panels to plot.")
        return None, None

    n_panels = len(panel_rows)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.2 * ncols, 4.2 * nrows),
        sharex=True,
        sharey=True,
    )

    axes = np.atleast_1d(axes).ravel()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    for ax, panel in zip(axes, panel_rows):
        if panel_by == "subtype":
            sub = d[d["Subtype"] == panel["Subtype"]].copy()
        else:
            sub = d[
                (d["Subtype"] == panel["Subtype"])
                & (d["Sample"] == panel["Sample"])
            ].copy()

        for _, g in sub.groupby(spectrum_id_cols, observed=True):
            g = g.sort_values(wave_col)

            x = g[wave_col].to_numpy(float)
            y = g[intensity_col].to_numpy(float)

            pos = g[norm_col].iloc[0]
            colour = cm(norm(pos))

            ax.plot(
                x,
                y,
                color=colour,
                alpha=alpha,
                linewidth=linewidth,
            )

        ax.set_title(panel["Title"])
        ax.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax.set_ylabel("Intensity")
        ax.grid(False)

        if xlim is not None:
            ax.set_xlim(*xlim)

        if ylim is not None:
            ax.set_ylim(*ylim)

    for ax in axes[n_panels:]:
        ax.axis("off")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes[:n_panels],
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label("Normalised position through line scan (%)")

    fig.suptitle(title, y=0.98)
    fig.subplots_adjust(
        left=0.08,
        right=0.90,
        bottom=0.10,
        top=0.90,
        wspace=0.25,
        hspace=0.30,
    )
    plt.show()

    return fig, axes

def plot_raman_average_spectra(
    spectra,
    *,
    subtype_order=None,
    colours=None,
    linestyles=None,
    wave_col="WaveRound",
    intensity_col="Intensity",
    err_mode="sem",          # "sem" or "std"
    average_unit="sample",   # "sample" or "point"
    show_error=True,
    error_alpha=0.25,
    linewidth=2.0,
    xlim=(1200, 1800),
    ylim=None,
    interp_grid=True,
    title="Average Raman spectra by subtype",
    show_peak_regions=False,
    peak_regions=None,
    peak_colours=None,
    peak_alpha=0.35,
    peak_linewidth=1.0,
    peak_label_y=0.98,
):
    """
    Average Raman spectra by subtype.

    average_unit:
        "point"
            Every x/y point spectrum is treated as one replicate.
            SEM = std / sqrt(number of point spectra).

        "sample"
            Point spectra are first averaged within each sample,
            then subtype mean/error is calculated across sample means.
            SEM = std / sqrt(number of samples).

    show_peak_regions=True:
        Adds vertical dashed lines at each peak-region boundary and labels
        the region using the first part of the region name.
    """

    def _peak_display_name(name):
        parts = str(name).split("_")

        if name.startswith("CH2CH3"):
            return "CH2/CH3"
        if name.startswith("AmideI"):
            return "Amide I"
        if name.startswith("AmideIII"):
            return "Amide III"

        return parts[0]

    def _add_peak_region_lines(ax):
        if not show_peak_regions or not peak_regions:
            return

        default_peak_colours = [
            "tab:purple",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:blue",
            "tab:brown",
            "tab:pink",
            "tab:cyan",
        ]

        used_labels = set()

        for i, (region_name, (x1, x2)) in enumerate(peak_regions):
            colour = (
                peak_colours.get(region_name)
                if isinstance(peak_colours, dict) and region_name in peak_colours
                else default_peak_colours[i % len(default_peak_colours)]
            )

            label = _peak_display_name(region_name)
            legend_label = label if label not in used_labels else None
            used_labels.add(label)

            ax.axvline(
                x1,
                color=colour,
                linestyle="--",
                alpha=peak_alpha,
                linewidth=peak_linewidth,
                label=legend_label,
                zorder=0,
            )

            ax.axvline(
                x2,
                color=colour,
                linestyle="--",
                alpha=peak_alpha,
                linewidth=peak_linewidth,
                zorder=0,
            )

            x_mid = (x1 + x2) / 2

            ax.text(
                x_mid,
                peak_label_y,
                label,
                color=colour,
                ha="center",
                va="top",
                rotation=90,
                fontsize=8,
                alpha=min(peak_alpha + 0.35, 1.0),
                transform=ax.get_xaxis_transform(),
            )

    colours = colours or {}
    linestyles = linestyles or {}

    if spectra is None or spectra.empty:
        print("No Raman spectra to average.")
        return None, None, pd.DataFrame()

    if average_unit not in {"point", "sample"}:
        raise ValueError("average_unit must be 'point' or 'sample'.")

    d = spectra.copy()
    d[wave_col] = pd.to_numeric(d[wave_col], errors="coerce")
    d[intensity_col] = pd.to_numeric(d[intensity_col], errors="coerce")
    d["Subtype"] = d["Subtype"].astype(str).str.strip()

    d = d.dropna(subset=[wave_col, intensity_col, "Subtype"]).copy()

    if xlim is not None:
        d = d[(d[wave_col] >= xlim[0]) & (d[wave_col] <= xlim[1])].copy()

    if d.empty:
        print("No Raman rows left after filtering.")
        return None, None, pd.DataFrame()

    present = list(pd.unique(d["Subtype"]))

    if subtype_order is None:
        subtypes = present
    else:
        subtypes = [st for st in subtype_order if st in present]

    if not subtypes:
        print("No matching subtypes to plot.")
        return None, None, pd.DataFrame()

    point_id_cols = ["Sample", "Subtype", "PointIndex", "x", "y"]
    point_id_cols = [c for c in point_id_cols if c in d.columns]

    if xlim is not None:
        wave_grid = np.arange(int(xlim[0]), int(xlim[1]) + 1)
    else:
        wave_grid = np.arange(
            int(np.nanmin(d[wave_col])),
            int(np.nanmax(d[wave_col])) + 1
        )

    def _point_stack(ds):
        rows = []

        for _, g in ds.groupby(point_id_cols, observed=True):
            g = g.sort_values(wave_col).copy()

            gg = (
                g.groupby(wave_col, as_index=False)[intensity_col]
                .mean()
                .sort_values(wave_col)
            )

            x = gg[wave_col].to_numpy(float)
            y = gg[intensity_col].to_numpy(float)

            ok = np.isfinite(x) & np.isfinite(y)
            x = x[ok]
            y = y[ok]

            if x.size < 2:
                continue

            xu, idx = np.unique(x, return_index=True)
            yu = y[idx]

            if interp_grid:
                y_grid = np.interp(wave_grid, xu, yu)
                y_grid[(wave_grid < xu.min()) | (wave_grid > xu.max())] = np.nan
            else:
                y_grid = np.full_like(wave_grid, np.nan, dtype=float)
                mapper = dict(zip(xu.astype(int), yu))
                present_waves = np.array([w for w in wave_grid if w in mapper])
                if present_waves.size:
                    loc = np.isin(wave_grid, present_waves)
                    y_grid[loc] = [mapper[w] for w in wave_grid[loc]]

            rows.append(y_grid)

        return np.vstack(rows) if rows else np.empty((0, len(wave_grid)))

    def _safe_nanmean_axis0(Y):
        Y = np.asarray(Y, dtype=float)
        out = np.full(Y.shape[1], np.nan, dtype=float)
        n = np.sum(np.isfinite(Y), axis=0)

        ok = n > 0
        if np.any(ok):
            out[ok] = np.nanmean(Y[:, ok], axis=0)

        return out, n

    def _safe_nanstd_axis0(Y):
        Y = np.asarray(Y, dtype=float)
        out = np.full(Y.shape[1], np.nan, dtype=float)
        n = np.sum(np.isfinite(Y), axis=0)

        ok = n > 1
        if np.any(ok):
            out[ok] = np.nanstd(Y[:, ok], axis=0, ddof=1)

        out[n == 1] = 0.0

        return out, n

    def _mean_err(Y):
        y_mean, n = _safe_nanmean_axis0(Y)
        y_std, _ = _safe_nanstd_axis0(Y)

        if err_mode.lower() == "std":
            y_err = y_std
        elif err_mode.lower() == "sem":
            y_err = y_std / np.sqrt(np.maximum(n, 1))
        else:
            raise ValueError("err_mode must be 'sem' or 'std'.")

        y_err = np.nan_to_num(y_err, nan=0.0, posinf=0.0, neginf=0.0)
        y_std = np.nan_to_num(y_std, nan=0.0, posinf=0.0, neginf=0.0)

        return y_mean, y_std, y_err, n

    fig, ax = plt.subplots(figsize=(10, 6))
    summary_rows = []

    for st in subtypes:
        ds = d[d["Subtype"] == st].copy()

        if ds.empty:
            continue

        if average_unit == "point":
            Y = _point_stack(ds)
            n_replicates = Y.shape[0]

        else:
            sample_means = []

            for sample, g_sample in ds.groupby("Sample", observed=True):
                Y_points = _point_stack(g_sample)

                if Y_points.size == 0:
                    continue

                sample_mean, sample_n = _safe_nanmean_axis0(Y_points)

                if np.any(np.isfinite(sample_mean)):
                    sample_means.append(sample_mean)

            Y = np.vstack(sample_means) if sample_means else np.empty((0, len(wave_grid)))
            n_replicates = Y.shape[0]

        if Y.size == 0:
            print(f"[Raman average] {st}: no usable spectra")
            continue

        y_mean, y_std, y_err, n = _mean_err(Y)

        colour = colours.get(st, None)
        linestyle = linestyles.get(st, "-")

        if show_error:
            ax.fill_between(
                wave_grid,
                y_mean - y_err,
                y_mean + y_err,
                color=colour,
                alpha=error_alpha,
                linewidth=0,
                zorder=1,
            )

        label = subtype_label(st) if "subtype_label" in globals() else st

        ax.plot(
            wave_grid,
            y_mean,
            color=colour,
            linestyle=linestyle,
            linewidth=linewidth,
            label=f"{label} (n={n_replicates})",
            zorder=3,
        )

        summary_rows.append(pd.DataFrame({
            "Subtype": st,
            wave_col: wave_grid,
            "mean": y_mean,
            "std": y_std,
            "err": y_err,
            "n": n,
            "n_replicates": n_replicates,
            "average_unit": average_unit,
        }))

    summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()

    _add_peak_region_lines(ax)

    ax.axhline(0, color="k", linewidth=1.0, alpha=0.4)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(False)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()

    return fig, ax, summary

def add_raman_bins(
    spectra,
    *,
    nbins=3,
    bin_col="Bin",
):
    """
    Add spatial bin labels to Raman point spectra.

    Binning is done within each Sample/Subtype line scan.
    Uses PointIndex order where available, otherwise x/y order.
    """

    d = spectra.copy()

    if d.empty:
        return d

    required = ["Sample", "Subtype", "PointIndex", "x", "y"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise KeyError(f"Missing required columns for Raman binning: {missing}")

    point_cols = ["Sample", "Subtype", "PointIndex", "x", "y"]

    points = (
        d[point_cols]
        .drop_duplicates()
        .copy()
    )

    for c in ["Sample", "PointIndex", "x", "y"]:
        points[c] = pd.to_numeric(points[c], errors="coerce")

    points["Subtype"] = points["Subtype"].astype(str).str.strip()

    bin_rows = []

    for (sample, subtype), g in points.groupby(["Sample", "Subtype"], observed=True):
        g = g.sort_values(["PointIndex", "x", "y"]).copy()
        n = len(g)

        if n == 0:
            continue

        # Assign approximately equal numbers of point spectra to each bin
        g[bin_col] = pd.cut(
            np.arange(n),
            bins=nbins,
            labels=np.arange(1, nbins + 1),
            include_lowest=True,
        ).astype(int)

        bin_rows.append(g)

    if not bin_rows:
        d[bin_col] = np.nan
        return d

    bin_df = pd.concat(bin_rows, ignore_index=True)

    d = d.merge(
        bin_df[point_cols + [bin_col]],
        on=point_cols,
        how="left",
    )

    return d

def plot_raman_binned_average_spectra(
    spectra,
    *,
    nbins=3,
    subtype_order=None,
    colours=None,
    linestyles=None,
    wave_col="WaveRound",
    intensity_col="Intensity",
    err_mode="sem",          # "sem" or "std"
    average_unit="sample",   # "sample" or "point"
    show_error=True,
    error_alpha=0.25,
    linewidth=2.0,
    xlim=(1200, 1800),
    ylim=None,
    interp_grid=True,
    ncols=1,
    panel_by="bin",          # "bin" or "subtype"
    bin_cmap="jet",
    title="Binned average Raman spectra by subtype",
):
    """
    Plot average Raman spectra split into spatial bins.

    panel_by="bin":
        Each panel = one spatial bin.
        Lines = subtype average spectra.

    panel_by="subtype":
        Each panel = one subtype.
        Lines = spatial bin average spectra.

    Error = SEM or STD using the same logic as plot_raman_average_spectra.

    average_unit:
        "point"  -> error across point spectra within each subtype/bin
        "sample" -> point spectra are averaged per sample/bin first,
                    then error is calculated across sample means
    """

    def _safe_nanmean_axis0(Y):
        Y = np.asarray(Y, dtype=float)

        if Y.size == 0:
            return np.array([]), np.array([])

        out = np.full(Y.shape[1], np.nan, dtype=float)
        n = np.sum(np.isfinite(Y), axis=0)

        ok = n > 0
        if np.any(ok):
            out[ok] = np.nanmean(Y[:, ok], axis=0)

        return out, n

    def _safe_nanstd_axis0(Y):
        Y = np.asarray(Y, dtype=float)

        if Y.size == 0:
            return np.array([]), np.array([])

        out = np.full(Y.shape[1], np.nan, dtype=float)
        n = np.sum(np.isfinite(Y), axis=0)

        ok = n > 1
        if np.any(ok):
            out[ok] = np.nanstd(Y[:, ok], axis=0, ddof=1)

        out[n == 1] = 0.0

        return out, n

    def _mean_err(Y):
        y_mean, n = _safe_nanmean_axis0(Y)
        y_std, _ = _safe_nanstd_axis0(Y)

        if err_mode.lower() == "std":
            y_err = y_std
        elif err_mode.lower() == "sem":
            y_err = y_std / np.sqrt(np.maximum(n, 1))
        else:
            raise ValueError("err_mode must be 'sem' or 'std'.")

        y_err = np.nan_to_num(y_err, nan=0.0, posinf=0.0, neginf=0.0)
        y_std = np.nan_to_num(y_std, nan=0.0, posinf=0.0, neginf=0.0)

        return y_mean, y_std, y_err, n

    def _add_bins(d):
        required = ["Sample", "Subtype", "PointIndex", "x", "y"]
        missing = [c for c in required if c not in d.columns]
        if missing:
            raise KeyError(f"Missing required columns for Raman binning: {missing}")

        point_cols = ["Sample", "Subtype", "PointIndex", "x", "y"]
        points = d[point_cols].drop_duplicates().copy()

        for c in ["Sample", "PointIndex", "x", "y"]:
            points[c] = pd.to_numeric(points[c], errors="coerce")

        points["Subtype"] = points["Subtype"].astype(str).str.strip()

        bin_rows = []

        for (sample, subtype), g in points.groupby(["Sample", "Subtype"], observed=True):
            g = g.sort_values(["PointIndex", "x", "y"]).copy()
            n = len(g)

            if n == 0:
                continue

            g["Bin"] = pd.cut(
                np.arange(n),
                bins=nbins,
                labels=np.arange(1, nbins + 1),
                include_lowest=True,
            ).astype(int)

            bin_rows.append(g)

        if not bin_rows:
            d["Bin"] = np.nan
            return d

        bin_df = pd.concat(bin_rows, ignore_index=True)

        return d.merge(
            bin_df[point_cols + ["Bin"]],
            on=point_cols,
            how="left",
        )

    def _point_stack(ds):
        rows = []

        for _, g in ds.groupby(point_id_cols, observed=True):
            g = g.sort_values(wave_col).copy()

            gg = (
                g.groupby(wave_col, as_index=False)[intensity_col]
                .mean()
                .sort_values(wave_col)
            )

            x = gg[wave_col].to_numpy(float)
            y = gg[intensity_col].to_numpy(float)

            ok = np.isfinite(x) & np.isfinite(y)
            x = x[ok]
            y = y[ok]

            if x.size < 2:
                continue

            xu, idx = np.unique(x, return_index=True)
            yu = y[idx]

            if interp_grid:
                y_grid = np.interp(wave_grid, xu, yu)
                y_grid[(wave_grid < xu.min()) | (wave_grid > xu.max())] = np.nan
            else:
                y_grid = np.full_like(wave_grid, np.nan, dtype=float)
                mapper = dict(zip(xu.astype(int), yu))
                present_waves = np.array([w for w in wave_grid if w in mapper])

                if present_waves.size:
                    loc = np.isin(wave_grid, present_waves)
                    y_grid[loc] = [mapper[w] for w in wave_grid[loc]]

            rows.append(y_grid)

        return np.vstack(rows) if rows else np.empty((0, len(wave_grid)))

    def _average_subset(ds):
        if ds.empty:
            return None

        if average_unit == "point":
            Y = _point_stack(ds)
            n_replicates = Y.shape[0]

        else:
            sample_means = []

            for _, g_sample in ds.groupby("Sample", observed=True):
                Y_points = _point_stack(g_sample)

                if Y_points.size == 0:
                    continue

                sample_mean, _ = _safe_nanmean_axis0(Y_points)

                if np.any(np.isfinite(sample_mean)):
                    sample_means.append(sample_mean)

            Y = np.vstack(sample_means) if sample_means else np.empty((0, len(wave_grid)))
            n_replicates = Y.shape[0]

        if Y.size == 0:
            return None

        y_mean, y_std, y_err, n = _mean_err(Y)

        return {
            "mean": y_mean,
            "std": y_std,
            "err": y_err,
            "n": n,
            "n_replicates": n_replicates,
        }

    colours = colours or {}
    linestyles = linestyles or {}

    if spectra is None or spectra.empty:
        print("No Raman spectra to average.")
        return None, None, pd.DataFrame()

    if average_unit not in {"point", "sample"}:
        raise ValueError("average_unit must be 'point' or 'sample'.")

    if panel_by not in {"bin", "subtype"}:
        raise ValueError("panel_by must be 'bin' or 'subtype'.")

    d = spectra.copy()
    d = _add_bins(d)

    d[wave_col] = pd.to_numeric(d[wave_col], errors="coerce")
    d[intensity_col] = pd.to_numeric(d[intensity_col], errors="coerce")
    d["Subtype"] = d["Subtype"].astype(str).str.strip()
    d["Bin"] = pd.to_numeric(d["Bin"], errors="coerce")

    d = d.dropna(subset=[wave_col, intensity_col, "Subtype", "Bin"]).copy()
    d["Bin"] = d["Bin"].astype(int)

    if xlim is not None:
        d = d[(d[wave_col] >= xlim[0]) & (d[wave_col] <= xlim[1])].copy()

    if d.empty:
        print("No Raman rows left after filtering/binning.")
        return None, None, pd.DataFrame()

    present = list(pd.unique(d["Subtype"]))

    if subtype_order is None:
        subtypes = present
    else:
        subtypes = [st for st in subtype_order if st in present]

    if not subtypes:
        print("No matching subtypes to plot.")
        return None, None, pd.DataFrame()

    point_id_cols = ["Sample", "Subtype", "PointIndex", "x", "y"]
    point_id_cols = [c for c in point_id_cols if c in d.columns]

    if xlim is not None:
        wave_grid = np.arange(int(xlim[0]), int(xlim[1]) + 1)
    else:
        wave_grid = np.arange(
            int(np.nanmin(d[wave_col])),
            int(np.nanmax(d[wave_col])) + 1,
        )

    summary_rows = []

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    if panel_by == "bin":
        panels = list(range(1, nbins + 1))
        n_panels = len(panels)
        panel_title = lambda b: f"Bin {b}"
    else:
        panels = subtypes
        n_panels = len(panels)
        panel_title = lambda st: str(st)

    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(10 * ncols, 4.2 * nrows),
        sharex=True,
        sharey=True,
    )

    axes = np.atleast_1d(axes).ravel()

    bin_cmap_obj = plt.get_cmap(bin_cmap, nbins)

    # ------------------------------------------------------------------
    # Plot: panels are bins, lines are subtypes
    # ------------------------------------------------------------------
    if panel_by == "bin":
        for ax, b in zip(axes, panels):
            dbin = d[d["Bin"] == b].copy()

            for st in subtypes:
                ds = dbin[dbin["Subtype"] == st].copy()
                result = _average_subset(ds)

                if result is None:
                    print(f"[Raman bin {b}] {st}: no usable spectra")
                    continue

                y_mean = result["mean"]
                y_std = result["std"]
                y_err = result["err"]
                n = result["n"]
                n_replicates = result["n_replicates"]

                colour = colours.get(st, None)
                linestyle = linestyles.get(st, "-")

                if show_error:
                    ax.fill_between(
                        wave_grid,
                        y_mean - y_err,
                        y_mean + y_err,
                        color=colour,
                        alpha=error_alpha,
                        linewidth=0,
                        zorder=1,
                    )

                ax.plot(
                    wave_grid,
                    y_mean,
                    color=colour,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=f"{st} (n={n_replicates})",
                    zorder=3,
                )

                summary_rows.append(pd.DataFrame({
                    "Subtype": st,
                    "Bin": b,
                    wave_col: wave_grid,
                    "mean": y_mean,
                    "std": y_std,
                    "err": y_err,
                    "n": n,
                    "n_replicates": n_replicates,
                    "average_unit": average_unit,
                }))

            ax.set_title(panel_title(b))
            ax.axhline(0, color="k", linewidth=1.0, alpha=0.4)
            ax.set_xlabel("Wavenumber (cm$^{-1}$)")
            ax.set_ylabel("Intensity")
            ax.grid(False)
            ax.legend(fontsize="small")

            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)

    # ------------------------------------------------------------------
    # Plot: panels are subtypes, lines are bins
    # ------------------------------------------------------------------
    else:
        for ax, st in zip(axes, panels):
            dsub = d[d["Subtype"] == st].copy()

            for b in range(1, nbins + 1):
                ds = dsub[dsub["Bin"] == b].copy()
                result = _average_subset(ds)

                if result is None:
                    print(f"[Raman {st}] Bin {b}: no usable spectra")
                    continue

                y_mean = result["mean"]
                y_std = result["std"]
                y_err = result["err"]
                n = result["n"]
                n_replicates = result["n_replicates"]

                colour = bin_cmap_obj(b - 1)
                linestyle = "-"

                if show_error:
                    ax.fill_between(
                        wave_grid,
                        y_mean - y_err,
                        y_mean + y_err,
                        color=colour,
                        alpha=error_alpha,
                        linewidth=0,
                        zorder=1,
                    )

                ax.plot(
                    wave_grid,
                    y_mean,
                    color=colour,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=f"Bin {b} (n={n_replicates})",
                    zorder=3,
                )

                summary_rows.append(pd.DataFrame({
                    "Subtype": st,
                    "Bin": b,
                    wave_col: wave_grid,
                    "mean": y_mean,
                    "std": y_std,
                    "err": y_err,
                    "n": n,
                    "n_replicates": n_replicates,
                    "average_unit": average_unit,
                }))

            ax.set_title(panel_title(st))
            ax.axhline(0, color="k", linewidth=1.0, alpha=0.4)
            ax.set_xlabel("Wavenumber (cm$^{-1}$)")
            ax.set_ylabel("Intensity")
            ax.grid(False)
            ax.legend(fontsize="small")

            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)

    for ax in axes[n_panels:]:
        ax.axis("off")

    summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()

    fig.suptitle(title, y=0.995)
    plt.tight_layout()
    plt.show()

    return fig, axes, summary

def tidy_raman_weighted_moments(
    raman_df,
    *,
    subtype_col="Subtype",
    sample_col="Sample",
    region_col="Region",
):
    """
    Convert wide Raman weighted moment table into long format.

    Output columns:
        Sample, Subtype, Region, PeakRegion, Metric, Value
    """

    d = raman_df.copy()
    d.columns = d.columns.astype(str).str.strip()

    id_cols = [c for c in [sample_col, subtype_col, "Technique", region_col, "RegionKey", "NpointsRegion", "ExportSubtype"] if c in d.columns]

    metric_suffixes = [
        "m1",
        "mu2",
        "mu3",
        "sigma",
        "skewness",
        "area_w",
        "max_intensity",
        "neg_area_frac",
        "n_points",
    ]

    rows = []

    metric_cols = [
        c for c in d.columns
        if any(c.endswith(f"_{suffix}") for suffix in metric_suffixes)
    ]

    for col in metric_cols:
        metric = None
        peak_region = None

        for suffix in metric_suffixes:
            end = f"_{suffix}"
            if col.endswith(end):
                metric = suffix
                peak_region = col[:-len(end)]
                break

        if metric is None:
            continue

        tmp = d[id_cols + [col]].copy()
        tmp = tmp.rename(columns={col: "Value"})
        tmp["PeakRegion"] = peak_region
        tmp["Metric"] = metric
        rows.append(tmp)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    out["Subtype"] = out[subtype_col].astype(str).str.strip()

    if sample_col in out.columns:
        out["Sample"] = out[sample_col].astype(str).str.strip()

    if region_col in out.columns:
        out["Region"] = out[region_col].astype(str).str.strip()

    return out

def plot_raman_weighted_moment_bars(
    raman_wm_long,
    *,
    metric="m1",
    peak_regions=None,
    subtype_order=("CT", "D7", "D10", "D14", "D21"),
    colours=None,
    ylabel=None,
    title=None,
    error_mode="sem",          # "sem" or "std"
    m1_normalise="none",       # "none", "control", "region_center", "region_start"
    control_subtype="CT",
    figsize=(12, 5.5),
    point_size=35,
    point_alpha=0.75,
    jitter=0.04,
    ylim=None,
):
    """
    Bar plot for one Raman metric across peak regions.

    x-axis = peak region
    grouped bars = subtype
    points = individual sample values

    For metric='m1', m1_normalise can be:
        "none"          -> raw peak centre
        "control"       -> subtract control mean within each peak region
        "region_center" -> subtract midpoint of peak-region bounds
        "region_start"  -> subtract lower bound of peak-region bounds
    """

    if raman_wm_long is None or raman_wm_long.empty:
        print("No Raman weighted moment data to plot.")
        return None, None, pd.DataFrame()

    colours = colours or {}

    allowed_norms = {"none", "control", "region_center", "region_start"}
    if m1_normalise not in allowed_norms:
        raise ValueError(f"m1_normalise must be one of {sorted(allowed_norms)}")

    d = raman_wm_long.copy()
    d["Subtype"] = d["Subtype"].astype(str).str.strip()
    d["Metric"] = d["Metric"].astype(str).str.strip()
    d["PeakRegion"] = d["PeakRegion"].astype(str).str.strip()
    d["Value"] = pd.to_numeric(d["Value"], errors="coerce")

    d = d[
        (d["Metric"] == metric)
        & d["Subtype"].isin(subtype_order)
        & np.isfinite(d["Value"])
    ].copy()

    peak_bounds = {}

    if peak_regions is not None:
        peak_names = []
        for p in peak_regions:
            if isinstance(p, tuple):
                peak_name, bounds = p
                peak_name = str(peak_name)
                peak_names.append(peak_name)
                peak_bounds[peak_name] = bounds
            else:
                peak_names.append(str(p))

        d = d[d["PeakRegion"].isin(peak_names)].copy()
        peak_regions = peak_names
    else:
        peak_regions = list(pd.unique(d["PeakRegion"]))

    if d.empty:
        print(f"No data found for metric={metric}.")
        return None, None, pd.DataFrame()

    # ------------------------------------------------------------
    # Optional WM1 normalisation / centring
    # ------------------------------------------------------------
    d["PlotValue"] = d["Value"]

    if metric == "m1" and m1_normalise != "none":

        if m1_normalise == "control":
            control_means = (
                d[d["Subtype"] == control_subtype]
                .groupby("PeakRegion")["Value"]
                .mean()
                .to_dict()
            )

            d["Baseline"] = d["PeakRegion"].map(control_means)
            d["PlotValue"] = d["Value"] - d["Baseline"]

        elif m1_normalise == "region_center":
            centres = {
                peak: (float(bounds[0]) + float(bounds[1])) / 2
                for peak, bounds in peak_bounds.items()
            }

            d["Baseline"] = d["PeakRegion"].map(centres)
            d["PlotValue"] = d["Value"] - d["Baseline"]

        elif m1_normalise == "region_start":
            starts = {
                peak: float(bounds[0])
                for peak, bounds in peak_bounds.items()
            }

            d["Baseline"] = d["PeakRegion"].map(starts)
            d["PlotValue"] = d["Value"] - d["Baseline"]

        d = d[np.isfinite(d["PlotValue"])].copy()

    if d.empty:
        print("No data left after m1 normalisation.")
        return None, None, pd.DataFrame()

    subtypes = [st for st in subtype_order if st in set(d["Subtype"])]
    peak_regions = [p for p in peak_regions if p in set(d["PeakRegion"])]

    rows = []

    for peak in peak_regions:
        for st in subtypes:
            vals = d.loc[
                (d["PeakRegion"] == peak) & (d["Subtype"] == st),
                "PlotValue"
            ].dropna().to_numpy(float)

            if vals.size == 0:
                continue

            mean = float(np.nanmean(vals))
            std = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0
            err = std if error_mode == "std" else std / np.sqrt(vals.size)

            rows.append({
                "PeakRegion": peak,
                "Subtype": st,
                "mean": mean,
                "std": std,
                "err": err,
                "n": vals.size,
            })

    summary = pd.DataFrame(rows)

    if summary.empty:
        print("No summary values to plot.")
        return None, None, summary

    x = np.arange(len(peak_regions))
    width = 0.82 / max(len(subtypes), 1)

    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(0)

    for j, st in enumerate(subtypes):
        offset = (j - (len(subtypes) - 1) / 2) * width

        s = (
            summary[summary["Subtype"] == st]
            .set_index("PeakRegion")
            .reindex(peak_regions)
            .reset_index()
        )

        means = pd.to_numeric(s["mean"], errors="coerce").to_numpy(float)
        errs = pd.to_numeric(s["err"], errors="coerce").to_numpy(float)

        xpos = x + offset
        colour = colours.get(st, None)

        ax.bar(
            xpos,
            means,
            width=width,
            color=colour,
            edgecolor="black",
            linewidth=0.6,
            label=st,
            zorder=2,
        )

        ax.errorbar(
            xpos,
            means,
            yerr=errs,
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3,
            zorder=3,
        )

        for i, peak in enumerate(peak_regions):
            vals = d.loc[
                (d["PeakRegion"] == peak) & (d["Subtype"] == st),
                "PlotValue"
            ].dropna().to_numpy(float)

            if vals.size == 0:
                continue

            xj = xpos[i] + rng.normal(0, jitter, size=vals.size)

            # ax.scatter(
            #     xj,
            #     vals,
            #     color="black",
            #     s=point_size,
            #     alpha=point_alpha,
            #     linewidths=0,
            #     zorder=4,
            # )

    ax.set_xticks(x)
    ax.set_xticklabels(peak_regions, rotation=35, ha="right")

    if ylabel is None:
        if metric == "m1" and m1_normalise == "control":
            ylabel = f"WM1 shift from {control_subtype} mean (cm$^{{-1}}$)"
        elif metric == "m1" and m1_normalise == "region_center":
            ylabel = "WM1 shift from peak-region centre (cm$^{-1}$)"
        elif metric == "m1" and m1_normalise == "region_start":
            ylabel = "WM1 position from peak-region start (cm$^{-1}$)"
        elif metric == "m1":
            ylabel = "Weighted moment 1 / peak centre (cm$^{-1}$)"
        else:
            ylabel = metric

    if title is None:
        title = f"Raman {metric} by peak region"

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(False)

    if metric == "m1" and m1_normalise in {"control", "region_center"}:
        ax.axhline(0, color="black", linewidth=1.0, alpha=0.6, zorder=1)

    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()

    return fig, ax, summary

def plot_raman_selected_weighted_moments(
    raman_wm_long,
    *,
    peak_regions,
    weighted_moments=("m1", "mu2"),
    all_peak_regions=RAMAN_PEAK_REGIONS,
    subtype_order=("CT", "D7", "D14"),
    colours=RAMAN_COLOURS,
    labels=None,
    error_mode="sem",
    normalise="control",      # "control", "zscore", "minmax", "none"
    control_subtype="CT",
    panel_by_peak=True,
    ncols=2,
    bar_alpha=0.85,
    capsize=3,
    ylim=None,
    title=None,
):
    """
    Plot selected Raman weighted moment metrics for selected peak regions.

    Parameters
    ----------
    raman_wm_long : pd.DataFrame
        Long-format Raman WM table from tidy_raman_weighted_moments().

        Expected columns can be either:
            Subtype, PeakRegion, Metric, Value
        or:
            Subtype, PeakRegion, m1, mu2, area_w, max_intensity, etc.

    peak_regions : list
        Peak regions to include.
        Can be:
            - names, e.g. ["AmideIII_1410_1500"]
            - 1-based indices from RAMAN_PEAK_REGIONS, e.g. [2, 3]

    weighted_moments : list
        Metrics to plot, e.g. ["m1", "mu2", "area_w"].

    normalise : str
        "control"
            subtracts CT/control mean for each PeakRegion + Metric.

        "zscore"
            converts values to z-score within each PeakRegion + Metric.

        "minmax"
            scales values 0–1 within each PeakRegion + Metric.

        "none"
            plots raw values. Only sensible when metrics have comparable scales.
    """

    def _resolve_peak_regions(peak_regions):
        region_names = [r[0] for r in all_peak_regions]
        out = []

        for r in peak_regions:
            if isinstance(r, int):
                idx = r - 1
                if idx < 0 or idx >= len(region_names):
                    raise ValueError(f"Peak region index {r} is out of range.")
                out.append(region_names[idx])
            else:
                r = str(r).strip()
                if r not in region_names:
                    raise ValueError(
                        f"Peak region '{r}' not found. Available: {region_names}"
                    )
                out.append(r)

        return out

    def _short_peak_label(name):
        name = str(name)
        if name.startswith("CH2CH3_LEFT"):
            return "CH2/CH3 left"
        if name.startswith("CH2CH3_RIGHT"):
            return "CH2/CH3 right"
        if name.startswith("AmideIII"):
            return "Amide III"
        if name.startswith("AmideI_LEFT"):
            return "Amide I left"
        if name.startswith("AmideI_MIDDLE"):
            return "Amide I middle"
        if name.startswith("AmideI_RIGHT"):
            return "Amide I right"
        return name.split("_")[0]

    def _metric_label(metric):
        lab = {
            "m1": "WM1 / peak centre",
            "mu2": "WM2 / spread",
            "mu3": "WM3",
            "sigma": "Sigma",
            "skewness": "Skewness",
            "area_w": "Weighted area",
            "max_intensity": "Peak height",
        }
        return lab.get(metric, metric)

    def _to_metric_long(df):
        d = df.copy()

        subtype_col = None
        for c in ["Subtype", "SubtypeClean", "ExportSubtype"]:
            if c in d.columns:
                subtype_col = c
                break

        if subtype_col is None:
            raise KeyError("Could not find a subtype column.")

        if "PeakRegion" not in d.columns:
            raise KeyError("raman_wm_long must contain 'PeakRegion'.")

        if {"Metric", "Value"}.issubset(d.columns):
            out = d[[subtype_col, "PeakRegion", "Metric", "Value"]].copy()
            out = out.rename(columns={subtype_col: "Subtype"})
            return out

        metric_cols = [m for m in weighted_moments if m in d.columns]

        if not metric_cols:
            raise KeyError(
                "Could not find Metric/Value columns or any selected metric columns "
                f"from {weighted_moments}."
            )

        out = d.melt(
            id_vars=[subtype_col, "PeakRegion"],
            value_vars=metric_cols,
            var_name="Metric",
            value_name="Value",
        ).rename(columns={subtype_col: "Subtype"})

        return out

    def _normalise_values(df):
        d = df.copy()
        d["Value"] = pd.to_numeric(d["Value"], errors="coerce")

        if normalise == "none":
            d["PlotValue"] = d["Value"]
            return d, "Raw value"

        if normalise == "control":
            ctrl = (
                d[d["Subtype"].astype(str) == str(control_subtype)]
                .groupby(["PeakRegion", "Metric"], as_index=False)["Value"]
                .mean()
                .rename(columns={"Value": "ControlMean"})
            )

            d = d.merge(ctrl, on=["PeakRegion", "Metric"], how="left")
            d["PlotValue"] = d["Value"] - d["ControlMean"]
            return d, f"Shift from {control_subtype} mean"

        if normalise == "zscore":
            stats = (
                d.groupby(["PeakRegion", "Metric"], as_index=False)["Value"]
                .agg(mu="mean", sd="std")
            )

            d = d.merge(stats, on=["PeakRegion", "Metric"], how="left")
            d["sd"] = d["sd"].replace(0, np.nan)
            d["PlotValue"] = (d["Value"] - d["mu"]) / d["sd"]
            return d, "Normalised value (z-score)"

        if normalise == "minmax":
            stats = (
                d.groupby(["PeakRegion", "Metric"], as_index=False)["Value"]
                .agg(vmin="min", vmax="max")
            )

            d = d.merge(stats, on=["PeakRegion", "Metric"], how="left")
            rng = d["vmax"] - d["vmin"]
            d["PlotValue"] = np.where(
                rng != 0,
                (d["Value"] - d["vmin"]) / rng,
                np.nan,
            )
            return d, "Normalised value (0–1)"

        raise ValueError("normalise must be 'control', 'zscore', 'minmax', or 'none'.")

    if raman_wm_long is None or raman_wm_long.empty:
        print("No Raman weighted moment data to plot.")
        return None, None, pd.DataFrame()

    labels = labels or {}
    selected_peaks = _resolve_peak_regions(peak_regions)
    selected_metrics = [str(m).strip() for m in weighted_moments]

    d = _to_metric_long(raman_wm_long)
    d["Subtype"] = d["Subtype"].astype(str).str.strip()
    d["PeakRegion"] = d["PeakRegion"].astype(str).str.strip()
    d["Metric"] = d["Metric"].astype(str).str.strip()

    d = d[
        d["PeakRegion"].isin(selected_peaks)
        & d["Metric"].isin(selected_metrics)
        & d["Subtype"].isin(subtype_order)
    ].copy()

    if d.empty:
        print("No matching Raman weighted moment rows after filtering.")
        return None, None, pd.DataFrame()

    d, y_label = _normalise_values(d)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["PlotValue"]).copy()

    summary = (
        d.groupby(["PeakRegion", "Metric", "Subtype"], as_index=False)["PlotValue"]
        .agg(mean="mean", std="std", n="count")
    )

    summary["sem"] = summary["std"] / np.sqrt(summary["n"].clip(lower=1))
    summary["err"] = summary["sem"] if error_mode.lower() == "sem" else summary["std"]

    if panel_by_peak:
        n_panels = len(selected_peaks)
        nrows = int(np.ceil(n_panels / ncols))

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(6.2 * ncols, 4.8 * nrows),
            sharey=True,
        )

        axes = np.atleast_1d(axes).ravel()

        for ax, peak in zip(axes, selected_peaks):
            s_peak = summary[summary["PeakRegion"] == peak].copy()

            x = np.arange(len(selected_metrics))
            width = 0.8 / max(len(subtype_order), 1)

            for i, subtype in enumerate(subtype_order):
                s = (
                    s_peak[s_peak["Subtype"] == subtype]
                    .set_index("Metric")
                    .reindex(selected_metrics)
                    .reset_index()
                )

                xpos = x + (i - (len(subtype_order) - 1) / 2) * width

                y = s["mean"].to_numpy(float)
                err = s["err"].to_numpy(float)

                ax.bar(
                    xpos,
                    y,
                    width=width,
                    color=colours.get(subtype, "grey"),
                    alpha=bar_alpha,
                    edgecolor="black",
                    linewidth=0.6,
                    label=labels.get(subtype, subtype),
                )

                ax.errorbar(
                    xpos,
                    y,
                    yerr=err,
                    fmt="none",
                    ecolor="black",
                    capsize=capsize,
                    linewidth=1,
                )

            ax.axhline(0, color="black", linewidth=1, alpha=0.45)
            ax.set_xticks(x)
            ax.set_xticklabels([_metric_label(m) for m in selected_metrics], rotation=30, ha="right")
            ax.set_title(_short_peak_label(peak))
            ax.set_ylabel(y_label)
            ax.grid(False)

            if ylim is not None:
                ax.set_ylim(*ylim)

        for ax in axes[len(selected_peaks):]:
            ax.axis("off")

        handles, leg_labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            leg_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(subtype_order),
            frameon=False,
        )

        fig.suptitle(
            title or "Selected Raman weighted moments",
            y=0.995,
        )

        plt.tight_layout(rect=[0, 0.07, 1, 0.96])
        plt.show()

        return fig, axes, summary

    # Combined plot: each peak/metric combination becomes an x-category
    plot_order = [
        (peak, metric)
        for peak in selected_peaks
        for metric in selected_metrics
    ]

    xlabels = [
        f"{_short_peak_label(peak)}\n{_metric_label(metric)}"
        for peak, metric in plot_order
    ]

    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(xlabels)), 5.5))

    x = np.arange(len(plot_order))
    width = 0.8 / max(len(subtype_order), 1)

    for i, subtype in enumerate(subtype_order):
        rows = []

        for peak, metric in plot_order:
            row = summary[
                (summary["PeakRegion"] == peak)
                & (summary["Metric"] == metric)
                & (summary["Subtype"] == subtype)
            ]

            if row.empty:
                rows.append({"mean": np.nan, "err": np.nan})
            else:
                rows.append({
                    "mean": row["mean"].iloc[0],
                    "err": row["err"].iloc[0],
                })

        y = np.asarray([r["mean"] for r in rows], dtype=float)
        err = np.asarray([r["err"] for r in rows], dtype=float)

        xpos = x + (i - (len(subtype_order) - 1) / 2) * width

        ax.bar(
            xpos,
            y,
            width=width,
            color=colours.get(subtype, "grey"),
            alpha=bar_alpha,
            edgecolor="black",
            linewidth=0.6,
            label=labels.get(subtype, subtype),
        )

        ax.errorbar(
            xpos,
            y,
            yerr=err,
            fmt="none",
            ecolor="black",
            capsize=capsize,
            linewidth=1,
        )

    ax.axhline(0, color="black", linewidth=1, alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=0, ha="right")
    ax.set_ylabel(y_label)
    ax.set_title(title or "Selected Raman weighted moments")
    ax.legend(frameon=False)
    ax.grid(False)

    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()

    return fig, ax, summary

def calculate_raman_amide_ratios(
    raman_df,
    *,
    subtype_order=("CT", "D7", "D10", "D14", "D21"),
    amideI_regions=(
        "AmideI_LEFT_1530_1590",
        "AmideI_MIDDLE_1590_1635",
        "AmideI_RIGHT_1635_1700",
    ),
    amideIII_region="AmideIII_1410_1500",
    height_metric="max_intensity",
    area_metric="area_w",
    amideI_height_mode="sum",  # "sum" or "max"
):
    """
    Calculate sample-level Amide I / Amide III ratios.

    Returns one row per sample with:
        AmideI_height
        AmideIII_height
        AmideI_III_height_ratio
        AmideI_area
        AmideIII_area
        AmideI_III_area_ratio
    """

    d = raman_df.copy()
    d.columns = d.columns.astype(str).str.strip()

    required = ["Sample", "Subtype"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise KeyError(f"Missing required Raman columns: {missing}")

    d["Subtype"] = d["Subtype"].astype(str).str.strip()
    d = d[d["Subtype"].isin(subtype_order)].copy()

    if d.empty:
        print("No Raman rows found for requested subtypes.")
        return pd.DataFrame()

    h_cols_I = [f"{r}_{height_metric}" for r in amideI_regions]
    a_cols_I = [f"{r}_{area_metric}" for r in amideI_regions]

    h_col_III = f"{amideIII_region}_{height_metric}"
    a_col_III = f"{amideIII_region}_{area_metric}"

    needed = h_cols_I + a_cols_I + [h_col_III, a_col_III]
    missing = [c for c in needed if c not in d.columns]

    if missing:
        print("[calculate_raman_amide_ratios] Missing columns:")
        for c in missing:
            print(f"  - {c}")
        return pd.DataFrame()

    for c in needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    if amideI_height_mode == "sum":
        d["AmideI_height"] = d[h_cols_I].sum(axis=1, min_count=1)
    elif amideI_height_mode == "max":
        d["AmideI_height"] = d[h_cols_I].max(axis=1)
    else:
        raise ValueError("amideI_height_mode must be 'sum' or 'max'.")

    d["AmideIII_height"] = d[h_col_III]
    d["AmideI_area"] = d[a_cols_I].sum(axis=1, min_count=1)
    d["AmideIII_area"] = d[a_col_III]

    d["AmideI_III_height_ratio"] = d["AmideI_height"] / d["AmideIII_height"]
    d["AmideI_III_area_ratio"] = d["AmideI_area"] / d["AmideIII_area"]

    d = d.replace([np.inf, -np.inf], np.nan)

    keep_cols = [
        c for c in [
            "Sample",
            "Subtype",
            "Technique",
            "Region",
            "RegionKey",
            "NpointsRegion",
            "AmideI_height",
            "AmideIII_height",
            "AmideI_III_height_ratio",
            "AmideI_area",
            "AmideIII_area",
            "AmideI_III_area_ratio",
        ]
        if c in d.columns
    ]

    return d[keep_cols].copy()

def plot_raman_amide_ratios(
    raman_df,
    *,
    subtype_order=("CT", "D7", "D10", "D14", "D21"),
    colours=None,
    error_mode="sem",  # "sem" or "std"
    amideI_height_mode="sum",
    figsize=(10, 5),
    ylim_height=None,
    ylim_area=None,
    point_size=35,
    point_alpha=0.75,
    jitter=0.045,
    print_summary=True,
):
    """
    Plot Amide I / Amide III ratios:
        1) height ratio
        2) area ratio
    """

    ratios = calculate_raman_amide_ratios(
        raman_df,
        subtype_order=subtype_order,
        amideI_height_mode=amideI_height_mode,
    )

    if ratios.empty:
        print("No Amide ratio data to plot.")
        return None, None, pd.DataFrame(), pd.DataFrame()

    colours = colours or {}

    metric_info = {
        "AmideI_III_height_ratio": {
            "title": " ",
            # "title": "Amide I / Amide III height ratio",
            "ylabel": "Height ratio",
            "ylim": ylim_height,
        },
        "AmideI_III_area_ratio": {
            "title": "Amide I / Amide III area ratio",
            "ylabel": "Area ratio",
            "ylim": ylim_area,
        },
    }

    summary_rows = []

    for metric in metric_info:
        for st in subtype_order:
            vals = ratios.loc[
                ratios["Subtype"].astype(str) == str(st),
                metric
            ].dropna().to_numpy(float)

            if vals.size == 0:
                continue

            mean = float(np.nanmean(vals))
            std = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0
            err = std if error_mode == "std" else std / np.sqrt(vals.size)

            summary_rows.append({
                "Metric": metric,
                "Subtype": st,
                "mean": mean,
                "std": std,
                "err": err,
                "n": vals.size,
            })

    summary = pd.DataFrame(summary_rows)

    if summary.empty:
        print("No summary values to plot.")
        return None, None, ratios, summary

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    axes = np.atleast_1d(axes)

    rng = np.random.default_rng(0)

    for ax, metric in zip(axes, metric_info):
        info = metric_info[metric]

        present_subtypes = [
            st for st in subtype_order
            if st in set(summary.loc[summary["Metric"] == metric, "Subtype"].astype(str))
        ]

        x = np.arange(len(present_subtypes))

        s = (
            summary[summary["Metric"] == metric]
            .set_index("Subtype")
            .reindex(present_subtypes)
            .reset_index()
        )

        y = pd.to_numeric(s["mean"], errors="coerce").to_numpy(float)
        yerr = pd.to_numeric(s["err"], errors="coerce").to_numpy(float)

        bar_colours = [colours.get(st, None) for st in present_subtypes]

        ax.bar(
            x,
            y,
            color=bar_colours,
            edgecolor="black",
            linewidth=0.6,
            zorder=2,
        )

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3,
            zorder=3,
        )

        for i, st in enumerate(present_subtypes):
            vals = ratios.loc[
                ratios["Subtype"].astype(str) == str(st),
                metric
            ].dropna().to_numpy(float)

            if vals.size == 0:
                continue

            xj = x[i] + rng.normal(0, jitter, size=vals.size)

            # ax.scatter(
            #     xj,
            #     vals,
            #     color="black",
            #     s=point_size,
            #     alpha=point_alpha,
            #     linewidths=0,
            #     zorder=4,
            # )

        ax.set_xticks(x)
        ax.set_xticklabels(present_subtypes, rotation=0, ha="right")
        ax.set_ylabel(info["ylabel"])
        ax.set_title(info["title"])
        ax.grid(False)
        ax.set_ylim(0.8,1.1)

        if info["ylim"] is not None:
            ax.set_ylim(*info["ylim"])

    plt.tight_layout()
    plt.show()

    # if print_summary:
    #     print("\n[Amide I / Amide III ratios]")
    #     print(summary.to_string(index=False))

    #     print("\n[Sample-level ratios]")
    #     print(
    #         ratios[
    #             [
    #                 "Sample",
    #                 "Subtype",
    #                 "AmideI_height",
    #                 "AmideIII_height",
    #                 "AmideI_III_height_ratio",
    #                 "AmideI_area",
    #                 "AmideIII_area",
    #                 "AmideI_III_area_ratio",
    #             ]
    #         ].to_string(index=False)
    #     )

    return fig, axes, ratios, summary

def _raman_weighted_moments_from_spectrum(x, y, xlim):
    """
    Calculate weighted moments and peak metrics from one averaged spectrum
    within one peak region.
    """

    x = np.asarray(x, float)
    y = np.asarray(y, float)

    lo, hi = float(xlim[0]), float(xlim[1])
    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= lo)
        & (x <= hi)
    )

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

    if xr.size < 2:
        return {
            "m1": np.nan,
            "mu2": np.nan,
            "mu3": np.nan,
            "sigma": np.nan,
            "skewness": np.nan,
            "area_w": np.nan,
            "max_intensity": np.nanmax(yr) if yr.size else np.nan,
            "neg_area_frac": np.nan,
            "n_points": int(xr.size),
        }

    total_abs_area = float(np.trapezoid(np.abs(yr), xr))
    neg_area = float(np.trapezoid(np.clip(-yr, 0, None), xr))
    neg_area_frac = neg_area / total_abs_area if total_abs_area > 0 else np.nan

    offset = -min(0.0, float(np.nanmin(yr)))
    w = np.clip(yr + offset, 0, None)
    wsum = float(np.nansum(w))

    max_intensity = float(np.nanmax(yr))
    area_w = float(np.trapezoid(w, xr))

    if not np.isfinite(wsum) or wsum <= 0:
        return {
            "m1": np.nan,
            "mu2": np.nan,
            "mu3": np.nan,
            "sigma": np.nan,
            "skewness": np.nan,
            "area_w": area_w,
            "max_intensity": max_intensity,
            "neg_area_frac": neg_area_frac,
            "n_points": int(xr.size),
        }

    m1 = float(np.nansum(w * xr) / wsum)
    mu2 = float(np.nansum(w * (xr - m1) ** 2) / wsum)
    mu3 = float(np.nansum(w * (xr - m1) ** 3) / wsum)

    sigma = float(np.sqrt(mu2)) if np.isfinite(mu2) and mu2 >= 0 else np.nan
    skewness = float(mu3 / (mu2 ** 1.5)) if np.isfinite(mu2) and mu2 > 0 and np.isfinite(mu3) else np.nan

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

def calculate_raman_binned_weighted_moments(
    raman_binned_avg,
    *,
    peak_regions,
    subtype_col=None,
    bin_col=None,
    wave_col=None,
    mean_col=None,
):
    """
    Calculate weighted moments from binned averaged Raman spectra.

    Auto-detects common column names from raman_binned_avg.

    Expected columns should include equivalents of:
        Subtype, Bin, Wave/WaveRound, mean/Mean
    """

    if raman_binned_avg is None or raman_binned_avg.empty:
        print("No binned Raman spectra provided.")
        return pd.DataFrame()

    d = raman_binned_avg.copy()
    d.columns = d.columns.astype(str).str.strip()

    def _pick_col(candidates, label):
        for c in candidates:
            if c in d.columns:
                return c
        raise KeyError(
            f"Could not find {label} column. Tried {candidates}. "
            f"Available columns are: {list(d.columns)}"
        )

    subtype_col = subtype_col or _pick_col(
        ["Subtype", "SubtypeClean", "subtype"],
        "subtype",
    )

    bin_col = bin_col or _pick_col(
        ["Bin", "bin", "BinNumber", "BinIndex"],
        "bin",
    )

    wave_col = wave_col or _pick_col(
        ["Wave", "WaveRound", "Wavenumber", "wavenumber", "wave"],
        "wave/wavenumber",
    )

    mean_col = mean_col or _pick_col(
        ["mean", "Mean", "MeanSpectrum", "IntensityMean", "Intensity", "y_mean"],
        "mean intensity",
    )

    # print(
    #     "[Raman binned WM] Using columns:",
    #     f"subtype={subtype_col}, bin={bin_col}, wave={wave_col}, mean={mean_col}"
    # )

    d[subtype_col] = d[subtype_col].astype(str).str.strip()
    d[bin_col] = pd.to_numeric(d[bin_col], errors="coerce")
    d[wave_col] = pd.to_numeric(d[wave_col], errors="coerce")
    d[mean_col] = pd.to_numeric(d[mean_col], errors="coerce")

    d = d.dropna(subset=[subtype_col, bin_col, wave_col, mean_col]).copy()

    rows = []

    for (subtype, bin_id), g in d.groupby([subtype_col, bin_col], observed=True):
        g = g.sort_values(wave_col)

        x = g[wave_col].to_numpy(float)
        y = g[mean_col].to_numpy(float)

        for peak_name, bounds in peak_regions:
            moms = _raman_weighted_moments_from_spectrum(x, y, bounds)

            for metric, value in moms.items():
                rows.append({
                    "Subtype": str(subtype),
                    "Bin": int(bin_id),
                    "PeakRegion": peak_name,
                    "Metric": metric,
                    "Value": value,
                })

    out = pd.DataFrame(rows)

    if out.empty:
        print("No binned weighted moments calculated.")
    # else:
        # print(f"[Raman binned WM] Calculated {len(out)} rows.")

    return out

def plot_raman_binned_weighted_moment_bars(
    raman_binned_wm_long,
    *,
    metric="m1",
    peak_regions=None,
    subtype_order=("CT", "D7", "D14"),
    bin_order=None,
    colours=None,
    ylabel=None,
    title=None,
    m1_normalise="none",      # "none", "control", "region_center", "region_start"
    control_subtype="CT",
    peak_region_bounds=None,
    figsize=(13, 5.8),
    ylim=None,
    capsize=3,
):
    """
    Plot binned Raman weighted moments.

    x-axis = peak region
    grouped bars = subtype/bin combinations

    For metric='m1', m1_normalise can be:
        "none"
        "control"       -> subtract CT value for the same peak region and bin
        "region_center" -> subtract midpoint of peak-region bounds
        "region_start"  -> subtract lower bound
    """

    if raman_binned_wm_long is None or raman_binned_wm_long.empty:
        print("No binned Raman weighted moment data to plot.")
        return None, None, pd.DataFrame()

    colours = colours or {}

    d = raman_binned_wm_long.copy()
    d["Subtype"] = d["Subtype"].astype(str).str.strip()
    d["PeakRegion"] = d["PeakRegion"].astype(str).str.strip()
    d["Metric"] = d["Metric"].astype(str).str.strip()
    d["Bin"] = pd.to_numeric(d["Bin"], errors="coerce")
    d["Value"] = pd.to_numeric(d["Value"], errors="coerce")

    d = d[
        (d["Metric"] == metric)
        & d["Subtype"].isin(subtype_order)
        & np.isfinite(d["Bin"])
        & np.isfinite(d["Value"])
    ].copy()

    d["Bin"] = d["Bin"].astype(int)

    if peak_regions is not None:
        peak_names = [p[0] if isinstance(p, tuple) else str(p) for p in peak_regions]
        d = d[d["PeakRegion"].isin(peak_names)].copy()
    else:
        peak_names = list(pd.unique(d["PeakRegion"]))

    if d.empty:
        print(f"No binned Raman data found for metric={metric}.")
        return None, None, pd.DataFrame()

    if bin_order is None:
        bin_order = sorted(d["Bin"].dropna().unique().tolist())

    subtypes = [st for st in subtype_order if st in set(d["Subtype"])]
    peak_names = [p for p in peak_names if p in set(d["PeakRegion"])]

    if peak_region_bounds is None:
        peak_region_bounds = {}
        if peak_regions is not None:
            for p in peak_regions:
                if isinstance(p, tuple):
                    peak_region_bounds[str(p[0])] = p[1]

    d["PlotValue"] = d["Value"]

    if metric == "m1" and m1_normalise != "none":

        if m1_normalise == "control":
            baseline = (
                d[d["Subtype"] == control_subtype]
                .groupby(["PeakRegion", "Bin"])["Value"]
                .mean()
                .rename("Baseline")
                .reset_index()
            )

            d = d.merge(baseline, on=["PeakRegion", "Bin"], how="left")
            d["PlotValue"] = d["Value"] - d["Baseline"]

        elif m1_normalise == "region_center":
            centres = {
                peak: (float(bounds[0]) + float(bounds[1])) / 2
                for peak, bounds in peak_region_bounds.items()
            }
            d["Baseline"] = d["PeakRegion"].map(centres)
            d["PlotValue"] = d["Value"] - d["Baseline"]

        elif m1_normalise == "region_start":
            starts = {
                peak: float(bounds[0])
                for peak, bounds in peak_region_bounds.items()
            }
            d["Baseline"] = d["PeakRegion"].map(starts)
            d["PlotValue"] = d["Value"] - d["Baseline"]

        else:
            raise ValueError("m1_normalise must be 'none', 'control', 'region_center', or 'region_start'.")

        d = d[np.isfinite(d["PlotValue"])].copy()

    # No error bars here unless we have per-sample binned spectra.
    # These values are currently calculated from the already averaged binned spectrum.
    summary = d.copy()

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(peak_names))
    n_bars = len(subtypes) * len(bin_order)
    width = 0.86 / max(n_bars, 1)

    handles = []

    for si, subtype in enumerate(subtypes):
        for bi, bin_id in enumerate(bin_order):
            j = si * len(bin_order) + bi
            offset = (j - (n_bars - 1) / 2) * width

            vals = []
            for peak in peak_names:
                row = summary[
                    (summary["Subtype"] == subtype)
                    & (summary["Bin"] == bin_id)
                    & (summary["PeakRegion"] == peak)
                ]

                vals.append(float(row["PlotValue"].iloc[0]) if not row.empty else np.nan)

            vals = np.asarray(vals, dtype=float)

            base_colour = colours.get(subtype, None)
            alpha = 0.45 + 0.45 * ((bi + 1) / max(len(bin_order), 1))

            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                color=base_colour,
                alpha=alpha,
                edgecolor="black",
                linewidth=0.5,
                label=f"{subtype} bin {bin_id}",
                zorder=2,
            )

            handles.append(bars[0])

    ax.set_xticks(x)
    ax.set_xticklabels(peak_names, rotation=35, ha="right")

    if ylabel is None:
        if metric == "m1" and m1_normalise == "control":
            ylabel = f"WM1 shift from {control_subtype} bin-matched value (cm$^{{-1}}$)"
        elif metric == "m1" and m1_normalise == "region_center":
            ylabel = "WM1 shift from peak-region centre (cm$^{-1}$)"
        elif metric == "m1" and m1_normalise == "region_start":
            ylabel = "WM1 position from peak-region start (cm$^{-1}$)"
        elif metric == "m1":
            ylabel = "WM1 / peak centre (cm$^{-1}$)"
        else:
            ylabel = metric

    if title is None:
        title = f"Binned Raman {metric}"

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)

    if metric == "m1" and m1_normalise in {"control", "region_center"}:
        ax.axhline(0, color="black", linewidth=1, alpha=0.7, zorder=1)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.legend(frameon=False, fontsize="x-small", ncol=min(n_bars, 6))
    plt.tight_layout()
    plt.show()

    return fig, ax, summary

def calculate_raman_binned_amide_ratios(
    raman_binned_wm_long,
    *,
    subtype_order=("CT", "D7", "D14"),
    amideI_regions=(
        "AmideI_LEFT_1530_1590",
        "AmideI_MIDDLE_1590_1635",
        "AmideI_RIGHT_1635_1700",
    ),
    amideIII_region="AmideIII_1410_1500",
    amideI_height_mode="sum",  # "sum" or "max"
):
    """
    Calculate Amide I / Amide III height and area ratios from binned WM table.
    """

    d = raman_binned_wm_long.copy()
    d["Subtype"] = d["Subtype"].astype(str).str.strip()
    d["PeakRegion"] = d["PeakRegion"].astype(str).str.strip()
    d["Metric"] = d["Metric"].astype(str).str.strip()
    d["Bin"] = pd.to_numeric(d["Bin"], errors="coerce")
    d["Value"] = pd.to_numeric(d["Value"], errors="coerce")

    d = d[
        d["Subtype"].isin(subtype_order)
        & np.isfinite(d["Bin"])
        & np.isfinite(d["Value"])
    ].copy()

    d["Bin"] = d["Bin"].astype(int)

    wide = (
        d.pivot_table(
            index=["Subtype", "Bin", "PeakRegion"],
            columns="Metric",
            values="Value",
            aggfunc="first",
        )
        .reset_index()
    )

    rows = []

    for (subtype, bin_id), g in wide.groupby(["Subtype", "Bin"], observed=True):
        g = g.copy()

        amideI = g[g["PeakRegion"].isin(amideI_regions)]
        amideIII = g[g["PeakRegion"] == amideIII_region]

        if amideI.empty or amideIII.empty:
            continue

        if amideI_height_mode == "sum":
            amideI_height = amideI["max_intensity"].sum()
        elif amideI_height_mode == "max":
            amideI_height = amideI["max_intensity"].max()
        else:
            raise ValueError("amideI_height_mode must be 'sum' or 'max'.")

        amideIII_height = float(amideIII["max_intensity"].iloc[0])

        amideI_area = amideI["area_w"].sum()
        amideIII_area = float(amideIII["area_w"].iloc[0])

        rows.append({
            "Subtype": subtype,
            "Bin": int(bin_id),
            "AmideI_height": amideI_height,
            "AmideIII_height": amideIII_height,
            "AmideI_III_height_ratio": amideI_height / amideIII_height if amideIII_height != 0 else np.nan,
            "AmideI_area": amideI_area,
            "AmideIII_area": amideIII_area,
            "AmideI_III_area_ratio": amideI_area / amideIII_area if amideIII_area != 0 else np.nan,
        })

    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)

def plot_raman_binned_amide_ratios(
    raman_binned_wm_long,
    *,
    subtype_order=("CT", "D7", "D14"),
    bin_order=None,
    colours=None,
    amideI_height_mode="sum",
    figsize=(11, 5),
    ylim_height=None,
    ylim_area=None,
):
    """
    Plot binned Amide I / Amide III ratios.
    Bars are split by bin for each subtype.
    """

    ratios = calculate_raman_binned_amide_ratios(
        raman_binned_wm_long,
        subtype_order=subtype_order,
        amideI_height_mode=amideI_height_mode,
    )

    if ratios.empty:
        print("No binned Amide ratio data to plot.")
        return None, None, ratios

    colours = colours or {}

    if bin_order is None:
        bin_order = sorted(ratios["Bin"].dropna().unique().tolist())

    subtypes = [st for st in subtype_order if st in set(ratios["Subtype"])]

    metrics = [
        ("AmideI_III_height_ratio", "Height ratio", "Binned Amide I / Amide III height ratio", ylim_height),
        ("AmideI_III_area_ratio", "Area ratio", "Binned Amide I / Amide III area ratio", ylim_area),
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    axes = np.atleast_1d(axes)

    x = np.arange(len(subtypes))
    width = 0.82 / max(len(bin_order), 1)

    for ax, (metric, ylabel, title, ylim) in zip(axes, metrics):
        for bi, bin_id in enumerate(bin_order):
            offset = (bi - (len(bin_order) - 1) / 2) * width

            vals = []
            for subtype in subtypes:
                row = ratios[
                    (ratios["Subtype"] == subtype)
                    & (ratios["Bin"] == bin_id)
                ]

                vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)

            vals = np.asarray(vals, dtype=float)
            bar_colours = [colours.get(st, None) for st in subtypes]
            alpha = 0.45 + 0.45 * ((bi + 1) / max(len(bin_order), 1))

            ax.bar(
                x + offset,
                vals,
                width=width,
                color=bar_colours,
                alpha=alpha,
                edgecolor="black",
                linewidth=0.6,
                label=f"Bin {bin_id}",
                zorder=2,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(subtypes, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(False)
        ax.legend(frameon=False)

        if ylim is not None:
            ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()

    return fig, axes, ratios

def _raman_spectra_to_matrix(
    spectra,
    *,
    wave_col="WaveRound",
    intensity_col="Intensity",
    xlim=(1200, 1800),
    id_cols=("Sample", "Subtype", "PointIndex", "x", "y"),
    interp_grid=True,
):
    """
    Convert Raman point spectra into a PCA-ready matrix.

    Rows = individual point spectra
    Columns = rounded wavenumbers
    """

    if spectra is None or spectra.empty:
        return pd.DataFrame(), np.array([]), np.empty((0, 0))

    d = spectra.copy()
    d[wave_col] = pd.to_numeric(d[wave_col], errors="coerce")
    d[intensity_col] = pd.to_numeric(d[intensity_col], errors="coerce")
    d["Subtype"] = d["Subtype"].astype(str).str.strip()

    d = d.dropna(subset=[wave_col, intensity_col, "Subtype"]).copy()

    if xlim is not None:
        d = d[(d[wave_col] >= xlim[0]) & (d[wave_col] <= xlim[1])].copy()

    if d.empty:
        return pd.DataFrame(), np.array([]), np.empty((0, 0))

    id_cols = [c for c in id_cols if c in d.columns]

    wave_grid = np.arange(
        int(np.nanmin(d[wave_col])),
        int(np.nanmax(d[wave_col])) + 1,
    )

    rows = []
    Y = []

    for keys, g in d.groupby(id_cols, observed=True):
        g = g.sort_values(wave_col).copy()

        gg = (
            g.groupby(wave_col, as_index=False)[intensity_col]
            .mean()
            .sort_values(wave_col)
        )

        x = gg[wave_col].to_numpy(float)
        y = gg[intensity_col].to_numpy(float)

        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]

        if x.size < 2:
            continue

        xu, idx = np.unique(x, return_index=True)
        yu = y[idx]

        if interp_grid:
            y_grid = np.interp(wave_grid, xu, yu)
            y_grid[(wave_grid < xu.min()) | (wave_grid > xu.max())] = np.nan
        else:
            y_grid = np.full(len(wave_grid), np.nan, dtype=float)
            mapper = dict(zip(xu.astype(int), yu))
            for i, w in enumerate(wave_grid):
                if w in mapper:
                    y_grid[i] = mapper[w]

        if not np.any(np.isfinite(y_grid)):
            continue

        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(id_cols, keys))

        if "NormPos" in g.columns:
            row["NormPos"] = pd.to_numeric(g["NormPos"], errors="coerce").mean()

        rows.append(row)
        Y.append(y_grid)

    meta = pd.DataFrame(rows)
    Y = np.vstack(Y) if Y else np.empty((0, len(wave_grid)))

    return meta, wave_grid, Y

def _run_simple_pca(Y, *, standardise=True):
    """
    PCA using numpy SVD.

    Returns:
        scores, loadings, explained_variance_ratio, mean, scale
    """

    Y = np.asarray(Y, dtype=float)

    if Y.size == 0 or Y.shape[0] < 2:
        return None

    # Fill occasional missing values with column means
    col_mean = np.nanmean(Y, axis=0)
    inds = np.where(~np.isfinite(Y))
    Y = Y.copy()
    Y[inds] = np.take(col_mean, inds[1])

    if standardise:
        mu = np.mean(Y, axis=0)
        sd = np.std(Y, axis=0, ddof=1)
        sd = np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)
        X = (Y - mu) / sd
    else:
        mu = np.mean(Y, axis=0)
        sd = np.ones(Y.shape[1])
        X = Y - mu

    Xc = X - np.mean(X, axis=0)

    U, S, VT = np.linalg.svd(Xc, full_matrices=False)

    scores = U * S
    loadings = VT.T

    var = (S ** 2) / max(Xc.shape[0] - 1, 1)
    evr = var / np.sum(var) if np.sum(var) > 0 else np.zeros_like(var)

    return {
        "scores": scores,
        "loadings": loadings,
        "explained_variance_ratio": evr,
        "mu": mu,
        "sd": sd,
    }

def plot_raman_pca_within_subtype(
    spectra,
    *,
    subtype_order=None,
    wave_col="WaveRound",
    intensity_col="Intensity",
    xlim=(1200, 1800),
    standardise=True,
    cmap="jet",
    ncols=3,
    point_size=35,
    alpha=0.75,
    title="Raman PCA within subtype",
    make_tables=True,
    plot_scores=True,
    plot_loadings=True,
    loading_pcs=("PC1", "PC2"),
    loading_top_n=12,
    peak_regions=None,
):
    """
    PCA separately within each subtype.

    Main PCA plot:
        each point = one Raman point spectrum
        colour = normalised position through line scan

    Additional outputs:
        variance_df:
            explained variance per subtype/PC

        loadings_df:
            strongest positive/negative loading wavenumbers for PC1/PC2

        score plots:
            PC score vs normalised position

        loading plots:
            PC loading vs wavenumber
    """

    meta, wave_grid, Y = _raman_spectra_to_matrix(
        spectra,
        wave_col=wave_col,
        intensity_col=intensity_col,
        xlim=xlim,
    )

    if meta.empty or Y.size == 0:
        print("No Raman spectra available for PCA.")
        return None, None, pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()

    present = list(pd.unique(meta["Subtype"].astype(str)))

    if subtype_order is None:
        subtypes = present
    else:
        subtypes = [st for st in subtype_order if st in present]

    if not subtypes:
        print("No matching Raman subtypes for PCA.")
        return None, None, pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()

    nrows = int(np.ceil(len(subtypes) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.2 * ncols, 4.6 * nrows),
        sharex=False,
        sharey=False,
    )

    axes = np.atleast_1d(axes).ravel()

    all_scores = []
    pca_info = {}
    last_scatter = None

    for ax, subtype in zip(axes, subtypes):
        idx = meta["Subtype"].astype(str) == str(subtype)
        meta_sub = meta.loc[idx].copy()
        Y_sub = Y[idx.to_numpy(), :]

        pca = _run_simple_pca(Y_sub, standardise=standardise)

        if pca is None:
            ax.set_title(f"{subtype}: insufficient data")
            ax.axis("off")
            continue

        scores = pca["scores"]
        evr = pca["explained_variance_ratio"]

        meta_sub["PC1"] = scores[:, 0]
        meta_sub["PC2"] = scores[:, 1] if scores.shape[1] > 1 else 0.0

        if scores.shape[1] > 2:
            meta_sub["PC3"] = scores[:, 2]

        if scores.shape[1] > 3:
            meta_sub["PC4"] = scores[:, 3]

        label = subtype_label(subtype) if "subtype_label" in globals() else subtype

        colour_values = (
            pd.to_numeric(meta_sub["NormPos"], errors="coerce")
            if "NormPos" in meta_sub.columns
            else np.arange(len(meta_sub))
        )

        last_scatter = ax.scatter(
            meta_sub["PC1"],
            meta_sub["PC2"],
            c=colour_values,
            cmap=cmap,
            vmin=0,
            vmax=100 if "NormPos" in meta_sub.columns else None,
            s=point_size,
            alpha=alpha,
            edgecolors="black",
            linewidths=0.3,
        )

        ax.set_title(label)
        ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)" if len(evr) > 1 else "PC2")
        ax.grid(False)

        meta_sub["PCA_Mode"] = "within_subtype"
        meta_sub["PCA_Subtype"] = subtype

        all_scores.append(meta_sub)

        pca_info[subtype] = {
            "wave_grid": wave_grid,
            "loadings": pca["loadings"],
            "explained_variance_ratio": evr,
            "standardised": standardise,
        }

    for ax in axes[len(subtypes):]:
        ax.axis("off")

    if last_scatter is not None:
        cbar = fig.colorbar(
            last_scatter,
            ax=axes[:len(subtypes)],
            fraction=0.025,
            pad=0.02,
        )
        cbar.set_label("Normalised position through line scan (%)")

    scores_df = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()

    fig.suptitle(title, y=0.995)
    plt.tight_layout()
    plt.show()

    if make_tables:
        variance_df, loadings_df = summarise_raman_pca_loadings(
            pca_info,
            pcs=loading_pcs,
            top_n=loading_top_n,
        )
    else:
        variance_df = pd.DataFrame()
        loadings_df = pd.DataFrame()

    if plot_scores and not scores_df.empty:
        plot_raman_pca_scores_vs_position(
            scores_df,
            pcs=loading_pcs,
            subtype_order=subtypes,
            ncols=ncols,
            title="Raman PCA scores vs normalised position",
        )

    if plot_loadings and pca_info:
        plot_raman_pca_loadings_by_subtype(
            pca_info,
            pcs=loading_pcs,
            subtype_order=subtypes,
            xlim=xlim,
            peak_regions=peak_regions,
            ncols=ncols,
            title="Raman PCA loadings by subtype",
        )

    return fig, axes, scores_df, pca_info, variance_df, loadings_df

def plot_raman_pca_loadings_by_subtype(
    pca_info,
    *,
    pcs=("PC1", "PC2"),
    subtype_order=None,
    xlim=(1200, 1800),
    peak_regions=None,
    figsize_per_panel=(5.5, 3.6),
    ncols=3,
    title="Raman PCA loadings by subtype",
):
    """
    Plot PCA loadings vs wavenumber.

    Each panel = subtype.
    Lines = selected PCs.
    """

    if not pca_info:
        print("No PCA info available.")
        return None, None

    available_subtypes = list(pca_info.keys())

    if subtype_order is None:
        subtypes = available_subtypes
    else:
        subtypes = [st for st in subtype_order if st in available_subtypes]

    if not subtypes:
        print("No matching subtypes in PCA info.")
        return None, None

    nrows = int(np.ceil(len(subtypes) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        sharex=True,
        sharey=False,
    )

    axes = np.atleast_1d(axes).ravel()

    for ax, subtype in zip(axes, subtypes):
        info = pca_info[subtype]

        wave_grid = np.asarray(info["wave_grid"], dtype=float)
        loadings = np.asarray(info["loadings"], dtype=float)
        evr = np.asarray(info["explained_variance_ratio"], dtype=float)

        for pc in pcs:
            pc_idx = int(str(pc).upper().replace("PC", "")) - 1

            if pc_idx >= loadings.shape[1]:
                continue

            label = (
                f"{pc.upper()} ({evr[pc_idx] * 100:.1f}%)"
                if pc_idx < len(evr)
                else pc.upper()
            )

            ax.plot(
                wave_grid,
                loadings[:, pc_idx],
                linewidth=1.3,
                label=label,
            )

        if peak_regions:
            for _, (x1, x2) in peak_regions:
                ax.axvline(x1, color="k", linestyle="--", alpha=0.18, linewidth=0.8)
                ax.axvline(x2, color="k", linestyle="--", alpha=0.18, linewidth=0.8)

        label = subtype_label(subtype) if "subtype_label" in globals() else subtype

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.45)
        ax.set_title(label)
        ax.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax.set_ylabel("Loading")
        ax.grid(False)
        ax.legend(fontsize="small", frameon=False)

        if xlim is not None:
            ax.set_xlim(*xlim)

    for ax in axes[len(subtypes):]:
        ax.axis("off")

    fig.suptitle(title, y=0.995)
    plt.tight_layout()
    plt.show()

    return fig, axes

def plot_raman_pca_scores_vs_position(
    scores_df,
    *,
    pcs=("PC1", "PC2"),
    subtype_order=None,
    colours=None,
    markers=None,
    ncols=3,
    alpha=0.75,
    point_size=30,
    title="Raman PCA scores vs normalised position",
):
    """
    Plot PC scores against normalised spatial position.

    Each panel = subtype.
    Lines/points = selected PC scores across position.
    """

    if scores_df is None or scores_df.empty:
        print("No PCA scores available.")
        return None, None

    if "NormPos" not in scores_df.columns:
        print("scores_df does not contain NormPos.")
        return None, None

    colours = colours or NI_COLOURS
    markers = markers or NI_MARKERS

    d = scores_df.copy()
    d["NormPos"] = pd.to_numeric(d["NormPos"], errors="coerce")
    d = d.dropna(subset=["NormPos"]).copy()

    available_subtypes = list(pd.unique(d["Subtype"].astype(str)))

    if subtype_order is None:
        subtypes = available_subtypes
    else:
        subtypes = [st for st in subtype_order if st in available_subtypes]

    if not subtypes:
        print("No matching subtypes in PCA scores.")
        return None, None

    nrows = int(np.ceil(len(subtypes) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.2 * ncols, 4.2 * nrows),
        sharex=True,
        sharey=False,
    )

    axes = np.atleast_1d(axes).ravel()

    for ax, subtype in zip(axes, subtypes):
        ds = d[d["Subtype"].astype(str) == subtype].copy()
        ds = ds.sort_values("NormPos")

        clean = normalise_subtype(subtype) if "normalise_subtype" in globals() else subtype
        colour = colours.get(clean, colours.get(subtype, "grey"))
        marker = markers.get(clean, markers.get(subtype, "o"))

        for pc in pcs:
            if pc not in ds.columns:
                continue

            # Plot all points
            ax.scatter(
                ds["NormPos"],
                ds[pc],
                color=colour,
                marker=marker,
                alpha=alpha,
                s=point_size,
                linewidths=0.3,
                edgecolors="black" if marker not in {"x", "+", "1", "2", "3", "4", "|", "_"} else None,
                label=pc,
            )

            # Mean trend by rounded position/bin-like values
            trend = (
                ds.groupby("NormPos", as_index=False, observed=True)[pc]
                .mean()
                .sort_values("NormPos")
            )

            ax.plot(
                trend["NormPos"],
                trend[pc],
                color=colour,
                linewidth=1.0,
                alpha=0.55,
            )

        label = subtype_label(subtype) if "subtype_label" in globals() else subtype

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_title(label)
        ax.set_xlabel("Normalised position through line scan (%)")
        ax.set_ylabel("PC score")
        ax.set_xlim(0, 100)
        ax.grid(False)
        ax.legend(frameon=False, fontsize="small")

    for ax in axes[len(subtypes):]:
        ax.axis("off")

    fig.suptitle(title, y=0.995)
    plt.tight_layout()
    plt.show()

    return fig, axes

def summarise_raman_pca_loadings(
    pca_info,
    *,
    pcs=("PC1", "PC2"),
    top_n=12,
):
    """
    Create tables describing Raman PCA results.

    Returns
    -------
    variance_df:
        Explained variance per subtype/PC.

    loadings_df:
        Top positive and negative loading wavenumbers per subtype/PC.
    """

    variance_rows = []
    loading_rows = []

    for subtype, info in pca_info.items():
        wave_grid = np.asarray(info["wave_grid"], dtype=float)
        loadings = np.asarray(info["loadings"], dtype=float)
        evr = np.asarray(info["explained_variance_ratio"], dtype=float)

        for i, frac in enumerate(evr, start=1):
            variance_rows.append({
                "Subtype": subtype,
                "PC": f"PC{i}",
                "ExplainedVarianceFraction": frac,
                "ExplainedVariancePercent": frac * 100,
            })

        for pc in pcs:
            pc_idx = int(str(pc).upper().replace("PC", "")) - 1

            if pc_idx >= loadings.shape[1]:
                continue

            loading = loadings[:, pc_idx]
            ok = np.isfinite(wave_grid) & np.isfinite(loading)

            w = wave_grid[ok]
            l = loading[ok]

            if len(l) == 0:
                continue

            pos_idx = np.argsort(l)[-top_n:][::-1]
            neg_idx = np.argsort(l)[:top_n]

            for rank, idx in enumerate(pos_idx, start=1):
                loading_rows.append({
                    "Subtype": subtype,
                    "PC": pc,
                    "Direction": "positive",
                    "Rank": rank,
                    "Wavenumber": w[idx],
                    "Loading": l[idx],
                    "AbsLoading": abs(l[idx]),
                })

            for rank, idx in enumerate(neg_idx, start=1):
                loading_rows.append({
                    "Subtype": subtype,
                    "PC": pc,
                    "Direction": "negative",
                    "Rank": rank,
                    "Wavenumber": w[idx],
                    "Loading": l[idx],
                    "AbsLoading": abs(l[idx]),
                })

    variance_df = pd.DataFrame(variance_rows)
    loadings_df = pd.DataFrame(loading_rows)

    return variance_df, loadings_df

def _make_raman_between_subtype_representatives(
    spectra,
    *,
    subtype_order=None,
    wave_col="WaveRound",
    intensity_col="Intensity",
    xlim=(1200, 1800),
    mode="binned",          # "single" or "binned"
    nbins=3,
    average_unit="sample",  # "sample" or "point"
    interp_grid=True,
):
    """
    Create representative spectra for between-subtype PCA.

    mode="single":
        one average spectrum per subtype

    mode="binned":
        one average spectrum per subtype/bin
    """

    meta, wave_grid, Y_points = _raman_spectra_to_matrix(
        spectra,
        wave_col=wave_col,
        intensity_col=intensity_col,
        xlim=xlim,
        interp_grid=interp_grid,
    )

    if meta.empty or Y_points.size == 0:
        return pd.DataFrame(), wave_grid, np.empty((0, len(wave_grid)))

    present = list(pd.unique(meta["Subtype"].astype(str)))

    if subtype_order is None:
        subtypes = present
    else:
        subtypes = [st for st in subtype_order if st in present]

    meta = meta[meta["Subtype"].isin(subtypes)].copy()
    Y_points = Y_points[meta.index.to_numpy(), :]
    meta = meta.reset_index(drop=True)

    if mode not in {"single", "binned"}:
        raise ValueError("mode must be 'single' or 'binned'.")

    if average_unit not in {"point", "sample"}:
        raise ValueError("average_unit must be 'point' or 'sample'.")

    if mode == "binned":
        bin_rows = []

        for (sample, subtype), g in meta.groupby(["Sample", "Subtype"], observed=True):
            g = g.sort_values(["PointIndex", "x", "y"]).copy()
            n = len(g)

            if n == 0:
                continue

            bins = np.linspace(0, 100, nbins + 1)

            if "NormPos" in g.columns:
                g["Bin"] = pd.cut(
                    pd.to_numeric(g["NormPos"], errors="coerce"),
                    bins=bins,
                    labels=np.arange(1, nbins + 1),
                    include_lowest=True,
                ).astype(int)
            else:
                g["Bin"] = pd.cut(
                    np.arange(n),
                    bins=nbins,
                    labels=np.arange(1, nbins + 1),
                    include_lowest=True,
                ).astype(int)

            bin_rows.append(g)

        meta = pd.concat(bin_rows, ignore_index=False).sort_index()

    else:
        meta["Bin"] = 1

    rows = []
    Y_reps = []

    for (subtype, b), g in meta.groupby(["Subtype", "Bin"], observed=True):
        idx = g.index.to_numpy()
        Y_sub = Y_points[idx, :]

        if average_unit == "sample":
            sample_means = []

            for sample, gs in g.groupby("Sample", observed=True):
                sample_idx = gs.index.to_numpy()
                sample_mean = np.nanmean(Y_points[sample_idx, :], axis=0)

                if np.any(np.isfinite(sample_mean)):
                    sample_means.append(sample_mean)

            Y_for_mean = np.vstack(sample_means) if sample_means else np.empty((0, len(wave_grid)))
        else:
            Y_for_mean = Y_sub

        if Y_for_mean.size == 0:
            continue

        mean_spec = np.nanmean(Y_for_mean, axis=0)

        if not np.any(np.isfinite(mean_spec)):
            continue

        rows.append({
            "Subtype": subtype,
            "Bin": int(b),
            "NormPos": ((int(b) - 0.5) / nbins) * 100 if mode == "binned" else np.nan,
            "n_replicates": Y_for_mean.shape[0],
            "mode": mode,
            "average_unit": average_unit,
        })

        Y_reps.append(mean_spec)

    rep_meta = pd.DataFrame(rows)
    Y_reps = np.vstack(Y_reps) if Y_reps else np.empty((0, len(wave_grid)))

    return rep_meta, wave_grid, Y_reps

def plot_raman_pca_between_subtypes(
    spectra,
    *,
    subtype_order=None,
    colours=None,
    markers=None,
    wave_col="WaveRound",
    intensity_col="Intensity",
    xlim=(1200, 1800),
    mode="binned",          # "single" or "binned"
    nbins=3,
    average_unit="sample",
    standardise=True,
    cmap="jet",
    point_size=80,
    alpha=0.85,
    show_loadings=True,
    show_score_trends=True,
    loading_pcs=("PC1", "PC2"),
    loading_n=12,
    peak_regions=None,
    title="Raman PCA between subtypes",
):
    """
    PCA between subtypes.

    mode="single":
        one average spectrum per subtype.

    mode="binned":
        one average spectrum per subtype/bin, preserving spatial information.

    Returns:
        fig, ax, rep_meta, pca_info, variance_df, loadings_df
    """

    def _pc_index(pc):
        return int(str(pc).upper().replace("PC", "")) - 1

    def _plot_between_loadings(pca_info):
        wave_grid = np.asarray(pca_info["wave_grid"], dtype=float)
        loadings = np.asarray(pca_info["loadings"], dtype=float)
        evr = np.asarray(pca_info["explained_variance_ratio"], dtype=float)

        fig_l, ax_l = plt.subplots(figsize=(10, 4.5))

        for pc in loading_pcs:
            idx = _pc_index(pc)

            if idx >= loadings.shape[1]:
                continue

            label = (
                f"{pc.upper()} ({evr[idx] * 100:.1f}%)"
                if idx < len(evr)
                else pc.upper()
            )

            ax_l.plot(
                wave_grid,
                loadings[:, idx],
                linewidth=1.3,
                label=label,
            )

        if peak_regions:
            for _, (x1, x2) in peak_regions:
                ax_l.axvline(x1, color="k", linestyle="--", alpha=0.18, linewidth=0.8)
                ax_l.axvline(x2, color="k", linestyle="--", alpha=0.18, linewidth=0.8)

        ax_l.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax_l.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax_l.set_ylabel("Loading")
        ax_l.set_title("Raman PCA loadings between subtypes")
        ax_l.legend(frameon=False)
        ax_l.grid(False)

        if xlim is not None:
            ax_l.set_xlim(*xlim)

        plt.tight_layout()
        plt.show()

        return fig_l, ax_l

    def _plot_score_trends(rep_meta, evr):
        if mode != "binned" or "NormPos" not in rep_meta.columns:
            return None, None

        pcs_to_plot = [pc for pc in loading_pcs if pc in rep_meta.columns]

        if not pcs_to_plot:
            return None, None

        fig_s, axes_s = plt.subplots(
            1,
            len(pcs_to_plot),
            figsize=(5.4 * len(pcs_to_plot), 4.3),
            sharex=True,
        )

        axes_s = np.atleast_1d(axes_s).ravel()

        for ax_s, pc in zip(axes_s, pcs_to_plot):
            for subtype in subtypes:
                ds = rep_meta[rep_meta["Subtype"].astype(str) == str(subtype)].copy()

                if ds.empty:
                    continue

                ds = ds.sort_values("NormPos")

                clean = normalise_subtype(subtype) if "normalise_subtype" in globals() else subtype
                colour = colours.get(clean, colours.get(subtype, "grey"))
                marker = markers.get(clean, markers.get(subtype, "o"))
                label = subtype_label(subtype) if "subtype_label" in globals() else subtype

                scatter_kwargs = dict(
                    color=colour,
                    marker=marker,
                    s=point_size * 0.6,
                    alpha=alpha,
                    label=label,
                    zorder=3,
                )

                if marker not in {"x", "+", "1", "2", "3", "4", "|", "_"}:
                    scatter_kwargs["edgecolors"] = "black"
                    scatter_kwargs["linewidths"] = 0.4

                ax_s.scatter(ds["NormPos"], ds[pc], **scatter_kwargs)

                ax_s.plot(
                    ds["NormPos"],
                    ds[pc],
                    color=colour,
                    linewidth=1.0,
                    alpha=0.45,
                    zorder=2,
                )

            idx = _pc_index(pc)
            pc_label = f"{pc.upper()} ({evr[idx] * 100:.1f}%)" if idx < len(evr) else pc.upper()

            ax_s.axhline(0, color="black", linewidth=0.8, alpha=0.4)
            ax_s.set_xlabel("Normalised position through line scan (%)")
            ax_s.set_ylabel("PC score")
            ax_s.set_title(pc_label)
            ax_s.set_xlim(0, 100)
            ax_s.grid(False)
            ax_s.legend(frameon=False, fontsize="small")

        fig_s.suptitle("Between-subtype Raman PCA scores vs spatial position", y=0.995)
        plt.tight_layout()
        plt.show()

        return fig_s, axes_s

    colours = colours or NI_COLOURS
    markers = markers or NI_MARKERS

    rep_meta, wave_grid, Y = _make_raman_between_subtype_representatives(
        spectra,
        subtype_order=subtype_order,
        wave_col=wave_col,
        intensity_col=intensity_col,
        xlim=xlim,
        mode=mode,
        nbins=nbins,
        average_unit=average_unit,
    )

    if rep_meta.empty or Y.size == 0:
        print("No representative Raman spectra available for between-subtype PCA.")
        return None, None, pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()

    pca = _run_simple_pca(Y, standardise=standardise)

    if pca is None:
        print("Not enough spectra for PCA.")
        return None, None, pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()

    scores = pca["scores"]
    evr = pca["explained_variance_ratio"]

    rep_meta = rep_meta.copy()

    for i in range(min(scores.shape[1], 6)):
        rep_meta[f"PC{i + 1}"] = scores[:, i]

    if "PC2" not in rep_meta.columns:
        rep_meta["PC2"] = 0.0

    fig, ax = plt.subplots(figsize=(7.2, 5.8))

    subtypes = (
        subtype_order
        if subtype_order is not None
        else list(pd.unique(rep_meta["Subtype"].astype(str)))
    )

    last_sc = None

    for subtype in subtypes:
        ds = rep_meta[rep_meta["Subtype"].astype(str) == str(subtype)].copy()

        if ds.empty:
            continue

        label = subtype_label(subtype) if "subtype_label" in globals() else subtype
        clean = normalise_subtype(subtype) if "normalise_subtype" in globals() else subtype

        marker = markers.get(clean, markers.get(subtype, "o"))
        colour = colours.get(clean, colours.get(subtype, "grey"))

        scatter_kwargs = dict(
            marker=marker,
            s=point_size,
            alpha=alpha,
            label=label,
            zorder=3,
        )

        if marker not in {"x", "+", "1", "2", "3", "4", "|", "_"}:
            scatter_kwargs["edgecolors"] = "black"
            scatter_kwargs["linewidths"] = 0.5

        if mode == "binned":
            last_sc = ax.scatter(
                ds["PC1"],
                ds["PC2"],
                c=ds["NormPos"],
                cmap=cmap,
                vmin=0,
                vmax=100,
                **scatter_kwargs,
            )

            ds = ds.sort_values("NormPos")

            ax.plot(
                ds["PC1"],
                ds["PC2"],
                color=colour,
                alpha=0.35,
                linewidth=1.0,
                zorder=1,
            )

        else:
            ax.scatter(
                ds["PC1"],
                ds["PC2"],
                color=colour,
                **scatter_kwargs,
            )

    if mode == "binned" and last_sc is not None:
        cbar = plt.colorbar(last_sc, ax=ax)
        cbar.set_label("Normalised position through line scan (%)")

    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)" if len(evr) > 1 else "PC2")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(False)

    pca_info = {
        "wave_grid": wave_grid,
        "loadings": pca["loadings"],
        "explained_variance_ratio": evr,
        "standardised": standardise,
        "mode": mode,
        "average_unit": average_unit,
    }

    plt.tight_layout()
    plt.show()

    # Make tables using the same helper as the within-subtype PCA.
    table_info = {
        "between_subtypes": pca_info
    }

    variance_df, loadings_df = summarise_raman_pca_loadings(
        table_info,
        pcs=loading_pcs,
        top_n=loading_n,
    )

    if show_score_trends:
        _plot_score_trends(rep_meta, evr)

    if show_loadings:
        _plot_between_loadings(pca_info)

    return fig, ax, rep_meta, pca_info, variance_df, loadings_df

def summarise_raman_pca_components_at_glance(
    pca_info,
    *,
    pcs=("PC1", "PC2", "PC3"),
    top_n=25,
    merge_gap=8,
    min_region_width=3,
    peak_regions=None,
):
    """
    Create an at-a-glance PCA interpretation table.

    Output format:
        PCA_Model | PC | Variance_% | Positive_contributors | Negative_contributors

    Contributors are grouped into approximate wavenumber ranges based on the
    strongest absolute loadings for each PC.
    """

    def _pc_index(pc):
        return int(str(pc).upper().replace("PC", "")) - 1

    def _region_label_for_range(lo, hi):
        if not peak_regions:
            return ""

        hits = []
        for name, (r1, r2) in peak_regions:
            overlap = max(0, min(hi, r2) - max(lo, r1))
            if overlap > 0:
                hits.append(name)

        if not hits:
            return ""

        short = []
        for h in hits:
            if h.startswith("CH2CH3"):
                short.append("CH2/CH3")
            elif h.startswith("AmideIII"):
                short.append("Amide III")
            elif h.startswith("AmideI"):
                short.append("Amide I")
            else:
                short.append(h.split("_")[0])

        return f" ({', '.join(dict.fromkeys(short))})"

    def _merge_waves_to_ranges(waves):
        waves = np.asarray(waves, dtype=float)
        waves = np.sort(waves[np.isfinite(waves)])

        if waves.size == 0:
            return ""

        ranges = []
        start = waves[0]
        prev = waves[0]

        for w in waves[1:]:
            if (w - prev) <= merge_gap:
                prev = w
            else:
                if (prev - start) >= min_region_width:
                    label = _region_label_for_range(start, prev)
                    ranges.append(f"{int(round(start))}–{int(round(prev))}{label}")
                else:
                    label = _region_label_for_range(start, prev)
                    ranges.append(f"{int(round(start))}{label}")
                start = w
                prev = w

        if (prev - start) >= min_region_width:
            label = _region_label_for_range(start, prev)
            ranges.append(f"{int(round(start))}–{int(round(prev))}{label}")
        else:
            label = _region_label_for_range(start, prev)
            ranges.append(f"{int(round(start))}{label}")

        return "; ".join(ranges)

    rows = []

    for model_name, info in pca_info.items():
        wave_grid = np.asarray(info["wave_grid"], dtype=float)
        loadings = np.asarray(info["loadings"], dtype=float)
        evr = np.asarray(info["explained_variance_ratio"], dtype=float)

        for pc in pcs:
            pc_idx = _pc_index(pc)

            if pc_idx >= loadings.shape[1]:
                continue

            loading = loadings[:, pc_idx]
            ok = np.isfinite(wave_grid) & np.isfinite(loading)

            w = wave_grid[ok]
            l = loading[ok]

            if w.size == 0:
                continue

            # Strongest positive contributors
            pos_mask = l > 0
            if np.any(pos_mask):
                pos_w = w[pos_mask]
                pos_l = l[pos_mask]
                pos_idx = np.argsort(pos_l)[-top_n:]
                pos_ranges = _merge_waves_to_ranges(pos_w[pos_idx])
            else:
                pos_ranges = ""

            # Strongest negative contributors
            neg_mask = l < 0
            if np.any(neg_mask):
                neg_w = w[neg_mask]
                neg_l = l[neg_mask]
                neg_idx = np.argsort(np.abs(neg_l))[-top_n:]
                neg_ranges = _merge_waves_to_ranges(neg_w[neg_idx])
            else:
                neg_ranges = ""

            rows.append({
                "PCA_Model": model_name,
                "PC": pc.upper(),
                "Variance_%": evr[pc_idx] * 100 if pc_idx < len(evr) else np.nan,
                "Positive_contributors": pos_ranges,
                "Negative_contributors": neg_ranges,
                "Interpretation": (
                    f"{pc.upper()} explains {evr[pc_idx] * 100:.1f}% of variance; "
                    f"positive loadings mainly around {pos_ranges or 'none'}; "
                    f"negative loadings mainly around {neg_ranges or 'none'}."
                    if pc_idx < len(evr)
                    else ""
                ),
            })


    return pd.DataFrame(rows)

def plot_raman_pca_scree(
    pca_info,
    *,
    subtype_order=None,
    max_pcs=10,
    title="Raman PCA explained variance",
):
    """
    Plot explained variance and cumulative explained variance for each PCA model.
    """

    if not pca_info:
        print("No PCA info available.")
        return None, None

    available = list(pca_info.keys())

    if subtype_order is None:
        subtypes = available
    else:
        subtypes = [st for st in subtype_order if st in available]

    if not subtypes:
        print("No matching PCA models.")
        return None, None

    fig, axes = plt.subplots(
        1,
        len(subtypes),
        figsize=(5 * len(subtypes), 4),
        sharey=True,
    )

    axes = np.atleast_1d(axes)

    for ax, subtype in zip(axes, subtypes):
        evr = np.asarray(pca_info[subtype]["explained_variance_ratio"], dtype=float)
        evr = evr[:max_pcs]

        pcs = np.arange(1, len(evr) + 1)
        cumulative = np.cumsum(evr)

        label = subtype_label(subtype) if "subtype_label" in globals() else subtype

        ax.bar(pcs, evr * 100, alpha=0.65, label="Individual PC")
        ax.plot(pcs, cumulative * 100, marker="o", linewidth=1.5, label="Cumulative")

        ax.set_title(label)
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Explained variance (%)")
        ax.set_xticks(pcs)
        ax.set_ylim(0, 100)
        ax.grid(False)
        ax.legend(frameon=False, fontsize="small")

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

    return fig, axes


# ====================================================================================================================================================
# MULTIMODAL CORRELATION HELPERS

def _normalise_corr_subtype(value):
    """
    Local subtype harmoniser for correlation functions only.

    This allows the new correlation functions to accept either:
        CT, D7, D14
    or:
        control, d7, d14

    without changing your existing loading/preparation functions.
    """

    if pd.isna(value):
        return ""

    key = str(value).strip().lower()

    mapping = {
        "ct": "control",
        "control": "control",
        "unwounded": "control",

        "pwd4": "d4",
        "d4": "d4",

        "pwd7": "d7",
        "d7": "d7",

        "pwd10": "d10",
        "d10": "d10",

        "pwd14": "d14",
        "d14": "d14",

        "pwd21": "d21",
        "d21": "d21",
    }

    return mapping.get(key, key)

def _normalise_corr_subtypes_to_plot(subtypes_to_plot):
    """
    Normalise a list of subtype labels for correlation functions only.
    """

    if subtypes_to_plot is None:
        return None

    return [_normalise_corr_subtype(st) for st in subtypes_to_plot]

def _add_corr_subtype_clean(
    df,
    source_cols=("SubtypeClean", "Subtype", "subtype", "ExportSubtype"),
):
    """
    Add a correlation-only subtype column called CorrSubtypeClean.

    Does not overwrite your existing SubtypeClean column.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    source_col = None

    for col in source_cols:
        if col in d.columns:
            source_col = col
            break

    if source_col is None:
        raise KeyError(f"Could not find subtype column. Checked: {source_cols}")

    d["CorrSubtypeClean"] = d[source_col].apply(_normalise_corr_subtype)

    return d

def _resolve_corr_subtypes(subtype_order=SUBTYPE_ORDER, subtypes_to_plot=None):
    """
    Resolve subtype order for correlation functions only.
    """

    subtype_order_clean = [_normalise_corr_subtype(st) for st in subtype_order]
    subtypes_to_plot_clean = _normalise_corr_subtypes_to_plot(subtypes_to_plot)

    if subtypes_to_plot_clean is None:
        return subtype_order_clean

    out = [st for st in subtype_order_clean if st in set(subtypes_to_plot_clean)]
    out += [st for st in subtypes_to_plot_clean if st not in out]

    return out

def _region_to_depth(region):
    """
    Convert region names to lower/upper depth labels.
    """
    r = str(region).strip().lower()

    if "sub" in r or "lower" in r:
        return "lower"
    if "epi" in r or "upper" in r:
        return "upper"

    return np.nan

def _depth_to_region(depth, modality_region="wound"):
    """
    Convert depth labels to standard region names.
    """
    depth = str(depth).strip().lower()
    modality_region = str(modality_region).strip().lower()

    if modality_region == "dermis":
        return "dermis_sub" if depth == "lower" else "dermis_epi"
    if modality_region == "wound":
        return "wound_sub" if depth == "lower" else "wound_epi"

    return depth

def _bin_to_depth(bin_number, nbins=2):
    """
    For nbins=2:
        Bin 1 = lower
        Bin 2 = upper
    """
    b = int(bin_number)

    if nbins == 2:
        return "lower" if b == 1 else "upper"

    if b <= nbins / 2:
        return "lower"

    return "upper"

def _subtype_sort_key(subtype, subtype_order=SUBTYPE_ORDER):
    subtype = str(subtype)
    return subtype_order.index(subtype) if subtype in subtype_order else 999

def _simple_corr_stats(x, y):
    """
    Return Pearson r using numpy only.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if len(x) < 3:
        return np.nan, len(x)

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, len(x)

    return float(np.corrcoef(x, y)[0, 1]), len(x)

def _plot_correlation(
    df,
    x_col,
    y_col,
    *,
    subtype_col="SubtypeClean",
    colour_by=None,
    cmap="jet",
    colours=NI_COLOURS,
    labels=NI_LABELS,
    markers=NI_MARKERS,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    panel_by_subtype=False,
    ncols=3,
    xlabel=None,
    ylabel=None,
    title=None,
    alpha=0.75,
    point_size=55,
    add_fit=False,
    connect_points=True,
    connect_alpha=0.28,
    connect_linewidth=0.8,
    xlim=None,
    ylim=None,
):
    """
    General correlation scatter plot.

    Correlation-safe:
        - accepts CT/D7/D14 or control/d7/d14
        - internally harmonises subtype labels for plotting/filtering

    If connect_points=True:
        joins points within each subtype using a thin transparent line.
        If colour_by exists, points are sorted by colour_by first.
        Otherwise, points are sorted by x_col.
    """

    if df is None or df.empty:
        print("No correlation data to plot.")
        return None, None, pd.DataFrame()

    d = df.copy()

    if subtype_col not in d.columns:
        raise KeyError(f"Correlation dataframe missing subtype column: {subtype_col}")

    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")

    d = d.dropna(subset=[x_col, y_col, subtype_col]).copy()

    if d.empty:
        print("No finite x/y data to plot.")
        return None, None, pd.DataFrame()

    d["CorrSubtypeClean"] = d[subtype_col].apply(_normalise_corr_subtype)

    subtype_order_clean = _resolve_corr_subtypes(
        subtype_order=subtype_order,
        subtypes_to_plot=subtypes_to_plot,
    )

    present_subtypes = [
        st for st in subtype_order_clean
        if st in set(d["CorrSubtypeClean"].astype(str))
    ]

    d = d[d["CorrSubtypeClean"].astype(str).isin(present_subtypes)].copy()

    if d.empty or not present_subtypes:
        print("No matching subtypes to plot.")
        print("Requested:", subtype_order_clean)
        print("Available:", sorted(df[subtype_col].astype(str).unique()))
        return None, None, pd.DataFrame()

    stats_rows = []

    def _add_fit(ax, x, y, colour="black"):
        ok = np.isfinite(x) & np.isfinite(y)
        x = np.asarray(x, float)[ok]
        y = np.asarray(y, float)[ok]

        if len(x) < 3 or np.nanstd(x) == 0:
            return

        m, c = np.polyfit(x, y, 1)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xx, m * xx + c, color=colour, linewidth=1.3, alpha=0.65)

    def _scatter_kwargs(subtype):
        marker = markers.get(subtype, "o")

        kwargs = {
            "marker": marker,
            "s": point_size,
            "alpha": alpha,
            "label": labels.get(subtype, subtype),
        }

        if marker not in {"x", "+", "1", "2", "3", "4", "|", "_"}:
            kwargs["edgecolors"] = "black"
            kwargs["linewidths"] = 0.4

        return kwargs

    def _sort_for_connection(ds):
        ds = ds.copy()

        if colour_by is not None and colour_by in ds.columns:
            ds[colour_by] = pd.to_numeric(ds[colour_by], errors="coerce")
            sort_cols = [colour_by, x_col, y_col]
        elif "NormPos" in ds.columns:
            ds["NormPos"] = pd.to_numeric(ds["NormPos"], errors="coerce")
            sort_cols = ["NormPos", x_col, y_col]
        elif "Bin" in ds.columns:
            ds["Bin"] = pd.to_numeric(ds["Bin"], errors="coerce")
            sort_cols = ["Bin", x_col, y_col]
        elif "Depth" in ds.columns:
            depth_order = {"lower": 0, "middle": 1, "upper": 2}
            ds["_DepthOrder"] = ds["Depth"].astype(str).str.lower().map(depth_order)
            sort_cols = ["_DepthOrder", x_col, y_col]
        else:
            sort_cols = [x_col, y_col]

        return ds.sort_values(sort_cols)

    def _connect(ax, ds, subtype):
        if not connect_points:
            return

        ds = _sort_for_connection(ds)
        x = ds[x_col].to_numpy(float)
        y = ds[y_col].to_numpy(float)

        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]

        if len(x) < 2:
            return

        ax.plot(
            x,
            y,
            color=colours.get(subtype, "black"),
            linewidth=connect_linewidth,
            alpha=connect_alpha,
            zorder=1,
        )

    # ------------------------------------------------------------------
    # Panel mode
    # ------------------------------------------------------------------
    if panel_by_subtype:
        nrows = int(np.ceil(len(present_subtypes) / ncols))

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.2 * ncols, 4.5 * nrows),
            sharex=False,
            sharey=False,
        )

        axes = np.atleast_1d(axes).ravel()
        last_sc = None

        for ax, subtype in zip(axes, present_subtypes):
            ds = d[d["CorrSubtypeClean"].astype(str) == subtype].copy()

            _connect(ax, ds, subtype)

            x = ds[x_col].to_numpy(float)
            y = ds[y_col].to_numpy(float)

            if colour_by is not None and colour_by in ds.columns:
                cvals = pd.to_numeric(ds[colour_by], errors="coerce").to_numpy(float)

                kwargs = _scatter_kwargs(subtype)
                kwargs.pop("label", None)

                last_sc = ax.scatter(
                    x,
                    y,
                    c=cvals,
                    cmap=cmap,
                    zorder=3,
                    **kwargs,
                )
            else:
                ax.scatter(
                    x,
                    y,
                    color=colours.get(subtype, "grey"),
                    zorder=3,
                    **_scatter_kwargs(subtype),
                )

            r, n = _simple_corr_stats(x, y)

            stats_rows.append({
                "SubtypeClean": subtype,
                "r": r,
                "n": n,
                "x_col": x_col,
                "y_col": y_col,
            })

            if add_fit:
                _add_fit(ax, x, y, colour="black")

            ax.set_title(
                f"{labels.get(subtype, subtype)} | r={r:.2f}, n={n}"
                if np.isfinite(r)
                else f"{labels.get(subtype, subtype)} | n={n}"
            )

            ax.set_xlabel(xlabel or x_col)
            ax.set_ylabel(ylabel or y_col)
            ax.grid(False)

            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)

        for ax in axes[len(present_subtypes):]:
            ax.axis("off")

        if last_sc is not None:
            cbar = fig.colorbar(
                last_sc,
                ax=axes[:len(present_subtypes)],
                fraction=0.025,
                pad=0.02,
            )
            cbar.set_label(colour_by)

        fig.suptitle(title or f"{y_col} vs {x_col}", y=0.995)
        plt.tight_layout()
        plt.show()

        return fig, axes, pd.DataFrame(stats_rows)

    # ------------------------------------------------------------------
    # Combined mode
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    last_sc = None

    for subtype in present_subtypes:
        ds = d[d["CorrSubtypeClean"].astype(str) == subtype].copy()

        _connect(ax, ds, subtype)

        x = ds[x_col].to_numpy(float)
        y = ds[y_col].to_numpy(float)

        if colour_by is not None and colour_by in ds.columns:
            cvals = pd.to_numeric(ds[colour_by], errors="coerce").to_numpy(float)

            last_sc = ax.scatter(
                x,
                y,
                c=cvals,
                cmap=cmap,
                zorder=3,
                **_scatter_kwargs(subtype),
            )
        else:
            ax.scatter(
                x,
                y,
                color=colours.get(subtype, "grey"),
                zorder=3,
                **_scatter_kwargs(subtype),
            )

    r_all, n_all = _simple_corr_stats(d[x_col], d[y_col])

    stats_rows.append({
        "SubtypeClean": "all",
        "r": r_all,
        "n": n_all,
        "x_col": x_col,
        "y_col": y_col,
    })

    for subtype in present_subtypes:
        ds = d[d["CorrSubtypeClean"].astype(str) == subtype]
        r, n = _simple_corr_stats(ds[x_col], ds[y_col])

        stats_rows.append({
            "SubtypeClean": subtype,
            "r": r,
            "n": n,
            "x_col": x_col,
            "y_col": y_col,
        })

    if add_fit:
        _add_fit(
            ax,
            d[x_col].to_numpy(float),
            d[y_col].to_numpy(float),
            colour="black",
        )

    if last_sc is not None:
        cbar = plt.colorbar(last_sc, ax=ax)
        cbar.set_label(colour_by)

    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)

    if np.isfinite(r_all):
        ax.set_title(title or f"{y_col} vs {x_col} | r={r_all:.2f}, n={n_all}")
    else:
        ax.set_title(title or f"{y_col} vs {x_col} | n={n_all}")

    ax.legend(frameon=False)
    ax.grid(False)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()

    return fig, ax, pd.DataFrame(stats_rows)

# =============================================================================
#                       MODALITY-SPECIFIC SUMMARY TABLES
# =============================================================================

def make_ni_upper_lower_summary(
    ni_raw,
    *,
    variables=("mod_Hertz", "tau_Visco"),
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
):
    """
    Split NI raw line scans into lower/upper using NormalisedPosition.

    Returns one row per sample/subtype/depth with mean NI variables.
    """

    subtype_order = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)

    if ni_raw is None or ni_raw.empty:
        return pd.DataFrame()

    base_var = variables[0]
    d = prepare_ni_raw_line_data(
        ni_raw=ni_raw,
        variable=base_var,
    )

    if d.empty:
        return pd.DataFrame()

    # Bring requested variables from original columns if prepare_ni_raw_line_data kept them
    for var in variables:
        if var not in d.columns and var != base_var:
            if var in ni_raw.columns:
                merge_cols = [c for c in ["Sample", "Subtype", "x", "y", var] if c in ni_raw.columns]
                if set(["Sample", "Subtype", var]).issubset(merge_cols):
                    d = d.merge(
                        ni_raw[merge_cols].drop_duplicates(),
                        on=[c for c in ["Sample", "Subtype", "x", "y"] if c in d.columns and c in merge_cols],
                        how="left",
                    )

    sample_col = "SampleKey" if "SampleKey" in d.columns else "Sample"

    d = d[d["SubtypeClean"].astype(str).isin(subtype_order)].copy()
    d["Depth"] = np.where(d["NormalisedPosition"] < 50, "lower", "upper")

    rows = []

    for (subtype, sample, depth), g in d.groupby(["SubtypeClean", sample_col, "Depth"], observed=True):
        row = {
            "SubtypeClean": str(subtype),
            "Sample": sample,
            "Depth": depth,
            "NormPos": 25.0 if depth == "lower" else 75.0,
        }

        for var in variables:
            if var == base_var:
                vals = g["Value"]
            elif var in g.columns:
                vals = pd.to_numeric(g[var], errors="coerce")
                if var in {
                    "mod_Hertz", "mod_OP", "E0_Visco", "Einf_Visco",
                    "G0_Visco", "G1_Visco", "Eff_file", "mod_file",
                }:
                    vals = vals / 1000
            else:
                vals = pd.Series(dtype=float)

            row[var] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan

        rows.append(row)

    return pd.DataFrame(rows)

def make_ni_binned_summary(
    ni_raw,
    *,
    variables=("mod_Hertz", "tau_Visco"),
    nbins=10,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
):
    """
    Split NI raw line scans into nbins along NormalisedPosition.

    Correlation-safe:
        - accepts CT/D7/D14 or control/d7/d14
        - outputs harmonised SubtypeClean for merging
    """

    if ni_raw is None or ni_raw.empty:
        return pd.DataFrame()

    subtypes_to_use = _resolve_corr_subtypes(
        subtype_order=subtype_order,
        subtypes_to_plot=subtypes_to_plot,
    )

    base_var = variables[0]

    d = prepare_ni_raw_line_data(
        ni_raw=ni_raw,
        variable=base_var,
    )

    if d.empty:
        return pd.DataFrame()

    d = _add_corr_subtype_clean(d)

    for var in variables:
        if var not in d.columns and var != base_var:
            if var in ni_raw.columns:
                merge_cols = [
                    c for c in ["Sample", "Subtype", "x", "y", var]
                    if c in ni_raw.columns
                ]

                merge_keys = [
                    c for c in ["Sample", "Subtype", "x", "y"]
                    if c in d.columns and c in merge_cols
                ]

                if merge_keys and var in merge_cols:
                    d = d.merge(
                        ni_raw[merge_cols].drop_duplicates(),
                        on=merge_keys,
                        how="left",
                    )

    sample_col = "SampleKey" if "SampleKey" in d.columns else "Sample"

    d = d[d["CorrSubtypeClean"].isin(subtypes_to_use)].copy()

    if d.empty:
        print("No NI rows found for requested subtypes.")
        return pd.DataFrame()

    bins = np.linspace(0, 100, nbins + 1)

    d["Bin"] = pd.cut(
        d["NormalisedPosition"],
        bins=bins,
        labels=np.arange(1, nbins + 1),
        include_lowest=True,
    ).astype(float)

    d["NormPosBin"] = d["Bin"].map({
        b: (bins[int(b) - 1] + bins[int(b)]) / 2
        for b in range(1, nbins + 1)
    })

    rows = []

    for (subtype, sample, b), g in d.groupby(
        ["CorrSubtypeClean", sample_col, "Bin"],
        observed=True,
    ):
        row = {
            "SubtypeClean": str(subtype),
            "Sample": sample,
            "Bin": int(b),
            "NormPos": float(g["NormPosBin"].iloc[0]),
        }

        for var in variables:
            if var == base_var:
                vals = g["Value"]

            elif var in g.columns:
                vals = pd.to_numeric(g[var], errors="coerce")

                if var in {
                    "mod_Hertz",
                    "mod_OP",
                    "E0_Visco",
                    "Einf_Visco",
                    "G0_Visco",
                    "G1_Visco",
                    "Eff_file",
                    "mod_file",
                }:
                    vals = vals / 1000

            else:
                vals = pd.Series(dtype=float)

            row[var] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan

        rows.append(row)

    return pd.DataFrame(rows)

def make_cell_upper_lower_summary(
    cell_raw,
    *,
    value_col=CELL_VALUE_COL,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
):
    """
    Convert cell upper/lower regions into one row per subtype/depth.
    """

    subtype_order = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)

    d = prepare_cell_raw(cell_raw, value_col=value_col)

    if d.empty:
        return pd.DataFrame()

    d = d[d["SubtypeClean"].astype(str).isin(subtype_order)].copy()
    d["Depth"] = d["Region"].apply(_region_to_depth)
    d = d.dropna(subset=["Depth"]).copy()

    summary = (
        d.groupby(["SubtypeClean", "Depth"], as_index=False, observed=True)[value_col]
        .agg(CellDensity="mean", CellDensity_std="std", CellDensity_n="count")
    )

    summary["NormPos"] = np.where(summary["Depth"] == "lower", 25.0, 75.0)

    return summary

def make_saxs_upper_lower_summary(
    saxs_points,
    *,
    parameters,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    trim_std_devs=6,
    normalise_curvearea=True,
):
    """
    Prepare SAXS upper/lower summaries.

    Correlation-safe:
        - accepts CT/D7/D14 or control/d7/d14
        - uses CorrSubtypeClean internally
        - outputs harmonised SubtypeClean for merging

    Returns one row per subtype/sample/depth with one column per parameter.
    """

    if saxs_points is None or saxs_points.empty:
        return pd.DataFrame()

    subtypes_to_use = _resolve_corr_subtypes(
        subtype_order=subtype_order,
        subtypes_to_plot=subtypes_to_plot,
    )

    all_rows = []

    for parameter in parameters:
        if parameter == "curvearea":
            d, value_col = prepare_saxs_point_data_for_plot(
                saxs_points=saxs_points,
                parameter=parameter,
                normalise=normalise_curvearea,
                trim_std_devs=trim_std_devs,
            )
        else:
            d, value_col = prepare_saxs_points_parameter(
                saxs_points=saxs_points,
                parameter=parameter,
                trim_std_devs=trim_std_devs,
                require_regions=SAXS_REGION_ORDER_4,
            )

        if d is None or d.empty:
            continue

        d = _add_corr_subtype_clean(
            d,
            source_cols=("SubtypeClean", "subtype", "Subtype", "ExportSubtype"),
        )

        d = d[d["region"].isin(SAXS_REGION_ORDER_4)].copy()
        d = d[d["CorrSubtypeClean"].isin(subtypes_to_use)].copy()

        if d.empty:
            continue

        d["Depth"] = d["region"].apply(_region_to_depth)
        d = d.dropna(subset=["Depth"]).copy()

        sample_means = (
            d.groupby(
                ["experiment", "subtype", "CorrSubtypeClean", "Filenumber", "Depth"],
                as_index=False,
                observed=True,
            )[value_col]
            .mean()
            .rename(columns={
                value_col: parameter,
                "CorrSubtypeClean": "SubtypeClean",
            })
        )

        all_rows.append(sample_means)

    if not all_rows:
        return pd.DataFrame()

    out = all_rows[0]

    for nxt in all_rows[1:]:
        out = out.merge(
            nxt,
            on=["experiment", "subtype", "SubtypeClean", "Filenumber", "Depth"],
            how="outer",
        )

    out["NormPos"] = np.where(out["Depth"] == "lower", 25.0, 75.0)

    return out

def make_saxs_peak_spread_upper_lower_summary(
    saxs_points,
    *,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
    min_rsq=0.3,
    gap_thresh_deg=20.0,
):
    """
    SAXS peak-position spread in lower/upper wound/dermis regions.
    """

    subtype_order = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)

    spread_samples, _ = calculate_saxs_peak_spread_samples(
        saxs_points=saxs_points,
        min_rsq=min_rsq,
        region_mode="split",
        gap_thresh_deg=gap_thresh_deg,
    )

    if spread_samples.empty:
        return pd.DataFrame()

    d = spread_samples.copy()
    d = d[d["SubtypeClean"].astype(str).isin(subtype_order)].copy()
    d["Depth"] = d["Region"].apply(_region_to_depth)
    d = d.dropna(subset=["Depth"]).copy()
    d["NormPos"] = np.where(d["Depth"] == "lower", 25.0, 75.0)

    return d.rename(columns={"SpreadDegrees": "PeakPositionSpread"})

def _weighted_moments_from_spectrum(x, y, xlim):
    """
    Match the Raman weighted moment logic used previously.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    lo, hi = float(xlim[0]), float(xlim[1])
    m = (x >= lo) & (x <= hi) & np.isfinite(x) & np.isfinite(y)

    if not np.any(m):
        return {
            "m1": np.nan,
            "mu2": np.nan,
            "mu3": np.nan,
            "sigma": np.nan,
            "skewness": np.nan,
            "area_w": np.nan,
            "max_intensity": np.nan,
            "n_points": 0,
        }

    xr = x[m]
    yr = y[m]

    offset = -min(0.0, float(np.nanmin(yr)))
    w = np.clip(yr + offset, 0, None)
    wsum = float(np.nansum(w))

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
            "n_points": int(xr.size),
        }

    m1 = float(np.nansum(w * xr) / wsum)
    mu2 = float(np.nansum(w * (xr - m1) ** 2) / wsum)
    mu3 = float(np.nansum(w * (xr - m1) ** 3) / wsum)
    sigma = float(np.sqrt(mu2)) if np.isfinite(mu2) and mu2 >= 0 else np.nan
    skew = float(mu3 / (mu2 ** 1.5)) if np.isfinite(mu2) and mu2 > 0 and np.isfinite(mu3) else np.nan
    area_w = float(np.trapezoid(w, xr)) if xr.size > 1 else np.nan

    return {
        "m1": m1,
        "mu2": mu2,
        "mu3": mu3,
        "sigma": sigma,
        "skewness": skew,
        "area_w": area_w,
        "max_intensity": max_intensity,
        "n_points": int(xr.size),
    }

def make_raman_binned_weighted_moments_from_spectra(
    raman_spectra,
    *,
    peak_regions=RAMAN_PEAK_REGIONS,
    nbins=2,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
):
    """
    Calculate Raman weighted moments from binned point spectra.

    Input should be raman_spectra from prepare_raman_point_spectra().

    Correlation-safe:
        - accepts CT/D7/D14 or control/d7/d14
        - uses existing NormPos if present
        - creates NormPos internally if missing
        - outputs harmonised SubtypeClean for merging
    """

    if raman_spectra is None or raman_spectra.empty:
        return pd.DataFrame()

    d = raman_spectra.copy()
    d.columns = d.columns.astype(str).str.strip()

    required = [
        "Sample",
        "Subtype",
        "PointIndex",
        "x",
        "y",
        "WaveRound",
        "Intensity",
    ]

    missing = [c for c in required if c not in d.columns]
    if missing:
        raise KeyError(f"raman_spectra missing required columns: {missing}")

    d = _add_corr_subtype_clean(d)
    subtypes_to_use = _resolve_corr_subtypes(
        subtype_order=subtype_order,
        subtypes_to_plot=subtypes_to_plot,
    )

    for col in ["Sample", "PointIndex", "x", "y", "WaveRound", "Intensity"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.dropna(
        subset=["Sample", "PointIndex", "x", "y", "WaveRound", "Intensity"]
    ).copy()

    d = d[d["CorrSubtypeClean"].isin(subtypes_to_use)].copy()

    if d.empty:
        print("No Raman spectra found for requested subtypes.")
        return pd.DataFrame()

    # Use existing NormPos if your Raman prep already created it.
    # Otherwise create it locally.
    if "NormPos" not in d.columns:
        point_cols = ["Sample", "Subtype", "CorrSubtypeClean", "PointIndex", "x", "y"]

        points = (
            d[point_cols]
            .drop_duplicates()
            .sort_values(["Sample", "Subtype", "PointIndex", "x", "y"])
            .copy()
        )

        norm_rows = []

        for (sample, subtype), g in points.groupby(["Sample", "Subtype"], observed=True):
            g = g.sort_values(["PointIndex", "x", "y"]).copy()
            n = len(g)

            g["NormPos"] = 0.0 if n <= 1 else np.linspace(0, 100, n)
            norm_rows.append(g)

        if not norm_rows:
            print("Could not create Raman normalised positions.")
            return pd.DataFrame()

        pos_df = pd.concat(norm_rows, ignore_index=True)

        d = d.merge(
            pos_df[point_cols + ["NormPos"]],
            on=point_cols,
            how="left",
        )

    d["NormPos"] = pd.to_numeric(d["NormPos"], errors="coerce")
    d = d.dropna(subset=["NormPos"]).copy()

    if d.empty:
        print("No Raman spectra left after assigning NormPos.")
        return pd.DataFrame()

    bins = np.linspace(0, 100, nbins + 1)

    d["Bin"] = pd.cut(
        d["NormPos"],
        bins=bins,
        labels=np.arange(1, nbins + 1),
        include_lowest=True,
    ).astype(int)

    d["Depth"] = d["Bin"].apply(lambda b: _bin_to_depth(b, nbins=nbins))

    rows = []

    for (subtype, sample, b, depth), g in d.groupby(
        ["CorrSubtypeClean", "Sample", "Bin", "Depth"],
        observed=True,
    ):
        mean_spec = (
            g.groupby("WaveRound", as_index=False, observed=True)["Intensity"]
            .mean()
            .sort_values("WaveRound")
        )

        x = mean_spec["WaveRound"].to_numpy(float)
        y = mean_spec["Intensity"].to_numpy(float)

        for peak_name, limits in peak_regions:
            moms = _weighted_moments_from_spectrum(x, y, limits)

            row = {
                "SubtypeClean": str(subtype),
                "Sample": sample,
                "Bin": int(b),
                "Depth": depth,
                "NormPos": float(np.nanmean(g["NormPos"])),
                "PeakRegion": peak_name,
            }

            row.update(moms)
            rows.append(row)

    return pd.DataFrame(rows)

def make_raman_amide_ratios_from_binned_wm(
    raman_binned_wm,
    *,
    amideI_regions=("AmideI_LEFT_1530_1590", "AmideI_MIDDLE_1590_1635", "AmideI_RIGHT_1635_1700"),
    amideIII_region="AmideIII_1410_1500",
    amideI_height_mode="max",
):
    """
    Calculate Amide I / Amide III height and area ratios from binned Raman WM data.

    Amide I height:
        max across Amide I subregions by default.

    Amide I area:
        sum across Amide I subregions.
    """

    if raman_binned_wm is None or raman_binned_wm.empty:
        return pd.DataFrame()

    d = raman_binned_wm.copy()

    id_cols = ["SubtypeClean", "Sample", "Bin", "Depth", "NormPos"]

    rows = []

    for keys, g in d.groupby(id_cols, observed=True):
        row = dict(zip(id_cols, keys))

        amideI = g[g["PeakRegion"].isin(amideI_regions)]
        amideIII = g[g["PeakRegion"] == amideIII_region]

        if amideI.empty or amideIII.empty:
            continue

        if amideI_height_mode == "max":
            amideI_height = amideI["max_intensity"].max()
        elif amideI_height_mode == "sum":
            amideI_height = amideI["max_intensity"].sum()
        else:
            raise ValueError("amideI_height_mode must be 'max' or 'sum'.")

        amideI_area = amideI["area_w"].sum()
        amideIII_height = amideIII["max_intensity"].iloc[0]
        amideIII_area = amideIII["area_w"].iloc[0]

        row["AmideI_height_combined"] = amideI_height
        row["AmideIII_height"] = amideIII_height
        row["AmideI_area_combined"] = amideI_area
        row["AmideIII_area"] = amideIII_area

        row["AmideI_III_height_ratio"] = (
            amideI_height / amideIII_height
            if np.isfinite(amideIII_height) and amideIII_height != 0
            else np.nan
        )

        row["AmideI_III_area_ratio"] = (
            amideI_area / amideIII_area
            if np.isfinite(amideIII_area) and amideIII_area != 0
            else np.nan
        )

        rows.append(row)

    return pd.DataFrame(rows)

def expand_crossmodal_values(
    left,
    right,
    *,
    group_cols=("SubtypeClean", "Depth"),
    left_value_col,
    right_value_col,
    left_keep_cols=None,
    right_keep_cols=None,
):
    """
    Expand two modality tables within matched groups.

    This creates all pairwise combinations within each group, e.g.
        SubtypeClean + Depth

    Useful when modalities are not measured on exactly the same sample,
    but are matched approximately by subtype and spatial region.

    Example:
        8 cell values x 3 SAXS values = 24 plotted points
        for that subtype/depth group.
    """

    if left is None or left.empty or right is None or right.empty:
        return pd.DataFrame()

    left_keep_cols = list(left_keep_cols or [])
    right_keep_cols = list(right_keep_cols or [])

    group_cols = list(group_cols)

    left_cols = group_cols + [left_value_col] + left_keep_cols
    right_cols = group_cols + [right_value_col] + right_keep_cols

    left_cols = [c for c in left_cols if c in left.columns]
    right_cols = [c for c in right_cols if c in right.columns]

    l = left[left_cols].copy()
    r = right[right_cols].copy()

    l[left_value_col] = pd.to_numeric(l[left_value_col], errors="coerce")
    r[right_value_col] = pd.to_numeric(r[right_value_col], errors="coerce")

    l = l.dropna(subset=group_cols + [left_value_col]).copy()
    r = r.dropna(subset=group_cols + [right_value_col]).copy()

    if l.empty or r.empty:
        return pd.DataFrame()

    l["_pair_key"] = l.groupby(group_cols, observed=True).cumcount()
    r["_pair_key"] = r.groupby(group_cols, observed=True).cumcount()

    # True all-pairs expansion within each group
    l["_tmp"] = 1
    r["_tmp"] = 1

    out = l.merge(
        r,
        on=group_cols + ["_tmp"],
        how="inner",
        suffixes=("_left", "_right"),
    )

    out = out.drop(columns=["_tmp", "_pair_key_left", "_pair_key_right"], errors="ignore")

    return out

def make_cell_upper_lower_values(
    cell_raw,
    *,
    value_col=CELL_VALUE_COL,
    subtype_order=SUBTYPE_ORDER,
    subtypes_to_plot=None,
):
    """
    Return raw cell-density values with lower/upper depth labels.

    This keeps every individual fibroblast-density value.
    """

    subtype_order = resolve_subtypes_to_plot(subtype_order, subtypes_to_plot)

    d = prepare_cell_raw(cell_raw, value_col=value_col)

    if d.empty:
        return pd.DataFrame()

    d = d[d["SubtypeClean"].astype(str).isin(subtype_order)].copy()
    d["Depth"] = d["Region"].apply(_region_to_depth)
    d = d.dropna(subset=["Depth"]).copy()

    d = d.rename(columns={value_col: "CellDensity"})
    d["NormPos"] = np.where(d["Depth"] == "lower", 25.0, 75.0)

    return d[["SubtypeClean", "Depth", "Region", "Sample", "CellDensity", "NormPos"]]

# =============================================================================
#                       10 MULTIMODAL CORRELATION FIGURES
# =============================================================================

def corr_1_fibroblast_density_vs_modulus(
    cell_raw,
    ni_raw,
    *,
    subtypes_to_plot=None,
    panel_by_subtype=False,
    hatches=False,
):
    """
    1. Fibroblast density vs indentation modulus.

    Matching axis:
        lower/upper regions.
    """

    cell = make_cell_upper_lower_summary(
        cell_raw,
        subtypes_to_plot=subtypes_to_plot,
    )

    ni = make_ni_upper_lower_summary(
        ni_raw,
        variables=("mod_Hertz",),
        subtypes_to_plot=subtypes_to_plot,
    )

    if cell.empty or ni.empty:
        print("Missing cell or NI data.")
        return None, None, pd.DataFrame()

    ni_summary = (
        ni.groupby(["SubtypeClean", "Depth"], as_index=False, observed=True)
        .agg(mod_Hertz=("mod_Hertz", "mean"), NormPos=("NormPos", "mean"))
    )

    corr_df = cell.merge(
        ni_summary,
        on=["SubtypeClean", "Depth"],
        how="inner",
    )

    return _plot_correlation(
        corr_df,
        x_col="CellDensity",
        y_col="mod_Hertz",
        colour_by="NormPos",
        subtypes_to_plot=subtypes_to_plot,
        panel_by_subtype=panel_by_subtype,
        xlabel="Fibroblast density (cells / mm²)",
        ylabel="Indentation modulus (kPa)",
        title="Fibroblast density vs indentation modulus",
    )

def corr_2_fibroblast_density_vs_tau(
    cell_raw,
    ni_raw,
    *,
    subtypes_to_plot=None,
    panel_by_subtype=False,
):
    """
    2. Fibroblast density vs tau.

    Matching axis:
        lower/upper regions.
    """

    cell = make_cell_upper_lower_summary(
        cell_raw,
        subtypes_to_plot=subtypes_to_plot,
    )

    ni = make_ni_upper_lower_summary(
        ni_raw,
        variables=("tau_Visco",),
        subtypes_to_plot=subtypes_to_plot,
    )

    if cell.empty or ni.empty:
        print("Missing cell or NI data.")
        return None, None, pd.DataFrame()

    ni_summary = (
        ni.groupby(["SubtypeClean", "Depth"], as_index=False, observed=True)
        .agg(tau_Visco=("tau_Visco", "mean"), NormPos=("NormPos", "mean"))
    )

    corr_df = cell.merge(
        ni_summary,
        on=["SubtypeClean", "Depth"],
        how="inner",
    )

    return _plot_correlation(
        corr_df,
        x_col="CellDensity",
        y_col="tau_Visco",
        colour_by="NormPos",
        subtypes_to_plot=subtypes_to_plot,
        panel_by_subtype=panel_by_subtype,
        xlabel="Fibroblast density (cells / mm²)",
        ylabel="Tau (s)",
        title="Fibroblast density vs tau",
    )

def corr_3_amide_ratio_vs_stiffness(
    raman_spectra,
    ni_raw,
    *,
    nbins=10,
    subtypes_to_plot=None,
    panel_by_subtype=True,
):
    """
    3. Amide I/III ratio vs stiffness.

    Matching axis:
        both Raman and NI are binned on normalised position.

    Accepts:
        subtypes_to_plot=["CT", "D7", "D14"]
    or:
        subtypes_to_plot=["control", "d7", "d14"]
    """

    subtypes_clean = _normalise_corr_subtypes_to_plot(subtypes_to_plot)

    rwm = make_raman_binned_weighted_moments_from_spectra(
        raman_spectra,
        nbins=nbins,
        subtypes_to_plot=subtypes_clean,
    )

    ratios = make_raman_amide_ratios_from_binned_wm(rwm)

    ni = make_ni_binned_summary(
        ni_raw,
        variables=("mod_Hertz",),
        nbins=nbins,
        subtypes_to_plot=subtypes_clean,
    )

    if ratios.empty or ni.empty:
        print("Missing Raman ratio or NI data.")
        return None, None, pd.DataFrame()

    raman_summary = (
        ratios.groupby(["SubtypeClean", "Bin"], as_index=False, observed=True)
        .agg(
            AmideI_III_height_ratio=("AmideI_III_height_ratio", "mean"),
            AmideI_III_area_ratio=("AmideI_III_area_ratio", "mean"),
            NormPos=("NormPos", "mean"),
        )
    )

    ni_summary = (
        ni.groupby(["SubtypeClean", "Bin"], as_index=False, observed=True)
        .agg(mod_Hertz=("mod_Hertz", "mean"))
    )

    corr_df = raman_summary.merge(
        ni_summary,
        on=["SubtypeClean", "Bin"],
        how="inner",
    )

    return _plot_correlation(
        corr_df,
        x_col="AmideI_III_height_ratio",
        y_col="mod_Hertz",
        colour_by="NormPos",
        subtypes_to_plot=subtypes_clean,
        panel_by_subtype=panel_by_subtype,
        xlabel="Amide I / Amide III height ratio",
        ylabel="Indentation modulus (kPa)",
        title="Amide I/III ratio vs stiffness",
    )

def corr_4_amideI_height_vs_saxs_collagen(
    raman_spectra,
    saxs_points,
    *,
    nbins=2,
    subtypes_to_plot=None,
    panel_by_subtype=False,
):
    """
    4. Amide I height vs SAXS total collagen intensity.

    Matching axis:
        lower/upper regions.

    Accepts:
        subtypes_to_plot=["CT", "D7", "D14"]
    or:
        subtypes_to_plot=["control", "d7", "d14"]
    """

    subtypes_clean = _normalise_corr_subtypes_to_plot(subtypes_to_plot)

    rwm = make_raman_binned_weighted_moments_from_spectra(
        raman_spectra,
        nbins=nbins,
        subtypes_to_plot=subtypes_clean,
    )

    if rwm.empty:
        print("Missing Raman binned WM data.")
        return None, None, pd.DataFrame()

    amideI_regions = [
        "AmideI_LEFT_1530_1590",
        "AmideI_MIDDLE_1590_1635",
        "AmideI_RIGHT_1635_1700",
    ]

    raman = (
        rwm[rwm["PeakRegion"].isin(amideI_regions)]
        .groupby(["SubtypeClean", "Depth"], as_index=False, observed=True)
        .agg(
            AmideI_height=("max_intensity", "max"),
            NormPos=("NormPos", "mean"),
        )
    )

    saxs = make_saxs_upper_lower_summary(
        saxs_points,
        parameters=("curvearea",),
        subtypes_to_plot=subtypes_clean,
        normalise_curvearea=True,
    )

    if saxs.empty:
        print("Missing SAXS collagen data.")
        return None, None, pd.DataFrame()

    saxs_summary = (
        saxs.groupby(["SubtypeClean", "Depth"], as_index=False, observed=True)
        .agg(curvearea=("curvearea", "mean"))
    )

    corr_df = raman.merge(
        saxs_summary,
        on=["SubtypeClean", "Depth"],
        how="inner",
    )

    if corr_df.empty:
        print("No matched Raman/SAXS rows after merging by SubtypeClean and Depth.")
        return None, None, pd.DataFrame()

    return _plot_correlation(
        corr_df,
        x_col="AmideI_height",
        y_col="curvearea",
        colour_by="NormPos",
        subtypes_to_plot=subtypes_clean,
        panel_by_subtype=panel_by_subtype,
        xlabel="Amide I peak height",
        ylabel="SAXS total collagen intensity",
        title="Amide I height vs SAXS collagen intensity",
    )

def corr_5_fibroblast_density_vs_saxs_collagen(
    cell_raw,
    saxs_points,
    *,
    subtypes_to_plot=None,
    panel_by_subtype=False,
):
    """
    5. Fibroblast density vs SAXS total collagen intensity.

    Uses all raw cell-density values and all SAXS sample/depth values.
    """

    cell = make_cell_upper_lower_values(
        cell_raw,
        subtypes_to_plot=subtypes_to_plot,
    )

    saxs = make_saxs_upper_lower_summary(
        saxs_points,
        parameters=("curvearea",),
        subtypes_to_plot=subtypes_to_plot,
        normalise_curvearea=True,
    )

    if cell.empty or saxs.empty:
        print("Missing cell or SAXS data.")
        return None, None, pd.DataFrame()

    saxs_values = saxs.rename(columns={"curvearea": "SAXS_collagen"}).copy()

    corr_df = expand_crossmodal_values(
        left=cell,
        right=saxs_values,
        group_cols=("SubtypeClean", "Depth"),
        left_value_col="CellDensity",
        right_value_col="SAXS_collagen",
        left_keep_cols=["NormPos"],
        right_keep_cols=["Filenumber"],
    )

    if corr_df.empty:
        print("No matched cell/SAXS rows after expanding by SubtypeClean and Depth.")
        return None, None, pd.DataFrame()

    if "NormPos" not in corr_df.columns:
        corr_df["NormPos"] = np.where(corr_df["Depth"] == "lower", 25.0, 75.0)

    return _plot_correlation(
        corr_df,
        x_col="CellDensity",
        y_col="SAXS_collagen",
        colour_by="NormPos",
        subtypes_to_plot=subtypes_to_plot,
        panel_by_subtype=panel_by_subtype,
        xlabel="Fibroblast density (cells / mm²)",
        ylabel="SAXS total collagen intensity",
        title="Fibroblast density vs SAXS collagen intensity",
    )

def corr_6_amideIII_wm1_shift_vs_saxs_dperiod(
    raman_spectra,
    saxs_points,
    *,
    nbins=2,
    control_subtype="CT",
    subtypes_to_plot=None,
    panel_by_subtype=False,
):
    """
    6. Amide III 1410-1500 WM1 shift vs SAXS D-period.

    Matching axis:
        lower/upper regions.
    """

    rwm = make_raman_binned_weighted_moments_from_spectra(
        raman_spectra,
        nbins=nbins,
        subtypes_to_plot=subtypes_to_plot,
    )

    if rwm.empty:
        print("Missing Raman binned WM data.")
        return None, None, pd.DataFrame()

    raman = rwm[rwm["PeakRegion"] == "AmideIII_1410_1500"].copy()

    control_means = (
        raman[raman["SubtypeClean"].astype(str) == control_subtype]
        .groupby("Depth", as_index=False, observed=True)["m1"]
        .mean()
        .rename(columns={"m1": "Control_m1"})
    )

    raman = raman.merge(control_means, on="Depth", how="left")
    raman["AmideIII_m1_shift"] = raman["m1"] - raman["Control_m1"]

    raman_summary = (
        raman.groupby(["SubtypeClean", "Depth"], as_index=False, observed=True)
        .agg(
            AmideIII_m1_shift=("AmideIII_m1_shift", "mean"),
            NormPos=("NormPos", "mean"),
        )
    )

    saxs = make_saxs_upper_lower_summary(
        saxs_points,
        parameters=("D_period",),
        subtypes_to_plot=subtypes_to_plot,
    )

    if saxs.empty:
        print("Missing SAXS D-period data.")
        return None, None, pd.DataFrame()

    saxs_summary = (
        saxs.groupby(["SubtypeClean", "Depth"], as_index=False, observed=True)
        .agg(D_period=("D_period", "mean"))
    )

    corr_df = raman_summary.merge(
        saxs_summary,
        on=["SubtypeClean", "Depth"],
        how="inner",
    )

    return _plot_correlation(
        corr_df,
        x_col="AmideIII_m1_shift",
        y_col="D_period",
        colour_by="NormPos",
        subtypes_to_plot=subtypes_to_plot,
        panel_by_subtype=panel_by_subtype,
        xlabel="Amide III WM1 shift from control (cm$^{-1}$)",
        ylabel="SAXS D-period (nm)",
        title="Amide III WM1 shift vs SAXS D-period",
    )

def corr_7_saxs_wa_vs_tau(
    saxs_points,
    ni_raw,
    *,
    subtypes_to_plot=None,
    panel_by_subtype=False,
):
    """
    7. SAXS wa vs tau.

    Matching axis:
        lower/upper regions.
    """

    saxs = make_saxs_upper_lower_summary(
        saxs_points,
        parameters=(SAXS_WA_PARAM,),
        subtypes_to_plot=subtypes_to_plot,
    )

    ni = make_ni_upper_lower_summary(
        ni_raw,
        variables=("tau_Visco",),
        subtypes_to_plot=subtypes_to_plot,
    )

    if saxs.empty or ni.empty:
        print("Missing SAXS wa or NI tau data.")
        return None, None, pd.DataFrame()

    saxs_summary = (
        saxs.groupby(["SubtypeClean", "Depth"], as_index=False, observed=True)
        .agg(wa=(SAXS_WA_PARAM, "mean"), NormPos=("NormPos", "mean"))
    )

    ni_summary = (
        ni.groupby(["SubtypeClean", "Depth"], as_index=False, observed=True)
        .agg(tau_Visco=("tau_Visco", "mean"))
    )

    corr_df = saxs_summary.merge(
        ni_summary,
        on=["SubtypeClean", "Depth"],
        how="inner",
    )

    return _plot_correlation(
        corr_df,
        x_col="wa",
        y_col="tau_Visco",
        colour_by="NormPos",
        subtypes_to_plot=subtypes_to_plot,
        panel_by_subtype=panel_by_subtype,
        xlabel="SAXS wa parameter",
        ylabel="Tau (s)",
        title="SAXS wa vs tau",
    )

def corr_8_saxs_collagen_vs_peak_position_spread(
    saxs_points,
    *,
    subtypes_to_plot=None,
    panel_by_subtype=False,
    min_rsq=0.3,
):
    """
    8. SAXS total collagen vs SAXS peak-position spread.

    Matching axis:
        lower/upper SAXS regions.
    """

    collagen = make_saxs_upper_lower_summary(
        saxs_points,
        parameters=("curvearea",),
        subtypes_to_plot=subtypes_to_plot,
        normalise_curvearea=True,
    )

    spread = make_saxs_peak_spread_upper_lower_summary(
        saxs_points,
        subtypes_to_plot=subtypes_to_plot,
        min_rsq=min_rsq,
    )

    if collagen.empty or spread.empty:
        print("Missing SAXS collagen or peak-position spread data.")
        return None, None, pd.DataFrame()

    collagen_summary = (
        collagen.groupby(["SubtypeClean", "Filenumber", "Depth"], as_index=False, observed=True)
        .agg(curvearea=("curvearea", "mean"), NormPos=("NormPos", "mean"))
    )

    spread_summary = (
        spread.groupby(["SubtypeClean", "Filenumber", "Depth"], as_index=False, observed=True)
        .agg(PeakPositionSpread=("PeakPositionSpread", "mean"))
    )

    corr_df = collagen_summary.merge(
        spread_summary,
        on=["SubtypeClean", "Filenumber", "Depth"],
        how="inner",
    )

    return _plot_correlation(
        corr_df,
        x_col="curvearea",
        y_col="PeakPositionSpread",
        colour_by="NormPos",
        subtypes_to_plot=subtypes_to_plot,
        panel_by_subtype=panel_by_subtype,
        xlabel="SAXS total collagen intensity",
        ylabel="Peak-position spread (degrees)",
        title="SAXS collagen intensity vs peak-position spread",
    )

def corr_9_raman_1330_1380_height_vs_stiffness(
    raman_spectra,
    ni_raw,
    *,
    nbins=10,
    subtypes_to_plot=None,
    panel_by_subtype=True,
):
    """
    9. Raman 1310-1380 peak height vs stiffness.

    Matching axis:
        Raman and NI binned by normalised position.
    """

    rwm = make_raman_binned_weighted_moments_from_spectra(
        raman_spectra,
        nbins=nbins,
        subtypes_to_plot=subtypes_to_plot,
    )

    if rwm.empty:
        print("Missing Raman binned WM data.")
        return None, None, pd.DataFrame()

    raman = (
        rwm[rwm["PeakRegion"] == "CH2CH3_RIGHT_1330_1380"]
        .groupby(["SubtypeClean", "Bin"], as_index=False, observed=True)
        .agg(
            Peak1350_height=("max_intensity", "mean"),
            Peak1350_area=("area_w", "mean"),
            NormPos=("NormPos", "mean"),
        )
    )

    ni = make_ni_binned_summary(
        ni_raw,
        variables=("mod_Hertz",),
        nbins=nbins,
        subtypes_to_plot=subtypes_to_plot,
    )

    if ni.empty:
        print("Missing NI stiffness data.")
        return None, None, pd.DataFrame()

    ni_summary = (
        ni.groupby(["SubtypeClean", "Bin"], as_index=False, observed=True)
        .agg(mod_Hertz=("mod_Hertz", "mean"))
    )

    corr_df = raman.merge(
        ni_summary,
        on=["SubtypeClean", "Bin"],
        how="inner",
    )

    return _plot_correlation(
        corr_df,
        x_col="Peak1350_height",
        y_col="mod_Hertz",
        colour_by="NormPos",
        subtypes_to_plot=subtypes_to_plot,
        panel_by_subtype=panel_by_subtype,
        xlabel="1330–1380 cm$^{-1}$ peak height",
        ylabel="Indentation modulus (kPa)",
        title="1330–1380 cm$^{-1}$ peak height vs stiffness",
    )

def corr_10_saxs_collagen_vs_modulus(
    saxs_points,
    ni_raw,
    *,
    subtypes_to_plot=None,
    panel_by_subtype=False,
):
    """
    10. SAXS total collagen vs indentation modulus.

    Matching axis:
        lower/upper regions.
    """

    saxs = make_saxs_upper_lower_summary(
        saxs_points,
        parameters=("curvearea",),
        subtypes_to_plot=subtypes_to_plot,
        normalise_curvearea=True,
    )

    ni = make_ni_upper_lower_summary(
        ni_raw,
        variables=("mod_Hertz",),
        subtypes_to_plot=subtypes_to_plot,
    )

    if saxs.empty or ni.empty:
        print("Missing SAXS collagen or NI stiffness data.")
        return None, None, pd.DataFrame()

    saxs_summary = (
        saxs.groupby(["SubtypeClean", "Depth"], as_index=False, observed=True)
        .agg(curvearea=("curvearea", "mean"), NormPos=("NormPos", "mean"))
    )

    ni_summary = (
        ni.groupby(["SubtypeClean", "Depth"], as_index=False, observed=True)
        .agg(mod_Hertz=("mod_Hertz", "mean"))
    )

    corr_df = saxs_summary.merge(
        ni_summary,
        on=["SubtypeClean", "Depth"],
        how="inner",
    )

    return _plot_correlation(
        corr_df,
        x_col="curvearea",
        y_col="mod_Hertz",
        colour_by="NormPos",
        subtypes_to_plot=subtypes_to_plot,
        panel_by_subtype=panel_by_subtype,
        xlabel="SAXS total collagen intensity",
        ylabel="Indentation modulus (kPa)",
        title="SAXS collagen intensity vs stiffness",
    )
# ====================================================================================================================================================
# ====================================================================================================================================================


# =============================================================================
# Read all exported data
# =============================================================================

multitech_data, combined_data = read_multitech_workbooks(
    data_root=DATA_ROOT,
    subtypes=SUBTYPES,
    sheets=SHEETS,
)

# =============================================================================
# Resolve old/new workbook sheet names
# =============================================================================

raman_df = combined_data.get("Raman", pd.DataFrame())
raw_raman_df = combined_data.get("RawRaman", pd.DataFrame())
cell_df = combined_data.get("Cell", pd.DataFrame())
saxs_df = combined_data.get("SAXS", pd.DataFrame())

# New NI export format
ni_linescan_new = combined_data.get("NI_LineScan", pd.DataFrame())
raw_ni_linescan_new = combined_data.get("RawNI_LineScan", pd.DataFrame())

# Old NI export format, e.g. control.xlsx
ni_linescan_old = combined_data.get("Nanoindentation", pd.DataFrame())
raw_ni_linescan_old = combined_data.get("RawNanoindentation", pd.DataFrame())

# Treat old Nanoindentation as line-scan NI
ni_df = pd.concat(
    [ni_linescan_new, ni_linescan_old],
    ignore_index=True,
    sort=False,
)

raw_ni_df = pd.concat(
    [raw_ni_linescan_new, raw_ni_linescan_old],
    ignore_index=True,
    sort=False,
)

# Grid NI only exists in newer bleo files
ni_grid_df = combined_data.get("NI_Grid", pd.DataFrame())
raw_ni_grid_df = combined_data.get("RawNI_Grid", pd.DataFrame())



# print("Loaded data:")
# for sheet, df in combined_data.items():
#     print(f"{sheet}: {df.shape[0]} rows x {df.shape[1]} columns")
    
# preview_loaded_data(combined_data)

tables = split_loaded_tables(
    raman_df=raman_df,
    raw_raman_df=raw_raman_df,
    cell_df=cell_df,
    saxs_df=saxs_df,
    ni_df=ni_df,
    raw_ni_df=raw_ni_df,
)

tables["ni_grid"] = ni_grid_df.copy()
tables["ni_grid_raw"] = raw_ni_grid_df.copy()

for key in list(tables):
    tables[key] = add_clean_subtype(tables[key])
    
# print("Split analysis tables:")
# for name, df in tables.items():
#     print(f"{name}: {df.shape[0]} rows x {df.shape[1]} columns")

for key in list(tables):
    tables[key] = add_clean_subtype(tables[key])
    
raman_spectra = prepare_raman_point_spectra(
    raw_raman_df,
    spectral_region="FP",
    anatomical_region=None,   # leave as None because export already selected dermis/wound
    wave_min=1200,
    wave_max=1800,
)

raman_wm_long = tidy_raman_weighted_moments(raman_df)

ni_raw_screened = screen_ni_raw_dataframe(
    tables["ni_raw"],
    variables=("mod_Hertz", "tau_Visco"),
    rsq_min_hertz=0.5,
    rsq_min_visco=0.5,
    hi_modulus_pa=100000,
    lo_modulus_pa=0,
    std_devs=3,
)

ni_grid_raw_screened = screen_ni_raw_dataframe(
    tables["ni_grid_raw"],
    variables=("mod_Hertz", "tau_Visco"),
    rsq_min_hertz=0.5,
    rsq_min_visco=0.5,
    hi_modulus_pa=100000,
    lo_modulus_pa=0,
    std_devs=3,
)


# ====================================================================================================================================================
# ====================================================================================================================================================
# ====================================================================================================================================================

# =============================================================================
#                               NANOINDENTATION
# =============================================================================

# ------- NI normalised position vs Modulus (raw)
fig, ax, ni_modulus_smooth = plot_ni_raw_linescan_by_subtype_smooth(
    ni_raw=ni_raw_screened,
    variable="mod_Hertz",
    ylabel="Indentation modulus (kPa)",
    title=" ",
    subtypes_to_plot=["pbsokn",  "4w", "4wokn"],
    window=18,
    grid_size=200,
    show_sample_lines=False,
    min_n=0,
    ylim=(0, 50),
    panel=False,
)


# ------- NI normalised position vs Modulus (binned)
fig, ax = plot_ni_linescan_by_subtype(
    ni_binned=tables["ni_binned"],
    variable="mod_Hertz",
    ylabel="Indentation modulus (kPa)",
    title="Nanoindentation modulus across normalised dermal position",
    ylim=(0,10),
)

# ------- NI normalised position vs Tau (raw)
fig, ax, ni_tau_smooth = plot_ni_raw_linescan_by_subtype_smooth(
    ni_raw=ni_raw_screened,
    variable="tau_Visco",
    ylabel="Tau (s)",
    subtypes_to_plot=["pbsokn",  "4w", "4wokn"],
    title="Raw nanoindentation relaxation time across normalised dermal position",
    window=18,
    grid_size=200,
    min_n=2,
    ylim=(0,20),
    panel=False,
)

# ------- NI normalised position vs Tau (binned)
fig, ax = plot_ni_linescan_by_subtype(
    ni_binned=tables["ni_binned"],
    variable="tau_Visco",
    ylabel="Tau (s)",
    title="Nanoindentation relaxation time across normalised dermal position",
    ylim=(0,10),
)


# ------- NI normalised position vs Modulus (raw) Straight line
fig, ax, ni_modulus_fit_df = plot_ni_linear_fit_by_subtype(
    ni_raw=tables["ni_raw"],
    ni_binned=tables["ni_binned"],
    variable="mod_Hertz",
    ylabel="Indentation modulus (kPa)",
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "bmmet", "pbsokn", "4wokn"],
    title="Linear fit of raw modulus across normalised dermal position",
    ylim=(0,10),
    panel=False,
)
# print(ni_modulus_fit_df)

# ------- NI normalised position vs tau (raw) Straight line
fig, ax, ni_tau_fit_df = plot_ni_linear_fit_by_subtype(
    ni_raw=tables["ni_raw"],
    ni_binned=tables["ni_binned"],
    variable="tau_Visco",
    ylabel="Tau (s)",
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "bmmet", "pbsokn", "4wokn"],
    title="Linear fit of raw tau across normalised dermal position",
    ylim=(0,10),
    panel=False,
)
# print(ni_tau_fit_df)

# ------- NI bin bar plot
fig, ax, ni_modulus_binned_summary = plot_ni_binned_bar_by_position(
    ni_raw=ni_raw_screened,
    variable="mod_Hertz",
    ylabel="Indentation modulus (kPa)",
    title="Binned nanoindentation modulus across dermal position",
    n_position_bins=5,
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "bmmet", "pbsokn", "4wokn"],
    errorbar="std",
)

subtypes_to_plot = ["pbs", "2w", "4w"]

subtypes_to_plot=[ "pbsmet" , "4w", "bmmet"]
subtypes_to_plot=[ "pbsokn" , "4w", "4wokn"]

# fig, ax, ni_grid_modulus_sample, ni_grid_modulus_summary = plot_ni_grid_upper_lower_bar(
#     ni_grid=ni_grid_raw_screened,
#     variable="mod_Hertz",
#     ylabel="Indentation modulus (kPa)",
#     title="",
#     subtypes_to_plot=[ "pbsmet" , "4w", "bmmet"],
#     regions_to_plot=("lower dermis", "upper dermis"),
#     errorbar="std",
#     ylim=(0, 50),
#     hatches=True,
#     show_points=False,
# )

fig, ax, ni_grid_sample_summary, ni_grid_group_summary, ni_grid_stats = plot_ni_grid_upper_lower_bar(
    ni_grid=ni_grid_raw_screened,
    variable="mod_Hertz",
    subtypes_to_plot=subtypes_to_plot,
    regions_to_plot=("lower dermis", "upper dermis"),
    errorbar="std",
    title=' ',
    ylim=(0, 50),
    hatches=True,
    show_points=False,
    run_stats=True,
    print_stats=True,
)

fig, ax, ni_grid_tau_sample_summary, ni_grid_tau_group_summary, ni_grid_tau_stats = plot_ni_grid_upper_lower_bar(
    ni_grid=ni_grid_raw_screened,
    variable="tau_Visco",
    ylabel="Tau (s)",
    title="",
    subtypes_to_plot=subtypes_to_plot,
    regions_to_plot=("lower dermis", "upper dermis"),
    errorbar="std",
    ylim=(0, 20),
    hatches=True,
    show_points=False,
    run_stats=True,
    print_stats=True,
)

bleo_ni_groups_for_diagnostic = {
    "Control / PBS / bleomycin": ["control", "pbs"],
    "bleomycin": ["2w", "4w"],
    "Metformin": ["pbsmet", "bmmet"],
    "OKN": ["pbsokn", "4wokn"],
}

bleo_ni_groups_for_diagnostic = {
    "Control / PBS / bleomycin": ["pbsmet", "bmmet"],
    }
ni_repeat_modulus_outputs = plot_ni_repeat_diagnostic_by_group(
    ni_raw=ni_raw_screened,
    ni_grid_raw=ni_grid_raw_screened,
    variable="mod_Hertz",
    groups=bleo_ni_groups_for_diagnostic,
    ylabel="Indentation modulus (kPa)",
    title_prefix="Bleomycin NI repeat diagnostic: modulus",
    ylim=(0, 100),
    errorbar="std",
    show_points=True,
    hatches=True,
    scatter_by_rsq=True,
    rsq_cmap="jet",
    rsq_vmin=0,
    rsq_vmax=1,
)

ni_repeat_tau_outputs = plot_ni_repeat_diagnostic_by_group(
    ni_raw=ni_raw_screened,
    ni_grid_raw=ni_grid_raw_screened,
    variable="tau_Visco",
    groups=bleo_ni_groups_for_diagnostic,
    ylabel="Tau (s)",
    title_prefix="Bleomycin NI repeat diagnostic: tau",
    ylim=(0, 20),
    errorbar="std",
    show_points=True,
    hatches=True,
    scatter_by_rsq=True,
    rsq_cmap="jet",
    rsq_vmin=0,
    rsq_vmax=1,
)

# ======PCA

ni_pca_df, ni_pca_info = prepare_ni_pca_data(
    ni_raw=ni_raw_screened,
    pca_vars=("mod_Hertz", "tau_Visco", "NormalisedPosition"),
    subtypes_to_plot=["pbsokn", "4w",  "4wokn"],
    standardise=True,
)

ni_pca_loadings = print_ni_pca_loadings(
    ni_pca_info,
    pca_vars=("mod_Hertz", "tau_Visco", "NormalisedPosition"),
)

ni_grid_region_pca_df, ni_grid_region_pca_info = prepare_ni_grid_region_pca_data(
    ni_grid_raw=ni_grid_raw_screened,
    pca_vars=("mod_Hertz", "tau_Visco", "RegionCode"),
    subtypes_to_plot=["pbsokn", "4w",  "4wokn"],
    standardise=True,
)

ni_grid_region_pca_loadings = print_ni_pca_loadings(
    ni_grid_region_pca_info,
    pca_vars=("mod_Hertz", "tau_Visco", "RegionCode"),
)

# Available PCA Vars:
# E0_Visco, Einf_Visco, G0_Visco, G1_Visco,
# Hold_LoadEnd, Hold_LoadStart, RelaxFrac,
# Rsq_Hertz, Rsq_OP, Rsq_Visco,
# TimeHeld, mod_Hertz, mod_OP, tau_Visco,
# NormalisedPosition

# ------- PCA scores
fig, ax = plot_ni_pca_scores(
    pca_df=ni_pca_df,
    pca_info=ni_pca_info,
    title="Nanoindentation PCA",
)

# ------- PCA by position (single)
fig, ax = plot_ni_pca_by_position_with_markers(
    pca_df=ni_pca_df,
    pca_info=ni_pca_info,
    title="Nanoindentation PCA coloured by normalised position",
    point_size=55,
    alpha=0.10,
)

fig, ax = plot_ni_pca_by_position_with_alpha(
    pca_df=ni_pca_df,
    pca_info=ni_pca_info,
    subtype_order=["pbs", "2w", "4w", "pbsmet", "bmmet", "pbsokn", "4wokn"],
    colours=NI_COLOURS,
    markers=NI_MARKERS,
    alpha_min=0.20,
    alpha_max=0.90,
    point_size=55,
    title="Nanoindentation PCA: subtype colour and spatial alpha",
)

# ------- PCA by position (multiplot panel)
fig, axes = plot_ni_pca_by_position_faceted(
    pca_df=ni_pca_df,
    pca_info=ni_pca_info,
    ncols=3,
    title=" ",
)

fig, axes = plot_ni_grid_pca_by_region_faceted(
    pca_df=ni_grid_region_pca_df,
    pca_info=ni_grid_region_pca_info,
    ncols=3,
    title="",
)
# print("Explained variance:")
# for i, ev in enumerate(ni_pca_info["explained_variance_ratio"], start=1):
#     print(f"PC{i}: {ev * 100:.2f}%")

# print("\nLoadings:")
# for var, row in zip(ni_pca_info["pca_vars"], ni_pca_info["loadings"]):
#     print(f"{var}: PC1={row[0]:.4f}, PC2={row[1]:.4f}")

# -------PCA Slopes
ni_pc_slopes_df, ni_pc_slope_summary_df = calculate_ni_pc_slopes(
    pca_df=ni_pca_df,
    pcs=("PC1", "PC2"),
)

# ------- PC1 slope vs position
fig, ax = plot_ni_pc_slope_points(
    slopes_df=ni_pc_slopes_df,
    summary_df=ni_pc_slope_summary_df,
    pc="PC1",
    title="PC1 slope across dermal position",
)

# ------- PC2 slope vs position
fig, ax = plot_ni_pc_slope_points(
    slopes_df=ni_pc_slopes_df,
    summary_df=ni_pc_slope_summary_df,
    pc="PC2",
    title="PC2 slope across dermal position",
)
# print(ni_pc_slope_summary_df)

# =============================================================================
#                               CELL
# =============================================================================

fig, ax, cell_spatial_summary, cell_spatial_ttests = plot_cell_spatial_4region_bar(
    cell_raw=tables["cell_raw"],
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "bmmet", "pbsokn", "4wokn"],
    ylim=None,
    hatches=True,
    add_ttests=True,
)
fig, axes, cell_hist_df = plot_cell_spatial_4region_histograms(
    cell_raw=tables["cell_raw"],
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "bmmet", "pbsokn", "4wokn"],
    bins=20,
    density=False,
    ncols=3,
)
# ------- 
fig, ax = plot_cell_dermis_vs_wound_bar(
    cell_raw=tables["cell_raw"],
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "bmmet", "pbsokn", "4wokn"],
    hatches=False,
)

# ------- 
fig, ax = plot_cell_wound_vs_dermis_shift(
    cell_raw=tables["cell_raw"],
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "bmmet", "pbsokn", "4wokn"],
    hatches=False,
)

# ------- 
fig, ax = plot_cell_upper_vs_lower_shift(
    cell_raw=tables["cell_raw"],
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "bmmet", "pbsokn", "4wokn"],
    hatches=False,
)

# =============================================================================
#                               SAXS
# =============================================================================

# ------- SAXS collagen intensity Dermis vs wound (no spatial) bar chart
fig, ax, saxs_collagen_stats = plot_saxs_total_collagen_dermis_vs_wound(
    saxs_points=tables["saxs_points"],
    parameter="curvearea",
    normalise=True,
    trim_std_devs=6,
    subtypes_to_plot=["pbs", "2w", "4w"],
    hatches=False,
    paired_test=True,
    print_stats=True,
)


fig, ax, saxs_collagen_stats = plot_saxs_total_collagen_dermis_vs_wound_pull_norm(
    saxs_points=tables["saxs_points"],
    parameter="curvearea",
    subtypes_to_plot=["pbs", "2w", "4w"],
    curvearea_thresh=0.00,
    saxs_thresh=0.0,
    param_thresh=0,
    trim_std_devs=6,
    stats_test="welch",
    hatches=False,
)

# ------- SAXS collagen intensity Dermis vs wound (WITH spatial) bar chart
fig, ax, saxs_collagen_4region_stats = plot_saxs_total_collagen_4region(
    saxs_points=tables["saxs_points"],
    parameter="curvearea",
    normalise=True,
    trim_std_devs=6,
    subtypes_to_plot=["pbs", "2w", "4w"],
    hatches=True,
    print_summary=True,
    run_stats=True,
    print_stats=True,
)

# ------- SAXS D-period Dermis vs wound (no spatial) bar chart
fig, ax, saxs_dperiod_stats = plot_saxs_dperiod_dermis_vs_wound(
    saxs_points=tables["saxs_points"],
    subtypes_to_plot=["pbs", "2w", "4w"],
    hatches=False,
    print_summary=True,
    run_stats=True,
    print_stats=True,
)

# ------- SAXS D-period Dermis vs wound (WITH spatial) bar chart
fig, ax, saxs_dperiod_4region_stats = plot_saxs_dperiod_4region(
    saxs_points=tables["saxs_points"],
    subtypes_to_plot=["pbs", "2w", "4w"],
    hatches=True,
    print_summary=True,
    run_stats=True,
    print_stats=True,
)

# ------- SAXS D-period % shift from dermis to wound
fig, ax = plot_saxs_dperiod_shift_dermis_to_wound(
    saxs_points=tables["saxs_points"],
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "bmmet", "pbsokn", "4wokn"],
    hatches=False,
)

# ------- SAXS D-period FWHM dermis vs wound (no spatial bar chart)
fig, ax, saxs_dperiod_fwhm_stats = plot_saxs_dperiod_fwhm_dermis_vs_wound(
    saxs_points=tables["saxs_points"],
    subtypes_to_plot=["pbs", "2w", "4w"],
    hatches=False,
    print_summary=True,
    run_stats=True,
    print_stats=True,
)

# ------- SAXS D-period % shift from matched dermis (lower wound vs lower dermis etc)
fig, ax = plot_saxs_dperiod_shift_4region(
    saxs_points=tables["saxs_points"],
    subtypes_to_plot=["pbs", "2w", "4w",],
    hatches=False,
)

# ------- SAXS D-period FWHM (WITH spatial) bar chart
fig, ax, saxs_dperiod_fwhm_4region_stats = plot_saxs_dperiod_fwhm_4region(
    saxs_points=tables["saxs_points"],
    subtypes_to_plot=["pbs", "2w", "4w"],
    hatches=True,
    print_summary=True,
    run_stats=True,
    print_stats=True,
)

# ------- SAXS wa (no spatial) bar chart
fig, ax = plot_saxs_wa_dermis_vs_wound(
    saxs_points=tables["saxs_points"],
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "bmmet", "pbsokn", "4wokn"],
    hatches=False,
)

# ------- SAXS wa (WITH spatial) bar chart
fig, ax = plot_saxs_wa_4region(
    saxs_points=tables["saxs_points"],
)

# ------- SAXS peak position circle plots upper dermis lower wound
fig, axes = plot_saxs_peak_position_circle_by_subtype(
    saxs_points=tables["saxs_points"],
    peak_param="peak_position_canonical",
    min_rsq=0.3,
    ncols=3,
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "bmmet", "pbsokn", "4wokn"],
)

# ------- SAXS peak position bar plot dermis vs wound (no spatial)
fig, ax, saxs_peak_position_stats = plot_saxs_peak_position_bar(
    saxs_points=tables["saxs_points"],
    min_rsq=0.3,
    ylim=(0, 180),
    subtypes_to_plot=["pbs", "2w", "4w"],
    hatches=False,
    print_summary=True,
    run_stats=True,
    print_stats=True,
)

# ------- SAXS peak position bar plot % shift from dermis to wound
fig, ax = plot_saxs_peak_position_shift_from_dermis(
    saxs_points=tables["saxs_points"],
    min_rsq=0.3,
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "bmmet", "pbsokn", "4wokn"],
    hatches=False,
)

# ------- SAXS peak position SPREAD bar plot dermis vs wound (no spatial)
fig, ax, saxs_peak_spread_stats = plot_saxs_peak_spread_dermis_vs_wound(
    saxs_points=tables["saxs_points"],
    subtypes_to_plot=["pbs", "2w", "4w"],
    hatches=False,
    print_summary=True,
    run_stats=True,
    print_stats=True,
)
# ------- SAXS peak position SPREAD bar plot dermis vs wound upper lower(WITH spatial)
fig, ax, saxs_peak_spread_4region_stats = plot_saxs_peak_spread_4region(
    saxs_points=tables["saxs_points"],
    subtypes_to_plot=["pbs", "2w", "4w",],
    hatches=True,
    print_summary=True,
    run_stats=True,
    print_stats=True,
)

# ------- SAXS peak position % SPREAD bar plot dermis to wound 
fig, ax = plot_saxs_peak_spread_shift_dermis_to_wound(
    saxs_points=tables["saxs_points"],
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "bmmet", "pbsokn", "4wokn"],
    hatches=False,
)

# ------- SAXS peak position % SPREAD bar plot matched region lower wound to lower dermis, upper wound to upper dermis
fig, ax = plot_saxs_peak_spread_shift_matched_dermis_to_wound(
    saxs_points=tables["saxs_points"],
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "bmmet", "pbsokn", "4wokn"],
    hatches=False,
)

# ------- SAXS peak position % SPREAD bar plot matched type lower dermis to upper dermis, lower wound to upper wound
fig, ax = plot_saxs_peak_spread_shift_lower_to_upper(
    saxs_points=tables["saxs_points"],
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "bmmet", "pbsokn", "4wokn"],
    hatches=False,
)

# =============================================================================
#                               RAMAN
# =============================================================================

# ------- All Raman spectra for every x,y point (multipanel)
fig, axes = plot_raman_point_spectra_by_subtype(
    raman_spectra,
    subtype_order=["pbs", "2w", "4w",],
    plot_every_nth=5,
    panel_by="subtype",
    ncols=3,
)

# ------- Average Raman spectra across entire line scan
fig, ax, raman_avg = plot_raman_average_spectra(
    raman_spectra,
    subtype_order=["pbs", "2w", "4w"],
    colours=RAMAN_COLOURS,
    linestyles=RAMAN_LINESTYLES,
    err_mode="sem",
    average_unit="sample",
    error_alpha=0.25,
    linewidth=2.0,
    xlim=(1200, 1800),
    show_peak_regions=True,
    peak_regions=RAMAN_PEAK_REGIONS,
)

# ------- Average Raman spectra for nbins (multipanel)
fig, axes, raman_binned_avg = plot_raman_binned_average_spectra(
    raman_spectra,
    nbins=3,
    subtype_order=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
    colours=RAMAN_COLOURS,
    linestyles=RAMAN_LINESTYLES,
    err_mode="sem",
    average_unit="sample",
    error_alpha=0.25,
    linewidth=2.0,
    xlim=(1200, 1800),
    ncols=1,
    panel_by="subtype",
    title="Binned Raman spectra by subtype",
)

# ------- PCA
fig, axes, raman_pca_scores, raman_pca_info, raman_pca_variance, raman_pca_loadings = plot_raman_pca_within_subtype(
    raman_spectra,
    subtype_order=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
    xlim=(1200, 1800),
    standardise=True,
    ncols=3,
    loading_pcs=("PC1", "PC2", "PC3"),
    loading_top_n=15,
    peak_regions=RAMAN_PEAK_REGIONS,
    plot_scores=True,
    plot_loadings=True,
    title="Within-subtype Raman PCA by spatial position",
)

fig, axes = plot_raman_pca_scree(
    raman_pca_info,
    subtype_order=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
    max_pcs=10,
)
# -----
# raman_pca_at_glance = summarise_raman_pca_components_at_glance(
#     raman_pca_info,
#     pcs=("PC1", "PC2", "PC3"),
#     top_n=25,
#     merge_gap=8,
#     peak_regions=RAMAN_PEAK_REGIONS,
# )
# cols = [
#     "PCA_Model",
#     "PC",
#     "Variance_%",
#     "Positive_contributors",
#     "Negative_contributors",
# ]

# pd.set_option("display.max_colwidth", None)
# pd.set_option("display.max_columns", None)
# pd.set_option("display.width", 2000)

# raman_pca_at_glance_display = raman_pca_at_glance.copy()
# raman_pca_at_glance_display["Variance_%"] = raman_pca_at_glance_display["Variance_%"].round(1)

# print(
#     raman_pca_at_glance_display[
#         ["PCA_Model", "PC", "Variance_%", "Positive_contributors", "Negative_contributors"]
#     ].to_string(index=False)
# )
# ----
fig, ax, raman_pca_between, raman_pca_between_info, raman_pca_between_variance, raman_pca_between_loadings = plot_raman_pca_between_subtypes(
    raman_spectra,
    subtype_order=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
    mode="binned",
    nbins=20,
    average_unit="sample",
    standardise=True,
    colours=RAMAN_COLOURS,
    markers=NI_MARKERS,
    xlim=(1200, 1800),
    loading_pcs=("PC1", "PC2", "PC3"),
    loading_n=15,
    peak_regions=RAMAN_PEAK_REGIONS,
    show_score_trends=True,
    show_loadings=True,
    title="Between-subtype Raman PCA using binned spectra",
)

fig, ax, raman_pca_between_single, raman_pca_between_single_info = plot_raman_pca_between_subtypes(
    raman_spectra,
    subtype_order=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
    mode="single",
    average_unit="sample",
    standardise=True,
    colours=RAMAN_COLOURS,
    markers=NI_MARKERS,
    xlim=(1200, 1800),
    title="Between-subtype Raman PCA using average spectra",
)



# print(raman_wm_long.head())
# print(raman_wm_long[["PeakRegion", "Metric"]].drop_duplicates().to_string(index=False))

# ------- Weighted moment calc from average Raman spectra - peak position
fig, ax, raman_m1_summary = plot_raman_weighted_moment_bars(
    raman_wm_long,
    metric="m1",
    peak_regions=RAMAN_PEAK_REGIONS,
    subtype_order=["pbs", "2w", "4w"],
    colours=RAMAN_COLOURS,
    ylabel="WM1 shift from CT mean (cm$^{-1}$)",
    title="Raman peak-centre shift relative to control",
    error_mode="sem",
    m1_normalise="control",
    control_subtype="pbs",
)

# ------- Weighted moment calc from average Raman spectra - peak spread
fig, ax, raman_mu2_summary = plot_raman_weighted_moment_bars(
    raman_wm_long,
    metric="mu2",
    peak_regions=RAMAN_PEAK_REGIONS,
    subtype_order=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
    colours=RAMAN_COLOURS,
    ylabel="Weighted moment 2",
    title="Raman peak spread by subtype",
    error_mode="sem",
)

# ------- Weighted moment calc from average Raman spectra - peak intensity
fig, ax, raman_height_summary = plot_raman_weighted_moment_bars(
    raman_wm_long,
    metric="max_intensity",
    peak_regions=RAMAN_PEAK_REGIONS,
    subtype_order=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
    colours=RAMAN_COLOURS,
    ylabel="Peak height / max intensity",
    title="Raman peak height by subtype",
    error_mode="sem",
)

# weighted moment calc from average Raman spectra - peak area
fig, ax, raman_area_summary = plot_raman_weighted_moment_bars(
    raman_wm_long,
    metric="area_w",
    peak_regions=RAMAN_PEAK_REGIONS,
    subtype_order=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
    colours=RAMAN_COLOURS,
    ylabel="Weighted area",
    title="Raman peak area by subtype",
    error_mode="sem",
)

fig, ax, raman_selected_wm = plot_raman_selected_weighted_moments(
    raman_wm_long,
    peak_regions=["AmideI_RIGHT_1635_1700"],
    weighted_moments=["mu2", "area_w", "max_intensity"],
    subtype_order=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
    colours=RAMAN_COLOURS,
    normalise="zscore",
    control_subtype="pbs",
    panel_by_peak=False,
    title=" ",
)


# calc from average Raman spectra - amide ratio
fig, axes, amide_ratios, amide_ratio_summary = plot_raman_amide_ratios(
    raman_df,
    subtype_order=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
    colours=RAMAN_COLOURS,
    error_mode="sem",
    amideI_height_mode="max",
)

raman_binned_wm_long = calculate_raman_binned_weighted_moments(
    raman_binned_avg,
    peak_regions=RAMAN_PEAK_REGIONS,
)


# ------- Weighted moment calc from BINNED Raman spectra - peak position
fig, ax, raman_binned_m1 = plot_raman_binned_weighted_moment_bars(
    raman_binned_wm_long,
    metric="m1",
    peak_regions=RAMAN_PEAK_REGIONS,
    subtype_order=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
    bin_order=[1, 2, 3],
    colours=RAMAN_COLOURS,
    m1_normalise="control",
    control_subtype="CT",
    peak_region_bounds=dict(RAMAN_PEAK_REGIONS),
    title="Binned Raman peak-centre shift from control",
)

# ------- Weighted moment calc from BINNED Raman spectra - peak spread
fig, ax, raman_binned_mu2 = plot_raman_binned_weighted_moment_bars(
    raman_binned_wm_long,
    metric="mu2",
    peak_regions=RAMAN_PEAK_REGIONS,
    subtype_order=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
    bin_order=[1, 2, 3],
    colours=RAMAN_COLOURS,
    ylabel="Weighted moment 2",
    title="Binned Raman WM2 by peak region",
)

# ------- Weighted moment calc from BINNED Raman spectra - peak intensity
fig, ax, raman_binned_height = plot_raman_binned_weighted_moment_bars(
    raman_binned_wm_long,
    metric="max_intensity",
    peak_regions=RAMAN_PEAK_REGIONS,
    subtype_order=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
    bin_order=[1, 2, 3],
    colours=RAMAN_COLOURS,
    ylabel="Peak height / max intensity",
    title="Binned Raman peak height by peak region",
)

# ------- calc from BINNED Raman spectra - Amide ratios
fig, axes, raman_binned_amide_ratios = plot_raman_binned_amide_ratios(
    raman_binned_wm_long,
    subtype_order=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
    bin_order=[1, 2, 3],
    colours=RAMAN_COLOURS,
    amideI_height_mode="max",
)
# =============================================================================

# 1. Fibroblast density vs modulus
fig, ax, corr1 = corr_1_fibroblast_density_vs_modulus(
    cell_raw=tables["cell_raw"],
    ni_raw=tables["ni_raw"],
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
    panel_by_subtype=False,
)

# 2. Fibroblast density vs tau
fig, ax, corr2 = corr_2_fibroblast_density_vs_tau(
    cell_raw=tables["cell_raw"],
    ni_raw=tables["ni_raw"],
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
)

# 3. Amide I/III ratio vs stiffness
fig, axes, corr3 = corr_3_amide_ratio_vs_stiffness(
    raman_spectra=raman_spectra,
    ni_raw=tables["ni_raw"],
    nbins=10,
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
    panel_by_subtype=True,
)

# 4. Amide I height vs SAXS collagen
fig, ax, corr4 = corr_4_amideI_height_vs_saxs_collagen(
    raman_spectra=raman_spectra,
    saxs_points=tables["saxs_points"],
    nbins=2,
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
)

# 5. Fibroblast density vs SAXS collagen
fig, ax, corr5 = corr_5_fibroblast_density_vs_saxs_collagen(
    cell_raw=tables["cell_raw"],
    saxs_points=tables["saxs_points"],
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
)

# 6. Amide III WM1 shift vs SAXS D-period
fig, ax, corr6 = corr_6_amideIII_wm1_shift_vs_saxs_dperiod(
    raman_spectra=raman_spectra,
    saxs_points=tables["saxs_points"],
    nbins=2,
    control_subtype="CT",
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
)

# 7. SAXS wa vs tau
fig, ax, corr7 = corr_7_saxs_wa_vs_tau(
    saxs_points=tables["saxs_points"],
    ni_raw=tables["ni_raw"],
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
)

# 8. SAXS collagen vs peak-position spread
fig, ax, corr8 = corr_8_saxs_collagen_vs_peak_position_spread(
    saxs_points=tables["saxs_points"],
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
)

# 9. Raman 1330–1380 height vs stiffness
fig, axes, corr9 = corr_9_raman_1330_1380_height_vs_stiffness(
    raman_spectra=raman_spectra,
    ni_raw=tables["ni_raw"],
    nbins=10,
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
    panel_by_subtype=True,
)

# 10. SAXS collagen vs modulus
fig, ax, corr10 = corr_10_saxs_collagen_vs_modulus(
    saxs_points=tables["saxs_points"],
    ni_raw=tables["ni_raw"],
    subtypes_to_plot=["pbs", "2w", "4w", "pbsmet", "pbsokn", "4wokn"],
)
# =============================================================================
# =============================================================================

print('\n')
end_time = time.time() 
elapsed_time = end_time - start_time
# Calculate minutes and seconds
minutes, seconds = divmod(elapsed_time, 60)

# Print the result
print(f"Analysis finished, Time Elapsed: {int(minutes)} minutes {np.round(seconds, 2)} seconds")

