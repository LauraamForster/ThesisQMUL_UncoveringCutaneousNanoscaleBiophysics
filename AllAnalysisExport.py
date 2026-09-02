#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:45:15 2026

@author: lauraforster
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ramanspy
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import AllAnalysisExport_functions as FN
import os
from collections import defaultdict
import re
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
#  paths
# -----------------------------
# RAMAN
DataDir = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Raman Data/Data")
DataDirGhosts = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Raman Data/Data/Bovine")

PeakDir = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Data Analysis/PeakManifests/PeakManifest.xlsx")
ManifestDir = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Papers and Presentations/Samples/Raman Manifest Wound.xlsx")
Save_folder = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Papers and Presentations/Samples/")

# NANOINDENTATION
NI_base_path = f"/Volumes/LauraDrive/Nanoindentation/NI_Data/Output/CSV/wounding"
NI_manifest_path = f"/Users/lauraforster/Documents/Uni/3 - PhD/Nanoindentation/Data Analysis/Analysis2025/Manifests/WoundingManifest6.csv"

# SAXS
ROI_XLSX = Path("/Volumes/LauraDrive/SAXS/Presentations and notes/Manifests/ROI_points_simplified.xlsx")
MANIFEST_XLSX = Path("/Volumes/LauraDrive/SAXS/Presentations and notes/Manifests/SAXS Manifest_wound.xlsx")
CSV_ROOT = Path("/Users/lauraforster/Documents/Uni/3 - PhD/SAXS/DLS visits")

# CELL
CELL_XLSX = Path("/Volumes/LauraDrive/Other Techniques/Cell Data/Cell Density/Spatial cell counts.xlsx")
CELL_SHEET = "summary"


# =============================================================================
# SETTINGS
# =============================================================================
# RAMAN
Type = "WOUND"  

FP_full = (700, 2400)
FP_crop = (1200, 1800)
FP_band = (1590, 1720)
EXT_full = (1950, 3500)
EXT_crop = (2500, 3500)
EXT_band = None

_CANONICAL_ORDER = ["despike", "smooth", "baseline", "normalise"]

Subtypes = ["CT", "PBS", "D7", "D10", "D14", "D21"]

Colours = {
    "CT": "grey",
    "PBS": "blue",
    "D7": "tomato",
    "D10": "mediumorchid",
    "D14": "royalblue",
    "D21": "mediumseagreen",
}

Linestyles = {
    "CT": "-",
    "PBS": "-",
    "D7": "--",
    "D10": "--",
    "D14": ":",
    "D21": "-",
}

Preprocess = ["despike", "smooth", "baseline"]
treatmentorder = "before"
normalisation = "crop"
NBins = 10
   
# NANOINDENTATION
NI_set_type = "wounding"
NI_regions = ["linescan"]
NI_groups = {
    "Group 1": ["control", "d7", "d14", "d21"]
}
NI_order = NI_groups["Group 1"]

NI_nbins = 5
NI_layer = "dermis"

NI_VAR_MAP = {
    "Eff_file": "Eff modulus from file",
    "mod_file": "modulus from file",

    "CP_Hertz": "Hertz - Contact Point",
    "mod_Hertz": "Hertz - Modulus(Pa) fit",
    "Rsq_Hertz": "Hertz - Rsq",

    "CP_RoV": "RoV - Contact Point",

    "mod_OP": "OP - Modulus",
    "Rsq_OP": "OP - Rsq",

    "TimeHeld": "Holding - Time Held (s)",
    "Hold_LoadStart": "Hold - Load Start",
    "Hold_LoadEnd": "Hold - Load End",
    "RelaxFrac": "Hold - Relaxation Fraction",

    "tau_Visco": "Visco (Analytic) - tau (s)",
    "G0_Visco": "Visco (Analytic) - G0 (Pa)",
    "G1_Visco": "Visco (Analytic) - G1 (Pa)",
    "E0_Visco": "Visco (Analytic) - E0 (Pa)",
    "Einf_Visco": "Visco (Analytic) - E_inf (Pa)",
    "Rsq_Visco": "ViscoAna_r2",
}

# SAXS
TYPE_ORDER = [
    "control", "d4", "d7", "d10", "d14", "d21",
    "VH", "TS", "PBS", "2W", "4W", "PBSMET", "BM", "WT", "KO",
]

ROI_SPECS = {
    "sample": 8,
    "dermis": 8,
    "wound": 8,
}

VALID_REGIONS = set(ROI_SPECS.keys())
SAXS_SPLIT = True

if SAXS_SPLIT:
    SAXS_REGION_ORDER = ["sample", "dermis_sub", "dermis_epi", "wound_sub", "wound_epi"]
    SAXS_ANALYSIS_REGIONS = ["dermis_sub", "dermis_epi", "wound_sub", "wound_epi"]
    DERMIS_REGIONS = ["dermis_sub", "dermis_epi"]
    WOUND_REGIONS = ["wound_sub", "wound_epi"]
else:
    SAXS_REGION_ORDER = ["sample", "dermis", "wound"]
    SAXS_ANALYSIS_REGIONS = ["dermis", "wound"]
    DERMIS_REGIONS = ["dermis"]
    WOUND_REGIONS = ["wound"]

SAXS_VALID_BASE_REGIONS = ["sample", "dermis", "wound"]
SAXS_CURVEAREA_THRESH = 0.1
SAXS_INTENSITY_THRESH = 0.00
SAXS_PARAM_THRESH = 0
SAXS_TRIM_STD_DEVS = 6

PARAMS_IQ = {
    "SAXS_intensity": ["total SAXS intensity"],
    "SAXS_norm": ["total_SAXS_norm_0_1"],
    "curvearea": ["area under third order curve"],
    "curvearea_norm": ["collagen_third_norm_0_1"],
    "D_period": ["D_period"],
    "secondmoment": ["secondmoment"],
    "wa_moment": ["wa_moment"],
    "peak_width_q": ["peak_width"],
    "peak_amplitude": ["peak_amplitude"],
    "fibril_radius": ["fibril_radius"],
    "q0": ["q0"],
    "deltaq0": ["deltaq0"],
}

PARAMS_ICHI = {
    "SAXS_intensity": ["total SAXS intensity"],
    "SAXS_norm": ["total_SAXS_norm_0_1"],
    "curvearea": ["area under third order curve"],
    "curvearea_norm": ["collagen_third_norm_0_1"],

    "peak_pos1": ["peak_position"],
    "peak_pos2": ["peak_position2"],
    "peak_position_canonical": ["peak_position_canonical"],
    "peak_position_folded": ["peak_position_folded"],

    "peak_width1": ["peak_width"],
    "peak_width2": ["peak_width2"],
    "peak_amp1": ["peak_amplitude"],
    "peak_amp2": ["peak_amplitude2"],
    "peak_height1": ["peak_height"],
    "peak_height2": ["peak_height2"],
    "rsq": ["rsq_gaussian_fit"],

    "area_fit": ["area_fit"],
    "area_peaks": ["area_peaks"],
    "area_total_bs": ["area_total_bs"],

    "SM": ["SM"],
    "AP": ["AP"],

    "peak_pos1_WM": ["wm1_p1"],
    "peak_width1_WM": ["wm2_p1"],
    "peak_skew1_WM": ["wm3_p1"],
    "peak_skewness1_WM": ["wm_skew_p1"],
    "peak_area1_WM": ["wm_area_p1"],

    "peak_pos2_WM": ["wm1_p2"],
    "peak_width2_WM": ["wm2_p2"],
    "peak_skew2_WM": ["wm3_p2"],
    "peak_skewness2_WM": ["wm_skew_p2"],
    "peak_area2_WM": ["wm_area_p2"],

    "peak_pos3_WM": ["wm1_p3"],
    "peak_width3_WM": ["wm2_p3"],
    "peak_skew3_WM": ["wm3_p3"],
    "peak_skewness3_WM": ["wm_skew_p3"],
    "peak_area3_WM": ["wm_area_p3"],

    "AP_WM": ["AP_WM"],
    "area_total_WM": ["wm_area_sum"],

    "bg_c": ["bg_c"],
    "FailReason": ["FailReason"],
}

SAXS_PARAMS_TO_EXPORT = [
    "SAXS_intensity",
    "SAXS_norm",
    "curvearea",
    "curvearea_norm",
    "D_period",
    "secondmoment",
    "wa_moment",
    "peak_width_q",
    "peak_amplitude",
    "fibril_radius",
    "q0",
    "deltaq0",
    "peak_position_canonical",
    "peak_position_folded",
    "peak_width1",
    "peak_height1",
    "rsq",
    "area_peaks",
    "SM",
    "AP",
    "AP_WM",
    "area_total_WM",
]

# CELLS
CELL_CONDITION_MAP = {
    "4": "d4",
    "7": "d7",
    "10": "d10",
    "14": "d14",
    "21": "d21",
}

CELL_REGION_MAP = {
    "lower dermis": "dermis_sub",
    "upper dermis": "dermis_epi",
    "lower wound bed": "wound_sub",
    "upper wound bed": "wound_epi",
}

CELL_VALUE_NAME = "Fibroblasts_per_mm2"
# =============================================================================
# PROCESSING RAMAN
# =============================================================================
sample_manifest_df = FN.RAMAN_read_Samplemanifest(ManifestDir)
peak_manifest_df = FN.RAMAN_read_Peakmanifest(PeakDir, sheet_name="Paper")

assignments, assign_colours = FN.RAMAN_BuildAssignmentsFromManifest(peak_manifest_df)

raman_dict = FN.RAMAN_CreateDict(sample_manifest_df, Type, Subtypes)
raman_dict = FN.RAMAN_readindata(DataDir, raman_dict)
raman_dict = FN.RAMAN_SplitSpectra(raman_dict)

raman_dict = FN.RAMAN_TreatSpectra(
    data_dict=raman_dict,
    Preprocess=Preprocess,
    FP_full=FP_full,
    FP_crop=FP_crop,
    EXT_full=EXT_full,
    EXT_crop=EXT_crop,
    treatmentorder=treatmentorder,
    normalisation=normalisation,
    FP_band=FP_band,
    EXT_band=EXT_band,
)

raman_dict = FN.RAMAN_TrimRegion(raman_dict, sample_manifest_df)
raman_dict, raman_binned_avg = FN.RAMAN_BinData(raman_dict, NBins=NBins)

raman_counts, raman_dict = FN.RAMAN_count_linescan_points(raman_dict)

raman_summary_df = FN.RAMAN_WeightedMoments_ByRegion(
    data_dict=raman_dict,
    TypestoPlot=Subtypes,
    region="dermis",
    peak_regions=[
        ("AmideI_LEFT_1530_1590", (1530, 1590)),
        ("AmideI_MIDDLE_1590_1635", (1590, 1635)),
        ("AmideI_RIGHT_1635_1700", (1635, 1700)),
        ("CH2CH3_1410_1500", (1410, 1500)),
        ("AMIDEIII_LEFT_1220_1330", (1220, 1330)),
        ("AMIDEIII_RIGHT_1330_1380", (1330, 1380)),
    ],
    use_FP=True,
    use_treated=True,
)

print(raman_summary_df.head())

# =============================================================================
# PROCESSING NANOINDENTATION
# =============================================================================

ni_dict = FN.NI_ReadManifest(
    manifest_path=NI_manifest_path,
    base_path=NI_base_path,
    regions=NI_regions,
    ST=NI_set_type,
)

ni_dict = FN.NI_CutSampleLengths(ni_dict)

ni_binned_dict, ni_points_df, ni_binned_df = FN.NI_PrepareBinnedData(
    data_dict=ni_dict,
    VAR_MAP=NI_VAR_MAP,
    nbins=NI_nbins,
    layer=NI_layer,
)

ni_summary_df = FN.NI_SampleSummary(ni_points_df, NI_VAR_MAP)

print(ni_summary_df.head())
print(ni_binned_df.head())
# =============================================================================
# =============================================================================
# Nanoindentation point-retention summary
# =============================================================================

def _find_first_existing_col(df, possible_cols):
    for col in possible_cols:
        if col in df.columns:
            return col
    return None


def _valid_numeric(series, threshold=None):
    vals = pd.to_numeric(series, errors="coerce")

    if threshold is None:
        return vals.notna()

    return vals.notna() & (vals > threshold)


def print_ni_point_retention_summary(ni_df, subtype_order=None):
    """
    Print nanoindentation point-retention counts.

    Counts:
        start points        = all indentation points
        Hertz fitted        = valid Hertz modulus
        Oliver-Pharr fitted = valid OP modulus
        Visco fitted        = valid viscoelastic tau / fit
        rejected/no fit     = no valid Hertz, OP or viscoelastic fit
    """

    df = ni_df.copy()

    subtype_col = _find_first_existing_col(
        df,
        ["SubtypeClean", "subtype", "Subtype", "TYPE", "Type", "type", "Condition", "condition"],
    )

    sample_col = _find_first_existing_col(
        df,
        ["Sample", "sample", "Sample Number", "SampleNumber", "sample_number", "SampleID", "sample_id"],
    )

    # These include both your short keys and the long names from NI_VAR_MAP
    hertz_col = _find_first_existing_col(
        df,
        [
            "mod_Hertz",
            "Hertz - Modulus(Pa) fit",
            "Hertz - Modulus (Pa) fit",
            "Hertz_Modulus",
            "Hertz_modulus",
        ],
    )

    hertz_rsq_col = _find_first_existing_col(
        df,
        [
            "Rsq_Hertz",
            "Hertz - Rsq",
            "Hertz_Rsq",
            "Hertz_rsq",
        ],
    )

    op_col = _find_first_existing_col(
        df,
        [
            "mod_OP",
            "OP - Modulus",
            "OP_Modulus",
            "OP_modulus",
            "OliverPharr_modulus",
        ],
    )

    op_rsq_col = _find_first_existing_col(
        df,
        [
            "Rsq_OP",
            "OP - Rsq",
            "OP_Rsq",
            "OP_rsq",
        ],
    )

    visco_col = _find_first_existing_col(
        df,
        [
            "tau_Visco",
            "Visco (Analytic) - tau (s)",
            "Visco_tau",
            "tau",
        ],
    )

    visco_rsq_col = _find_first_existing_col(
        df,
        [
            "Rsq_Visco",
            "ViscoAna_r2",
            "Visco_r2",
            "Visco_Rsq",
        ],
    )

    required = {
        "subtype": subtype_col,
        "Hertz modulus": hertz_col,
        "Oliver-Pharr modulus": op_col,
        "Viscoelastic tau": visco_col,
    }

    missing = [name for name, col in required.items() if col is None]

    if missing:
        print("\n[Nanoindentation point-retention summary]")
        print("Could not run because these columns were not found:")
        print(missing)
        print("\nAvailable columns are:")
        print(df.columns.tolist())
        return None, None

    if sample_col is None:
        sample_col = subtype_col

    df["_Hertz_valid"] = _valid_numeric(df[hertz_col])
    df["_OP_valid"] = _valid_numeric(df[op_col])
    df["_Visco_valid"] = _valid_numeric(df[visco_col])
    
    # -----------------------------
    # Count physically high modulus values >100 kPa
    # -----------------------------
    HUNDRED_KPA = 100000
    
    df["_Hertz_over_100kPa"] = pd.to_numeric(df[hertz_col], errors="coerce") > HUNDRED_KPA
    df["_OP_over_100kPa"] = pd.to_numeric(df[op_col], errors="coerce") > HUNDRED_KPA
    
    n_hertz_valid = df["_Hertz_valid"].sum()
    n_op_valid = df["_OP_valid"].sum()
    
    n_hertz_over_100kPa = df["_Hertz_over_100kPa"].sum()
    n_op_over_100kPa = df["_OP_over_100kPa"].sum()
    
    pct_hertz_over_100kPa = 100 * n_hertz_over_100kPa / n_hertz_valid
    pct_op_over_100kPa = 100 * n_op_over_100kPa / n_op_valid

    df["_Any_fit"] = df["_Hertz_valid"] | df["_OP_valid"] | df["_Visco_valid"]
    df["_Rejected_no_fit"] = ~df["_Any_fit"]

    group_cols = [subtype_col, sample_col]

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            start_points=(hertz_col, "size"),
            Hertz_points=("_Hertz_valid", "sum"),
            OP_points=("_OP_valid", "sum"),
            Visco_points=("_Visco_valid", "sum"),
            rejected_no_fit=("_Rejected_no_fit", "sum"),
        )
        .reset_index()
    )

    summary["Hertz_lost"] = summary["start_points"] - summary["Hertz_points"]
    summary["OP_lost"] = summary["start_points"] - summary["OP_points"]
    summary["Visco_lost"] = summary["start_points"] - summary["Visco_points"]

    summary["pct_Hertz"] = 100 * summary["Hertz_points"] / summary["start_points"]
    summary["pct_OP"] = 100 * summary["OP_points"] / summary["start_points"]
    summary["pct_Visco"] = 100 * summary["Visco_points"] / summary["start_points"]
    summary["pct_rejected_no_fit"] = 100 * summary["rejected_no_fit"] / summary["start_points"]

    if subtype_order is not None:
        summary[subtype_col] = pd.Categorical(
            summary[subtype_col],
            categories=subtype_order,
            ordered=True,
        )
        summary = summary.sort_values([subtype_col, sample_col])

    print("\n" + "=" * 120)
    print("NANOINDENTATION POINT-RETENTION SUMMARY BY SAMPLE")
    print("=" * 120)

    print(
        summary[
            group_cols
            + [
                "start_points",
                "Hertz_points",
                "Hertz_lost",
                "OP_points",
                "OP_lost",
                "Visco_points",
                "Visco_lost",
                "rejected_no_fit",
                "pct_Hertz",
                "pct_OP",
                "pct_Visco",
                "pct_rejected_no_fit",
            ]
        ].to_string(index=False)
    )

    subtype_summary = (
        summary.groupby(subtype_col, observed=False)
        .agg(
            n_samples=(sample_col, "count"),
            mean_start_points=("start_points", "mean"),
            mean_pct_Hertz=("pct_Hertz", "mean"),
            mean_pct_OP=("pct_OP", "mean"),
            mean_pct_Visco=("pct_Visco", "mean"),
            mean_pct_rejected_no_fit=("pct_rejected_no_fit", "mean"),
        )
        .reset_index()
    )

    print("\n" + "=" * 120)
    print("AVERAGE % OF POINTS RETAINED PER SAMPLE, BY SUBTYPE")
    print("=" * 120)
    print(subtype_summary.to_string(index=False))

    total_points = summary["start_points"].sum()

    print("\n" + "=" * 120)
    print("TOTAL % OF POINTS RETAINED ACROSS ALL SUBTYPES")
    print("=" * 120)

    print(f"Hertz fitted:          {100 * summary['Hertz_points'].sum() / total_points:.1f}%")
    print(f"Oliver-Pharr fitted:   {100 * summary['OP_points'].sum() / total_points:.1f}%")
    print(f"Viscoelastic fitted:   {100 * summary['Visco_points'].sum() / total_points:.1f}%")
    print(f"Rejected / no fit:     {100 * summary['rejected_no_fit'].sum() / total_points:.1f}%")
    
    print("\n" + "=" * 120)
    print("MODULUS VALUES ABOVE 100 kPa")
    print("=" * 120)
    
    print(
        f"Hertz >100 kPa:        {n_hertz_over_100kPa} / {n_hertz_valid} "
        f"({pct_hertz_over_100kPa:.2f}% of valid Hertz fits)"
    )
    
    print(
        f"Oliver-Pharr >100 kPa: {n_op_over_100kPa} / {n_op_valid} "
        f"({pct_op_over_100kPa:.2f}% of valid OP fits)"
    )

    print("\nColumns used:")
    print(f"  subtype:       {subtype_col}")
    print(f"  sample:        {sample_col}")
    print(f"  Hertz modulus: {hertz_col}")
    print(f"  Hertz Rsq:     {hertz_rsq_col}")
    print(f"  OP modulus:    {op_col}")
    print(f"  OP Rsq:        {op_rsq_col}")
    print(f"  Visco tau:     {visco_col}")
    print(f"  Visco Rsq:     {visco_rsq_col}")
    

    return summary, subtype_summary


ni_retention_summary = print_ni_point_retention_summary(
    ni_points_df,
    subtype_order=NI_order,
)
# =============================================================================
# PROCESSING SAXS
# =============================================================================

saxs_data_dict, saxs_tidy_df = FN.SAXS_build_data_dict_and_tidy(
    ROI_XLSX=ROI_XLSX,
    MANIFEST_XLSX=MANIFEST_XLSX,
    CSV_ROOT=CSV_ROOT,
    ROI_SPECS=ROI_SPECS,
    valid_regions=SAXS_VALID_BASE_REGIONS,
    roi_sheet="ROIs",
    sample_region="sample",
    split=SAXS_SPLIT,
    print_dbg=False,
)

saxs_filtered_all_df, saxs_summary_all_df, saxs_per_sample_all_df, saxs_points_all_df = FN.SAXS_process_all_saxs_parameters(
    tidy=saxs_tidy_df,
    parameters=SAXS_PARAMS_TO_EXPORT,
    pooled=False,
    agg="mean",
    subtype_order=TYPE_ORDER,
    region_order=SAXS_REGION_ORDER,
    curvearea_thresh=SAXS_CURVEAREA_THRESH,
    saxs_thresh=SAXS_INTENSITY_THRESH,
    param_thresh=SAXS_PARAM_THRESH,
    trim_std_devs=SAXS_TRIM_STD_DEVS,
    PARAMS_IQ=PARAMS_IQ,
    PARAMS_ICHI=PARAMS_ICHI
)

print(saxs_tidy_df.head())
print(saxs_per_sample_all_df.head())
print(saxs_summary_all_df.head())

# =============================================================================
# SAXS point-retention summary
# =============================================================================
# =============================================================================
# SAXS point-retention summary
# =============================================================================

def _find_first_existing_col(df, possible_cols):
    for col in possible_cols:
        if col in df.columns:
            return col
    return None


def _valid_numeric(series, threshold=None):
    vals = pd.to_numeric(series, errors="coerce")

    if threshold is None:
        return vals.notna()

    return vals.notna() & (vals > threshold)


def print_saxs_point_retention_summary(
    saxs_df,
    subtype_order=None,
    curvearea_thresh=0.1,
    saxs_thresh=0.0,
):
    """
    Print SAXS point-retention counts per sample/scan.
    """

    df = saxs_df.copy()

    subtype_col = _find_first_existing_col(
        df,
        ["subtype", "SubtypeClean", "subtype_clean", "Subtype", "TYPE", "Type", "type"],
    )

    sample_col = _find_first_existing_col(
        df,
        ["Sample", "sample", "Sample Number", "SampleNumber", "sample_number"],
    )

    scan_col = _find_first_existing_col(
        df,
        ["Filenumber", "FileNumber", "file", "File", "Scan", "scan", "scan_number"],
    )

    total_saxs_col = _find_first_existing_col(
        df,
        ["total SAXS intensity", "SAXS_intensity"],
    )

    total_saxs_norm_col = _find_first_existing_col(
        df,
        ["total_SAXS_norm_0_1", "SAXS_norm"],
    )

    collagen_col = _find_first_existing_col(
        df,
        ["area under third order curve", "curvearea"],
    )

    collagen_norm_col = _find_first_existing_col(
        df,
        ["collagen_third_norm_0_1", "curvearea_norm"],
    )

    dperiod_col = _find_first_existing_col(
        df,
        ["D_period", "D period", "d_period"],
    )

    ichi_col = _find_first_existing_col(
        df,
        [
            "AP_WM",
            "peak_position_canonical",
            "peak_position_folded",
            "wm1_p1",
            "peak_position",
            "AP",
        ],
    )

    required = {
        "subtype": subtype_col,
        "sample": sample_col,
        "scan": scan_col,
        "total SAXS": total_saxs_col,
        "collagen": collagen_col,
        "D-period": dperiod_col,
        "IChi": ichi_col,
    }

    missing = [name for name, col in required.items() if col is None]

    if missing:
        print("\n[SAXS point-retention summary]")
        print("Could not run because these columns were not found:")
        print(missing)
        print("\nAvailable columns are:")
        print(df.columns.tolist())
        return None, None

    # Gates
    saxs_gate_col = total_saxs_norm_col or total_saxs_col
    collagen_gate_col = collagen_norm_col or collagen_col

    df["_has_total_saxs"] = _valid_numeric(df[saxs_gate_col], threshold=saxs_thresh)
    df["_has_collagen"] = _valid_numeric(df[collagen_gate_col], threshold=curvearea_thresh)
    df["_has_dperiod"] = _valid_numeric(df[dperiod_col])
    df["_has_ichi"] = _valid_numeric(df[ichi_col])

    group_cols = [subtype_col, sample_col, scan_col]

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            start_points=(total_saxs_col, "size"),
            total_SAXS_points=("_has_total_saxs", "sum"),
            collagen_points=("_has_collagen", "sum"),
            D_period_points=("_has_dperiod", "sum"),
            IChi_points=("_has_ichi", "sum"),
        )
        .reset_index()
    )

    summary["pct_total_SAXS"] = 100 * summary["total_SAXS_points"] / summary["start_points"]
    summary["pct_collagen"] = 100 * summary["collagen_points"] / summary["start_points"]
    summary["pct_D_period"] = 100 * summary["D_period_points"] / summary["start_points"]
    summary["pct_IChi"] = 100 * summary["IChi_points"] / summary["start_points"]

    if subtype_order is not None:
        summary[subtype_col] = pd.Categorical(
            summary[subtype_col],
            categories=subtype_order,
            ordered=True,
        )
        summary = summary.sort_values([subtype_col, sample_col, scan_col])

    print("\n" + "=" * 110)
    print("SAXS POINT-RETENTION SUMMARY BY SAMPLE / SCAN")
    print("=" * 110)

    print(
        summary[
            group_cols
            + [
                "start_points",
                "total_SAXS_points",
                "collagen_points",
                "D_period_points",
                "IChi_points",
                "pct_total_SAXS",
                "pct_collagen",
                "pct_D_period",
                "pct_IChi",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 110)
    print("AVERAGE % OF POINTS RETAINED PER SCAN, ACROSS ALL FILES / SUBTYPES")
    print("=" * 110)

    print(f"Total SAXS intensity:      {summary['pct_total_SAXS'].mean():.1f}%")
    print(f"Total collagen intensity:  {summary['pct_collagen'].mean():.1f}%")
    print(f"D-period:                  {summary['pct_D_period'].mean():.1f}%")
    print(f"IChi angular information:  {summary['pct_IChi'].mean():.1f}%")

    subtype_summary = (
        summary.groupby(subtype_col, observed=False)
        .agg(
            n_scans=(scan_col, "count"),
            mean_pct_total_SAXS=("pct_total_SAXS", "mean"),
            mean_pct_collagen=("pct_collagen", "mean"),
            mean_pct_D_period=("pct_D_period", "mean"),
            mean_pct_IChi=("pct_IChi", "mean"),
        )
        .reset_index()
    )

    print("\n" + "=" * 110)
    print("AVERAGE % OF POINTS RETAINED PER SCAN, BY SUBTYPE")
    print("=" * 110)
    print(subtype_summary.to_string(index=False))

    return summary, subtype_summary


saxs_retention_summary, saxs_retention_by_subtype = print_saxs_point_retention_summary(
    saxs_tidy_df,
    subtype_order=TYPE_ORDER,
    curvearea_thresh=SAXS_CURVEAREA_THRESH,
    saxs_thresh=SAXS_INTENSITY_THRESH,
)

# =============================================================================
# PROCESSING CELL
# =============================================================================

cell_points_df, cell_summary_df = FN.CELL_read_spatial_cell_density_data(
    cell_xlsx=CELL_XLSX,
    sheet_name=CELL_SHEET,
    condition_map=CELL_CONDITION_MAP,
    region_map=CELL_REGION_MAP,
    value_name=CELL_VALUE_NAME,
)

print(cell_points_df.head())
print(cell_summary_df)

# =============================================================================
# Multi-technique Excel export
# =============================================================================

EXPORT_ROOT = Path("/Volumes/LauraDrive/Multitech_Export")

EXPORT_SUBTYPES = ["control", "d4", "d7", "d10", "d14", "d21"]

# =============================================================================
# Final export run block
# =============================================================================

FN.export_multitech_workbooks(
    export_root=EXPORT_ROOT,
    export_subtypes=EXPORT_SUBTYPES,

    raman_summary_df=raman_summary_df,
    raman_dict=raman_dict,

    cell_points_df=cell_points_df,
    cell_summary_df=cell_summary_df,

    saxs_per_sample_all_df=saxs_per_sample_all_df,
    saxs_points_all_df=saxs_points_all_df,

    ni_summary_df=ni_summary_df,
    ni_binned_df=ni_binned_df,
    ni_points_df=ni_points_df,
)



