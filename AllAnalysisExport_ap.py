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
import AllAnalysisExport_ap_functions as FN
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
ManifestDir = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Papers and Presentations/Samples/Raman Manifest AP.xlsx")
Save_folder = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Papers and Presentations/Samples/")

# NANOINDENTATION
NI_base_path = f"/Volumes/LauraDrive/Nanoindentation/NI_Data/Output/CSV/AP1"
NI_manifest_path = f"/Users/lauraforster/Documents/Uni/3 - PhD/Nanoindentation/Data Analysis/Analysis2025/Manifests/AP1Manifest.csv"

# SAXS
ROI_XLSX = Path("/Volumes/LauraDrive/SAXS/Presentations and notes/Manifests/ROI_points_simplified.xlsx")
MANIFEST_XLSX = Path("/Volumes/LauraDrive/SAXS/Presentations and notes/Manifests/SAXS Manifest_AP.xlsx")
CSV_ROOT = Path("/Users/lauraforster/Documents/Uni/3 - PhD/SAXS/DLS visits")

# CELL
CELL_XLSX = Path("/Volumes/LauraDrive/Other Techniques/Cell Data/Cell Density/Spatial cell counts.xlsx")
CELL_SHEET = "summary"

# =============================================================================
# SETTINGS
# =============================================================================
# RAMAN
Type = "AP1" 

FP_full = (700, 2400)
FP_crop = (1200, 1800)
FP_band = (1590, 1720)
EXT_full = (1950, 3500)
EXT_crop = (2500, 3500)
EXT_band = None

_CANONICAL_ORDER = ["despike", "smooth", "baseline", "normalise"]

Subtypes = [ "ac", "cl", "ts", "vh"]

Colours = {
    "ac":"grey",
    "cl":"tomato",
    "ts":"royalblue",
    "vh":"lightcoral",
}

Linestyles = {
    "ac":"-",
    "cl":"-",
    "ts":"-",
    "vh":"-", 
}

Preprocess = ["despike", "smooth", "baseline"]
treatmentorder = "before"
normalisation = "crop"
NBins = 10
   
# NANOINDENTATION
NI_set_type = "AP1"
NI_regions = ["line scan"]

NI_groups = {
        "AP1": ["ac", "cl", "vh", "ts"]
    }

NI_order = NI_groups["AP1"]

NI_nbins = 5
NI_layer = None

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
    "ac", 
    "cl",
    "ts",
    "vh"
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

REPORT_GROUPS = {
    "AP1": ["ts", "vh", "ac", "cl"],
}
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
        ("AmideIII_1410_1500", (1410, 1500)),
        ("CH2CH3_LEFT_1220_1330", (1220, 1330)),
        ("CH2CH3_RIGHT_1330_1380", (1330, 1380)),
    ],
    use_FP=True,
    use_treated=True,
)

# print(raman_summary_df.head())
FN.REPORT_print_raman_summary(
    raman_dict=raman_dict,
    raman_summary_df=raman_summary_df,
    groups=REPORT_GROUPS,
)

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
    region_keys=NI_regions,
)

ni_summary_df = FN.NI_SampleSummary(ni_points_df, NI_VAR_MAP)

# print(ni_summary_df.head())
# print(ni_binned_df.head())
FN.REPORT_print_ni_summary(
    ni_dict=ni_dict,
    ni_points_df=ni_points_df,
    groups=REPORT_GROUPS,
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

# print(saxs_tidy_df.head())
# print(saxs_per_sample_all_df.head())
# print(saxs_summary_all_df.head())

FN.REPORT_print_saxs_summary(
    saxs_tidy_df=saxs_tidy_df,
    groups=REPORT_GROUPS,
)

# =============================================================================
# PROCESSING CELL
# =============================================================================

if Type == "WOUND":
    cell_points_df, cell_summary_df = FN.CELL_read_spatial_cell_density_data(
        cell_xlsx=CELL_XLSX,
        sheet_name=CELL_SHEET,
        condition_map=CELL_CONDITION_MAP,
        region_map=CELL_REGION_MAP,
        value_name=CELL_VALUE_NAME,
    )

    print(cell_points_df.head())
    print(cell_summary_df)

else:
    cell_points_df = pd.DataFrame()
    cell_summary_df = pd.DataFrame()
    
# print(cell_points_df.head())
# print(cell_summary_df)
FN.REPORT_print_cell_summary(
    cell_points_df=cell_points_df,
    cell_summary_df=cell_summary_df,
    groups=REPORT_GROUPS,
    type_label=Type,
)

# =============================================================================
# Multi-technique Excel export
# =============================================================================

EXPORT_ROOT = Path("/Volumes/LauraDrive/Multitech_Export_ap1")


EXPORT_SUBTYPES = [
    "ac",
    "cl",
    "ts",
    "vh"
]
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




