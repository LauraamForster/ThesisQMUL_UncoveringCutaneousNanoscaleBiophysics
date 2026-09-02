#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 14:29:44 2026

@author: lauraforster
"""

import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ramanspy
import RamanFunctions_NEW2 as RF

start_time = time.time()

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DataDir = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Raman Data/Data")
PeakDir = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Data Analysis/PeakManifests/ramanPeaksData-forLF4.csv")
ManifestDir = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Papers and Presentations/Samples/Raman Manifest New2.xlsx")
Save_folder = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Papers and Presentations/Samples/")

# ---------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------
Type = "Wounds"
subtypes, colours, linestyles = RF.Types(Type)

TypestoPlot = ["CT", "D7", "D10", "D14"]   # adjust as needed
region = "dermis"                          # "subcut", "dermis", "epi", "left glass", "right glass"
normalisation = "band"                     # "band" or "crop"
plotall_treat = "False"                    # "False", "screen", "pdf"
treatmentorder = "before"                  # "before" or "after"

# only 442 nm for wounds
FP_full_442 = (178, 2073)
FP_crop_442 = (700, 1800)
FP_band_442 = (1590, 1720)

PEAK_LINES = [1200, 1400, 1550, 1750]
xlim = FP_crop_442

# preprocessing
Preprocess = ["despike", "smooth", "baseline", "normalise"]

# optional smoothing for averaged display only
smoothing = 5

# ---------------------------------------------------------------------
# Load manifests and data
# ---------------------------------------------------------------------
sample_manifest_df = RF.read_Samplemanifest(ManifestDir)
peak_manifest_df = RF.read_Peakmanifest(PeakDir)

data_dict = RF.CreateDict(sample_manifest_df, subtypes)
data_dict = RF.readindata(DataDir, data_dict)

# split line scans into point spectra
data_dict = RF.SplitSpectra(data_dict, colours, linestyles, SN=None, step=10)

# ---------------------------------------------------------------------
# Preprocess all spectra
# ---------------------------------------------------------------------
# Use the wounds-style TreatSpectra signature you already built
data_dict = RF.TreatSpectra(
    data_dict,
    Save_folder,
    Preprocess,
    FP_full_442,
    FP_crop_442,
    FP_band_442,
    colours,
    linestyles,
    plotall_treat,
    treatmentorder,
    normalisation,
    SN=None,
    step=20,
)

# ---------------------------------------------------------------------
# Trim spectra to biological regions using the manifest
# ---------------------------------------------------------------------
# This creates keys like:
#   FP_Spectra_Treated_Dermis
#   FP_Spectra_Treated_Subcut
#   FP_Spectra_Treated_Epi
# etc
data_dict = RF.TrimRegion(
    data_dict,
    sample_manifest_df,
    colours,
    linestyles,
    SN=None,
    step=10,
)

# ---------------------------------------------------------------------
# Plot subtype averages within a chosen region
# ---------------------------------------------------------------------
curves = RF.PlotRegionAverageSpectra(
    data_dict,
    TypestoPlot=TypestoPlot,
    region=region,
    use_FP=True,
    use_treated=True,
    AveragebyType=True,
    show_error=True,
    error_alpha=0.25,
    err_mode="sem",     # or "std"
    axvlines=PEAK_LINES,
    xlim=xlim,
    ylim=None,
    verbose=True,
)

# ---------------------------------------------------------------------
# Weighted moment export by biological region
# ---------------------------------------------------------------------
PEAK_REGIONS = [
    ("AmideI_1550_1800",   (1550, 1750)),
    ("AmideIII_1300_1600", (1400, 1550)),
    ("CH2CH3_1150_1450",   (1200, 1400)),
]

out_xlsx = Save_folder / "Outputs" / f"WeightedMoments_{region.capitalize()}.xlsx"

df_avg, per_sample_tables = RF.ExportWeightedMoments_ByRegion(
    data_dict,
    TypestoPlot=TypestoPlot,
    region=region,
    peak_regions=PEAK_REGIONS,
    out_xlsx=out_xlsx,
    use_FP=True,
    use_treated=True,
    verbose=True,
)

elapsed = time.time() - start_time
print(f"\nAnalysis of Raman finished, Time Elapsed: {elapsed/60:.2f} minutes")





