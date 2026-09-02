#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 14:29:44 2026

@author: lauraforster
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 08:56:59 2025

@author: lauraforster
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
import ramanspy
import RamanFunctions_Bovine2 as RF
import time

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 15,
})

start_time = time.time()

DataDir = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Raman Data/Data/Bovine")
PeakDir = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Data Analysis/PeakManifests/ramanPeaksData-forLF4.csv")  # same folder as script by default
ManifestDir = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Papers and Presentations/Samples/Raman Manifest_Bovine.xlsx")
Save_folder = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Papers and Presentations/Samples/")

ManifestDir_wound = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Papers and Presentations/Samples/Raman Manifest Wound.xlsx")
DataDir_wound = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Raman Data/Data")
Save_folder_wound = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Papers and Presentations/Samples/")
sample_manifest_wound_df = RF.read_Samplemanifest(ManifestDir_wound)

Type = "Bovine"
subtypes, colours, linestyles = RF.Types(Type)

Type2 = "WOUND"
TypestoPlot = ["CT", "D7"]
# TypestoPlot = []
subtypes_wound, colours_wound, linestyles_wound = RF.Types(Type2)


FP_full_785 = (1065, 2085)
FP_full_633 = (200, 2044)
FP_full_532 = (65, 2680)
FP_full_442 = (178, 2073)

FP_crop_785 = (1200,2000)
FP_crop_633 = (700, 2000)
FP_crop_532 = (1100, 2600)
FP_crop_442 = (700, 1800)


FP_band_785 = (1590, 1720)
FP_band_633 = (1590, 1720)
FP_band_532 = (1590, 1720)
FP_band_442 = (1590, 1720)

plotall_treat = False    # False -> no plots, screen -> screen plots, "pdf" -> pdf output

COMPARE_SAMPLES = [
    "fasicleWET_glass_linescan_442_10a0.5.txt",
    "fasicleDRY_glass_linescan_442_10a0.5.txt",
]

middle_fraction = 0.8   # middle 20% of points (0.20 = 20%)
use_cropped = True       # True: use Spectra_Treated (cropped+ROI norm). False: Spectra_Treated_Full 
smoothing = 0
normalisation = 'band' #band or crop
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------Initialisation------------------------------------------------------------

# Step 1: Load Sample Manifest
sample_manifest_df = RF.read_Samplemanifest(ManifestDir)
sample_manifest_wound_df = RF.read_Samplemanifest(ManifestDir_wound)

# Step 2: Load Peak Manifest
peak_manifest_df = RF.read_Peakmanifest(PeakDir)

# Step 3: 
data_dict = RF.CreateDict(sample_manifest_df, subtypes) 
wound_dict = RF.CreateDict(sample_manifest_wound_df, subtypes_wound)

# RF.DebugFindScans(data_dict, needle="sample1")

# Step 4:
data_dict = RF.readindata(DataDir, data_dict)
wound_dict = RF.readindata(DataDir_wound, wound_dict)

# Step 5
# SN = "285", step = 10, step = None
data_dict = RF.SplitSpectra(data_dict, colours, linestyles, SN=None, step=10)
wound_dict = RF.SplitSpectra(wound_dict, colours_wound, linestyles_wound, SN=None, step=10)

# # ---------------------------------------------------------------Preprocessing------------------------------------------------------------
Preprocess = ['despike', 'smooth','baseline', 'normalise'] # ['despike', 'smooth', 'baseline', 'normalise']

treatmentorder = "before"    #before -> Treat spectra then crop; after -> crop to analysis window then Treat Spectra

# # ------------------------------------------------------------------------------------------------------------------------------------------

# # Step 6: Process spectra using those pipelines
data_dict = RF.TreatSpectra(data_dict, Save_folder, Preprocess,  
                FP_full_785, FP_full_633, FP_full_532, FP_full_442,
                 FP_crop_785, FP_crop_633, FP_crop_532, FP_crop_442, 
                 FP_band_785, FP_band_633, FP_band_532, FP_band_442, 
                 colours, linestyles, plotall_treat, treatmentorder, normalisation, SN=None, step=20)

wound_dict = RF.TreatSpectra(
    wound_dict,
    Save_folder_wound,
    Preprocess,
    FP_full_442, FP_full_442, FP_full_442, FP_full_442,
    FP_crop_442, FP_crop_442, FP_crop_442, FP_crop_442,
    FP_band_442, FP_band_442, FP_band_442, FP_band_442,
    colours_wound,
    linestyles_wound,
    plotall_treat,
    treatmentorder,
    normalisation,
    SN=None,
    step=20,
)

wound_curves = RF.GetSubtypeAverageSpectraForOverlay(
    wound_dict,
    TypestoPlot=TypestoPlot,
    use_cropped=True,
    middle_fraction=None,
)



bovine_curves = RF.GetBovineMiddleAverageSpectraForOverlay(
    data_dict,
    sample_names=COMPARE_SAMPLES,
    middle_fraction=middle_fraction,
    use_cropped=True,
)

PEAK_LINES = [1220, 1300, 1310, 1380, 1410, 1500, 1530, 1589, 1590, 1635, 1700]

# xlim=FP_crop_442
xlim=(1200,1800)

RF.PlotWoundBovineOverlay(
    wound_curves,
    bovine_curves,
    axvlines=PEAK_LINES,
    xlim=xlim,
    ylim=(0,0.025),
    title="Bovine fascicle spectra overlaid with CT, D7 and D14 wound skin averages",
)

# avg = RF.PlotMiddleAverageSpectra(
#     data_dict,
#     sample_names=COMPARE_SAMPLES,
#     middle_fraction=middle_fraction,
#     smoothing=smoothing,
#     use_cropped=True,
#     axvlines=PEAK_LINES,
#     xlim=xlim,
# )







