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
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import ramanspy
import RamanFunctions_wounds as RF
import time

start_time = time.time()

DataDir = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Raman Data/Data")
DataDirGhosts = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Raman Data/Data/Bovine")
PeakDir = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Data Analysis/PeakManifests/PeakManifest.xlsx")  # same folder as script by default
ManifestDir = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Papers and Presentations/Samples/Raman Manifest Wound.xlsx")
Save_folder = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Papers and Presentations/Samples/")


Type = 'WOUND' #WOUND #BLEO #AP1
subtypes, colours, linestyles = RF.Types(Type)


FP_full = (700, 2400)
FP_crop = (1200, 1800)
FP_band = (1590, 1720)    # your collagen band
FP_band = (1590, 1720)    # your collagen band
EXT_full = (1950, 3500)
EXT_crop = (2500, 3500)
EXT_band = None           

Split = True        # False -> joined plot of FP and EX; True -> separate FP & EX plots
PlotXY = False      # plot XY trajectories for each subtype

Fill   = True       # shade annotated peak regions from CSV
Individual = False  # True -> Plot all the spectra on a single plot; False -> plot seperately
PCAorder = 'Trim'   # Trim -> Do PCA on only dermis; Whole -> Do PCA on entire line scan

plotall_treat = 'None'     # True -> save PDF of preprocessing; False ->  output inline

# Saveplots = False   # True -> save PNGs; False -> show inline
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------Initialisation------------------------------------------------------------

# Step 1: Load Sample Manifest
sample_manifest_df = RF.read_Samplemanifest(ManifestDir)

# Step 2: Load Peak Manifest
peak_manifest_df = RF.read_Peakmanifest(PeakDir, sheet_name="Paper")

# Step 3: 
data_dict = RF.CreateDict(sample_manifest_df, subtypes) 

# Step 4:
data_dict = RF.readindata(DataDir, data_dict)

# Step 5
# SN = "285", step = 10, step = None
data_dict = RF.SplitSpectra(data_dict, colours, linestyles, SN=None, step=10)

# ---------------------------------------------------------------Preprocessing------------------------------------------------------------
Preprocess = ['despike', 'smooth','baseline', 'normalise'] # ['despike', 'smooth', 'baseline', 'normalise']
meancentre = 'with' #with -> with meancentering; without -> without mean centering
PCAorder == 'Trim'  #Trim -> PCA do just the Dermis region; Whole -> PCA do the entire line scan
n_components = 10
n_bins = 3
removeoutliers = ['17']
treatmentorder = "before"    #before -> Treat spectra then crop; after -> crop to analysis window then Treat Spectra
normalisation = "band"    # "band" or "crop"
xlsx_path = Save_folder / "WeightedMoments_Dermis.xlsx"

# ------------------------------------------------------------------------------------------------------------------------------------------

# Step 6: Process spectra using those pipelines
data_dict = RF.TreatSpectra(data_dict, Save_folder, Preprocess,  FP_full, FP_crop, EXT_full, EXT_crop, colours, linestyles, plotall_treat, treatmentorder,normalisation, SN=None, step=50, FP_band=FP_band, EXT_band=EXT_band)

# Step 7: Trim to layers in sample
data_dict = RF.TrimRegion(data_dict, sample_manifest_df, colours, linestyles, SN='None', step=10)

TypestoPlot=["CT", "D7", "D14"]
# TypestoPlot=["CT", "D7", "D10", "D14", "D21"]
region="dermis"
# PEAK_REGIONS = [
#     ("AmideI_1550_1800", (1550, 1750)),
#     ("AmideIII_1300_1600", (1400, 1550)),
#     ("CH2CH3_1150_1450", (1200, 1400)),
# ]

PEAK_REGIONS = [
    ("CH2CH3_LEFT_1220_1330", (1220, 1300)),
    ("CH2CH3_RIGHT_1330_1380", (1310, 1380)),
    
    ("AmideIII_1410_1500", (1410, 1500)),
    
    ("AmideI_LEFT_1530_1590", (1530, 1589)),
    ("AmideI_MIDDLE_1590_1635", (1590, 1634)),
    ("AmideI_RIGHT_1635_1700", (1635, 1700)),
]


peak_lines = sorted({x for _, (x1, x2) in PEAK_REGIONS for x in (x1, x2)})

curves = RF.PlotRegionAverageSpectra(data_dict,TypestoPlot, region, use_FP=True, AveragebyType=False, show_error=True, axvlines=peak_lines)


df_avg, per_sample = RF.ExportWeightedMoments_ByRegion(data_dict,TypestoPlot,region,peak_regions=PEAK_REGIONS,
                                                        out_xlsx = xlsx_path)

NBins = 3
PlotType="ByType" # ByType" or ByBin"
use_FP=True

binned_avg = RF.BinRegionAndPlot(data_dict, TypestoPlot, NBins, region, use_FP, PlotType, colours, linestyles, peak_regions=PEAK_REGIONS)

RF.AppendBinsToWeightedMoments(NBins,xlsx_path=xlsx_path)

Peak="CH2CH3_RIGHT_1330_1380"
WM="2" #'1', '2', '3', 'sigma', 'skew', 'area'

RF.PlotBinsWeightedMoment(xlsx_path=xlsx_path,
                          Peak=Peak, WM=WM,colours=colours,linestyles=linestyles)

df_t, df_m = RF.BinStats_TTests_and_MixedLM_Summary(
    xlsx_path=xlsx_path,
    TypestoPlot=TypestoPlot
)

RF.PlotOverallWeightedMomentBars(
    xlsx_path=xlsx_path,
    TypestoPlot=TypestoPlot,
    WMs=("1", "2", "height"),
    colours=colours,
    error_mode="sem"
)

RF.PlotAmideRatios(
    xlsx_path=xlsx_path,
    TypestoPlot=("CT", "D7", "D14"),
    colours=colours,
    error_mode="sem"
)

RF.PlotAnnotatedAverageSpectra(
    data_dict,
    TypestoPlot=["CT", "D7", "D14"],
    region="dermis",
    use_FP=True,
    AveragebyType=True,
    show_error=True,
    peak_regions=PEAK_REGIONS,
    peak_manifest_df=peak_manifest_df,
    component_colours={
        "Elastin": "tab:red",
        "Fibronectin": "tab:pink",
        "Collagen": "tab:blue",
        "Fibrin": "tab:brown",
        "Triolein": "tab:green",
        "Hyaluronic acid": "tab:purple",
        "Tryptophan": "tab:orange",
        "Lipids": "tab:olive",
        "Phenylalanine": "tab:cyan",
        "Tyrosine": "goldenrod",
        "GAG": "tab:gray",
    },
    colours=colours,
    linestyles=linestyles,
    xlim=(1200, 1800)
)

# RF.PlotPeakCentreAndSpread(
#     xlsx_path=xlsx_path,
#     peak_regions=PEAK_REGIONS,
#     TypestoPlot=("CT", "D7", "D14"),
#     colours=colours,
#     linestyles=linestyles,
#     x_mode="absolute",     # shows actual WM1 positions
#     normalise_x=False
# )


# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
print('\n')
end_time = time.time() 
elapsed_time = end_time - start_time
# Calculate minutes and seconds
minutes, seconds = divmod(elapsed_time, 60)

# Print the result
print(f"Analysis of Raman finished, Time Elapsed: {int(minutes)} minutes {np.round(seconds, 2)} seconds")






