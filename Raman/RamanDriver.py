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
import RamanFunctions2 as RF
import time

start_time = time.time()

DataDir = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Raman Data/Data")
PeakDir = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Data Analysis/PeakManifests/ramanPeaksData-forLF4.csv")  # same folder as script by default
ManifestDir = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Papers and Presentations/Samples/Raman Manifest New2.xlsx")
Save_folder = Path("/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Papers and Presentations/Samples/")


Type = 'WOUND' #WOUND #BLEO #AP1
subtypes, colours, linestyles = RF.Types(Type)

NBins = 3

FP_full = (700, 2400)
FP_crop = (1200, 1800)
EXT_full = (1950, 3500)
EXT_crop = (2500, 3500)

Split = True        # False -> joined plot of FP and EX; True -> separate FP & EX plots
PlotXY = False      # plot XY trajectories for each subtype

Fill   = True       # shade annotated peak regions from CSV
Individual = False  # True -> Plot all the spectra on a single plot; False -> plot seperately
PCAorder = 'Trim'   # Trim -> Do PCA on only dermis; Whole -> Do PCA on entire line scan

plotall_treat = 'False'     # True -> save PDF of preprocessing; False ->  output inline
plot_mode_AverageSpectra = 'None'

plotall_PCA = 'pdf' # pdf -> save PDF of preprocessing; screen ->  output inline; None ->  no plots
plotall_poolloadings = 'pdf' # pdf -> save PDF of preprocessing; screen ->  output inline; None ->  no plots
plotall_persubtype = 'pdf'
plotall_singlePCA = 'pdf'
plot_mode_BinPCA = 'pdf'
plot_mode_BinPCA_av = 'pdf'
plot_mode_MPCA = 'pdf'
# Saveplots = False   # True -> save PNGs; False -> show inline
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------Initialisation------------------------------------------------------------

# Step 1: Load Sample Manifest
sample_manifest_df = RF.read_Samplemanifest(ManifestDir)

# Step 2: Load Peak Manifest
peak_manifest_df = RF.read_Peakmanifest(PeakDir)

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

# ------------------------------------------------------------------------------------------------------------------------------------------

# Step 6: Process spectra using those pipelines
data_dict = RF.TreatSpectra(data_dict, Save_folder, Preprocess,  FP_full, FP_crop, EXT_full, EXT_crop, colours, linestyles, plotall_treat, treatmentorder, SN='1', step=50)

# Step 7: Trim to layers in sample
data_dict = RF.TrimRegion(data_dict, sample_manifest_df, colours, linestyles, SN=None, step=10)

RF.AverageSpectra(data_dict,Save_folder,PCAorder,removeoutliers, plot_mode_AverageSpectra,region="FP")

counts_df, data_dict = RF.count_linescan_points(data_dict)
print(counts_df)

# --------------------------------------------------------------------PCA-----------------------------------------------------------------
# --------PCA per sample

# Step 8: PCA analysis on each sample number
plotall = True
data_dict = RF.PCA(meancentre, Save_folder, PCAorder, data_dict, colours, linestyles, n_components,  plotall_PCA, region="FP")

#  Step 9: 
# use step 8 PCA on each sample number and use each loadings output to look at common trends within each subtype
# RF.PCA_poolloadings(data_dict, plotall_poolloadings, Save_folder, removeoutliers, region="FP")
# data_dict = RF.PCA_poolloadings_quant(data_dict,Save_folder,removeoutliers, region="FP",norm_mode="maxabs",smooth=True,sg_window=21,sg_poly=3,n_pcs=2, peak_prom=0.12,  peak_distance=7,  peak_tol=12.0     )
# ------------------------------------------------------------------------------------------------------------------------------------------
# --------PCA averaging

# Step 10: 
# Pool PCA across repeats within a subtype. run PCA once on that pooled set to get a common PCA for the subtype
# RF.PCApersubtype(data_dict, Save_folder, removeoutliers, plotall_persubtype, PCAorder, meancentre, n_components, region="FP")

# step 11
# Do a PCA on all samples and all subtypes to get a single PC basis for the whole experiment and then plot scores grouped by subtype and repeat
# to see differences between subtypes and within subtype variability. 
# RF.singlePCA(data_dict, Save_folder, removeoutliers, PCAorder, meancentre, n_components, plotall_singlePCA, region="FP",save_xlsx=True)
# ------------------------------------------------------------------------------------------------------------------------------------------
# --------Binwise PCA to get spatial averages

# step 12
# Perform PCA on spectra binned across the depth of all samples to obtain a single PC basis and compare subtype score distributions by bin. 
# Bin each sample's line-scan into n_bins contiguous segments and run PCA per bin.
# RF.BinnedPCA_whole(data_dict,Save_folder,PCAorder,meancentre,n_components,n_bins,plot_mode_BinPCA,region="FP")

# step 13
# Perform PCA on averaged spectra per bin across all samples to reduce within-bin noise and highlight subtype-level differences. 
# Bin-wise PCA on a *subtype-averaged* line (keeps spatial structure).
# RF.BinnedPCA_average(data_dict,Save_folder,PCAorder,meancentre,n_components,n_bins,plot_mode_BinPCA_av, removeoutliers=None, region="FP", pos_grid_n=101, pos_window=0.5)
# ------------------------------------------------------------------------------------------------------------------------------------------
# --------3D PCA with spatial variation accounting

# step 14
# Perform multiway PCA (MPCA) on the 3D tensor of sample × position × spectrum to capture joint spatial–spectral variance patterns. 
# RF.PCA_MPCA(data_dict, Save_folder, PCAorder, meancentre, plot_mode_MPCA, n_components,  removeoutliers=None, pos_grid=100, pos_components=3, sample_components=None,tensorly_backend="numpy", region="FP")

# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
print('\n')
end_time = time.time() 
elapsed_time = end_time - start_time
# Calculate minutes and seconds
minutes, seconds = divmod(elapsed_time, 60)

# Print the result
print(f"Analysis of Raman finished, Time Elapsed: {int(minutes)} minutes {np.round(seconds, 2)} seconds")






