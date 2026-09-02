#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 17:26:58 2025

@author: lauraforster
"""

import pandas as pd
import os
import NI_Analysis_Functions4 as NAF
import numpy as np

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
})
# ------------------------------------------------------------------------------------------------------------

set_type = 'wounding'
# ------------------------------------------------------------------------------------------------------------
if set_type == 'Bleo':
    manifest = 'BleomycinManifest'
    regions = ["lowerdermis", "upperdermis", "linescan"]
    groups = {
        "Group 1": ["PBS", "2W", "4W_3R", "4W_5R"],
        "Group 2": ["PBS_MET", "4W_MET", "BM_MET"],
        "Group 3": ["PBS_OKN", "4W_OKN"]
    }
    colors = {'PBS': 'lightblue', '2W': 'purple', '4W_3R': 'darkblue', '4W_5R': 'blue', 'PBS_MET': 'lightgreen', '4W_MET': 'darkgreen', 'BM_MET': 'green', 'PBS_OKN': 'red', '4W_OKN': 'orange'}
elif set_type == 'AP1':
    manifest = 'AP1Manifest'
    regions = ["linescan", "horiz_linescan"]
    groups = {
        "Group 1": ["TS", "VH"],
        "Group 2": ["CL", "AC"],
    }
    colors = {'TS': 'blue', 'VH': 'green', 'AC': 'red', 'CL': 'orange'}
    
# elif set_type == 'wounding':
#     manifest = 'WoundingManifest2'
#     regions = ["linescan", "horiz_linescan"]
#     groups = {
#         "Group 1": ["control", "d7", "d10", "VH", "d14", "d21", "AC"],
#         "Group 2": ["control", "d7",  "d14", "d21"]
#     }
#     colors = {"control": "grey","d7": "tomato","d10": "mediumorchid","VH": 'red', "d14": "royalblue","d21": "mediumseagreen", "AC": "mediumseagreen"
# }

    
elif set_type == 'wounding':
    manifest = 'WoundingManifest6'
    regions = ["linescan"]
    groups = {
        "Group 1": ["control", "d7",  "d14", "d21"]
    }
    colors = {"control": "grey","d7": "tomato", "d10": "mediumorchid","d14":"royalblue", "d21": "mediumseagreen",}
    
    linestyles = {
        "control": "-",
        "d7": "--",
        "d14": ":"
    }

# ------------------------------------------------------------------------------------------------------------
# base_path = f"/Users/lauraforster/Documents/Uni/3 - PhD/Nanoindentation/Data Analysis/Analysis2025/Output/CSV/{set_type}"
# manifest_path = f"/Users/lauraforster/Documents/Uni/3 - PhD/Nanoindentation/Data Analysis/Analysis2025/Manifests/{manifest}.csv"

base_path = f"/Volumes/LauraDrive/Nanoindentation/NI_Data/Output/CSV/{set_type}"
manifest_path = f"/Users/lauraforster/Documents/Uni/3 - PhD/Nanoindentation/Data Analysis/Analysis2025/Manifests/{manifest}.csv"
# ------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------DATA IS EXTRACTED, ANALYSED AND OUTPUT----------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
nbins = 5
layer="dermis"
order = groups["Group 1"]  # e.g. ["control","d7","d14"]

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Loop over each sample in the dictionary, screen the data, split up into skin layers and bin into nbins
data_dict= NAF.ReadManifest(manifest_path, base_path, regions, set_type)
data_dict = NAF.CutSampleLengths(data_dict)
binned_dict = NAF.PrepareBinnedData(data_dict, nbins, layer)
# # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
PlotVariable = "tau_Visco"  #Eff_file, mod_file, CP_Hertz, mod_Hertz, Rsq_Hertz, CP_RoV, mod_OP, Rsq_OP, TimeHeld, Hold_LoadStart
#                              #Hold_LoadEnd, RelaxFrac, tau_Visco, G0_Visco, G1_Visco, E0_Visco, Einf_Visco, Rsq_Visco
CorrelationVariables = ["mod_Hertz", "tau_Visco"]
pooled = True
# # after binned_dict made:
# fig, ax = NAF.PlotBarByBin(binned_dict, order, nbins, PlotVariable, colors, pooled=pooled, ylim=(0,5), ylog=False)
fig, ax = NAF.PlotBarByBin(binned_dict, order, nbins, PlotVariable, colors, pooled=pooled, ylim=None, ylog=False)

# df_sum, df_pts = NAF.export_plotbarbybin_to_excel(
#     binned_dict, order, nbins, PlotVariable,
#     pooled=pooled,
#     out_xlsx="binned_barplot_export.xlsx",
# )

# NAF.PlotScatterTrends(
#     binned_dict, order, PlotVariable,
#     colors=colors, nbins=nbins,title="Hertz modulus vs normalised position")

# NAF.PlotBestFitByBin(
#     binned_dict,order=order, nbins=nbins, PlotVariable=PlotVariable,colors=colors,
#     scale=1000,pooled=False,linestyles=linestyles,title="Best fit through binned means: ")

# NAF.PlotSpatialCurves1x3(
#     binned_dict,order=order, PlotVariable=PlotVariable,colors=colors,
#     scale=1000,title="parameter across normalised position",fixed_ylim=(0,0.01))



# fig, ax = NAF.PlotViolinByBin(binned_dict, order, nbins, PlotVariable, colors, pooled=pooled, ylim=(0,5), ylog=False)
# fig, ax = NAF.PlotViolinByBin(binned_dict, order, nbins, PlotVariable, colors, pooled=pooled, ylim=None, ylog=False)

# fig, ax = NAF.PlotScatterByNormPos(binned_dict, order, nbins, PlotVariable, colors, bestfit=False, ylim=(9e-2, 5e0), ylog=True)
# fig, ax = NAF.PlotScatterByNormPos(binned_dict, order, nbins, PlotVariable, colors, bestfit=True, ylim=None, ylog=False)

# fig, ax = NAF.PlotHistogram(binned_dict, order, PlotVariable, colors, ylog=False, normalise=True, SplitHistos=True)

# fig, ax = NAF.PlotCorrelation(binned_dict, order,CorrelationVariables[0], 
#                                 CorrelationVariables[1], colors=colors,bestfit=False, xlim=(0,5000), ylim=(0,0.5), xlog=False, ylog=False) 
                             
# fig, ax = NAF.PlotCorrelation(binned_dict, order,CorrelationVariables[0], 
                                # CorrelationVariables[1], colors=colors,bestfit=False, xlim=None, ylim=None, xlog=False, ylog=False) 


binned_dict = NAF.AddDerivedViscoVars(binned_dict)

NAF.PlotDerivedVisco1x4(
    binned_dict,
    order=order,
    PlotVariable="visco_ratio",
    colors=colors,
    title="visco_ratio across normalised position",
    fixed_ylim = (0,0.01)
)

# NAF.PlotDerivedVisco1x4(
#     binned_dict,
#     order=order,
#     PlotVariable="visco_index",
#     colors=colors,
#     title="visco_index across normalised position",
#     fixed_ylim = (0,50000)
# )


# pca_vars = ( "tau_Visco", "mod_Hertz", "norm")   # or ("tau_Visco","Einf_Visco","RelaxFrac","norm")
# PCA1log = False
# PCA2log = False
# PCA3log = False

# results, figs = NAF.PCAVisualSuite(
#     binned_dict,
#     order=order,
#     pca_vars=pca_vars,
#     colors=colors,                 # your dict: {"control":"black","d7":"blue","d14":"red",...}
#     markers={"control":"o","d7":"x","d14":"^", "d21": "*"},  # optional
#     standardise=True,
#     cmap="jet",
#     PCA1log=PCA1log, PCA2log=PCA2log, PCA3log=PCA3log
# )
# NAF.PlotPCASlopeBars(
#     results,
#     order=order,
#     colors=colors,
#     title="PC1 and PC2 slope summary"
# )

# # pca_vars = ("tau_Visco", "mod_Hertz", "norm") 
# results_by_bin, figs_by_bin = NAF.PCAByBin(
#     binned_dict, order, nbins, pca_vars,
#     colors=colors, standardise=True, pooled=True, dbg=True
# )
# NAF.PrintPCABinCloudSummary(results_by_bin, order)


# vars_to_model = pca_vars
# mm_results, mm_figs = NAF.MixedModelSuite(
#     binned_dict,
#     order=order,                 # ["control","d7","d14"]
#     vars_to_model=vars_to_model,
#     colors=colors,               # your dict: {"control":"black", "d7":"blue", "d14":"red"}
#     depth_key="norm_pos",
#     depth_scale="0-1",           # model in 0–1, nicer slopes
#     re_group="sample",
#     add_line_vc=True,
#     min_n_per_group=30,
#     make_plots=True,
    # dbg=False
# )


                            










