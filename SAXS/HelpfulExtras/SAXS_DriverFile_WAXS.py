#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 17:19:09 2025

@author: lauraforster
"""
import time
import os
import numpy as np
from numpy import array
from pathlib import Path
from h5py import File
import matplotlib.pyplot as plt
from  pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import logging
import ReductionScript_WAXS as RS
import FittingScript_WAXS2 as FS
import VisualisingScript as VS
import Utils as Utils
import pandas as pd

start_time = time.time()

# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------ Inputs -----------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------
Experiment = 'Feb25' #or Dec23, or Feb22 or July23 or Feb25 or May24

Filelist = [810589]

# ------------------------------------------------------------------------------------------------------------------------
# Frame numbers to process
# xframe = [5,6,7]
# yframe = [0,1,2]
# process all frames
xframe = None
yframe = None
# ------------------------------------------------------------------------------------------------------------------------
# REDUCTION
ProcessIqRed   = False           #Iq Reduction
ProcessIChiRed = False           #IChi Reduction

# FITTING 
ProcessIqFit   = True           #Iq Fitting
ProcessIChiFit = False           #IChi Fitting

# VISUALSE ANALYSIS 
Visualise      = False           #Visualise parameters
HeatmapPlot    = False           #View IQ parameter heatmap
AngularOverlay = False           #View ICHi parameter heatmap
# ------------------------------------------------------------------------------------------------------------------------
# ICHI - Ring vs Entire ANALYSIS
DynamicIchi       = False        #Dynamically create the IChi ring positions in q from IQ data
IChi_EntireReduct = True        #True = Reduce IChi over one range /// False = Using Three rings
IChi_EntireFit    = True        #True Fit IChi over one range /// False = Using Three rings

# Check Shapes - Check the shape of the resultant output to ensure it matches
CheckIqRed = False              #Check Iq reduction shape
CheckBSD   = False              #Check BSDiodes output shape
CheckIqCSV = False              #Check Iq fit paramete output shape
# ------------------------------------------------------------------------------------------------------------------------
# PLOTS DISPLAY
#Select True to display inline plots for each processed Frame, False will produce a PDF 
PlotIqRed   = False             #Iq Reduction
PlotIqFit   = False             #Iq Fitting
PlotIchiRed = False             #Ichi Reduction
PlotIchiFit = False             #Ichi Fitting

chiRangesPlot = False           #Ichi ranges for rings
# ------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------Define reduction Parameters------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# IQ REDUCTION
nq = 4000 #4000                      #No. points in q
chi_range = [0,360]             #Radial range
# q_range = [0.1,1.1]             #q range
q_range = [2.5,10]             #q range

# ICHI REDUCTION
nchi = 180                      #No. points in chi
chi_range_chi = [0,360]         #Radial range
q_range_centre = [0.1,1.1]      #q range

inner_q = [0.26,0.27]           #Inner Ichi range
outer_q = [0.31,0.32]           #Outer Ichi range
cent, inn, outt = 0.01, 0.02, 0.03 #Dynamic Ranges for centre, inner and outer rings per file

# ------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------DefineFitting Parameters---------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ******************************************************  IQ   ***********************************************************
order = 3                       # Peak order to fit
order_position = 0.292          # Approximate q position for the peak
x_peak_wid = 0.02               # Approximate width of the peak
xborderLL, xborderLR = 0.02, 0.012  # LEFT of the peak max and min, for baseline subtraction
xborderRL, xborderRR = 0.012, 0.02  # RIGHT of the peak max and min, for baseline subtraction
minimum_area_threshold = 0.05 # Minimum peak area % 0.05
rsq_min = 0.2 #0.2                   # Minimum Rsq for CSV saving
A,b,d = 200, -30, 0.1           # Fit values

# ******************************************************  ICHI   ********************************************************
threshold_areaIChi = 0.000     # Minimum area threshold for IChi
# ******************************************************  HEATMAP   *****************************************************
WhatPlot = 'SAXS_norm' #'SAXS','SAXS_norm' 'curvearea', 'curvearea_norm' , 'Dperiod'. 'wMu'

threshold_saxs_intensity = 0
threshold_area = 0.000
threshold_Dperiod = 60
perc_above_baseline=20
max_intensity=0.2

if WhatPlot == 'SAXS':
    zmin, zmax = 0,2000
if WhatPlot == 'SAXS_norm' or 'curvearea_norm':
    zmin, zmax = 0.2,1
if WhatPlot == 'curvearea':
    zmin, zmax = 0,0.05
if WhatPlot == 'Dperiod':
    zmin, zmax = 64, 67
if WhatPlot == 'wMu':
    zmin, zmax = 0, 0.15
if WhatPlot == 'fibril_radius':
    zmin, zmax = 0, 400

# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------Paths -----------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
sample_beginning = "i22-"
sample_end = ".nxs"
if Experiment == 'May24':
    identifier = 'Single' 
    MASK_PATH = Path('/Volumes/Expansion/Calib Files/sm36684May24/SAXS_mask.nxs')
    CALIBRANT_PATH = Path('/Volumes/Expansion/Calib Files/sm36684May24/SAXS_calibration.nxs')
    sample_loc = "/Volumes/Seagate/dls/sm36684-1_LauraForster/"
    
if Experiment == 'Feb22':
    identifier = 'Single' 
    MASK_PATH = Path('/Volumes/Expansion/Calib Files/smxxxxFeb22/SAXS_mask.nxs')
    CALIBRANT_PATH = Path('/Volumes/Expansion/Calib Files/smxxxxFeb22/SAXS_calibration.nxs')
    sample_loc = "/Volumes/Seagate/sm29784-5/" 

if Experiment == 'July23':
    identifier = 'Single' 
    MASK_PATH = Path('/Volumes/Expansion/Calib Files/sm33398July23/SAXS_mask.nxs')
    CALIBRANT_PATH = Path('/Volumes/Expansion/Calib Files/sm33398July23/SAXS_calibration.nxs')
    sample_loc = "/Volumes/Seagate/sm33398-1/" 

if Experiment == 'Feb25':
    identifier = 'Single' 
    # MASK_PATH = Path('/Volumes/Expansion/Calib Files/sm38399Feb25/SAXS_mask.nxs')
    MASK_PATH = Path('/Volumes/LauraDrive/Calib Files/sm38399Feb25/WAXS/WAXS_mask.nxs')
    # CALIBRANT_PATH = Path('/Volumes/Expansion/Calib Files/sm38399Feb25/SAXS_calibration.nxs')
    CALIBRANT_PATH = Path('/Volumes/LauraDrive/Calib Files/sm38399Feb25/WAXS/WAXS_calibration.nxs')
    # sample_loc = "/Volumes/Seagate/dls/sm38399-1_LauraForster/" 
    sample_loc = "/Volumes/LauraDrive/SAXS processed/" 
    # sample_loc = "/Volumes/Seagate/dls/sm38399-1/" 
    

Output_directoryCSV = f'/Users/lauraforster/Documents/Uni/3 - PhD/SAXS/DLS visits/{Experiment}/WAXS_CSVs/'
Output_directorybsd = f'/Users/lauraforster/Documents/Uni/3 - PhD/SAXS/DLS visits/{Experiment}/WAXS_BSDs/'
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------ Reduction & Fitting ---------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

# *********************************************************Single Files****************************************************
# --------Reduce Iq and IChi for Single files (single grid scan)
for Filenumber in Filelist:
    Filenumber = str(Filenumber)    
 # --------Reduce Iq   
    if ProcessIqFit == True:
        FS.ProcessIQFitting(Filenumber, Output_directoryCSV, Output_directorybsd, sample_loc, xcoords=xframe, ycoords=yframe)
        print(f'I_Q Fitting outputted to {Output_directoryCSV} as .csv file')

# ------------------------------------------------------------------------------------------------------------------------
# -------------------result_rows -----------------------------------------------------------------------------------------------------
# ------------------------------------------------------------ Heatmap plotting ------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

# --------Visualise via heatmap
if Visualise == True:
    VS.heatmap(Filenumber, Output_directoryCSV, HeatmapPlot, AngularOverlay, WhatPlot, threshold_saxs_intensity, threshold_area, threshold_Dperiod, zmin, zmax, perc_above_baseline, max_intensity)

# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------ End -------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------


print('\n')
end_time = time.time() 
elapsed_time = end_time - start_time

# Calculate minutes and seconds
minutes, seconds = divmod(elapsed_time, 60)

# Print the result
print(f"Analysis of SAXS finished, Time Elapsed: {int(minutes)} minutes {np.round(seconds, 2)} seconds")


import os
# os.system('afplay /Users/lauraforster/Desktop/chime-and-chomp-84419.mp3')






