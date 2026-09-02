#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:19:15 2025

@author: lauraforster
"""
import matplotlib.pyplot as plt
import NI_Model_Functions2  as NIM
import time
import os
start_time = time.time()
# ------------------------------------------------------------------------------------------------------------
group = 'JEB'
settype = 'WT_D7'
toscan = 'WT_D7-2'
foldergroup = 'WOUNDED DISEASE'
region = 'line scan' #only relevant for Bleo

# group = 'AP1'
# settype = 'CL vert'
# toscan = 'AC5'
# foldergroup='AC'
# region = 'upper dermis' #only relevant for Bleo

if group == 'Bleomycin':
    # input_folder = (f'/Users/lauraforster/Documents/Uni/3 - PhD/Nanoindentation/NI_Data/Data/{group}/{settype}/{toscan}/{region}/matrix_scan01')
    input_folder = (f'/Volumes/LauraDrive/Nanoindentation/NI_Data/Data/{group}/{settype}/{toscan}/{region}/matrix_scan01')

    output_folderCSV = f'/Volumes/LauraDrive/Nanoindentation/NI_Data/Output/CSV/{group}/{settype}/{region}'
    outputfilenameCSV = f"OutputExcel_{toscan}_linescan.csv" 
elif group == 'JEB':
        # input_folder = (f'/Users/lauraforster/Documents/Uni/3 - PhD/Nanoindentation/NI_Data/Data/{group}/{settype}/{toscan}/{region}/matrix_scan01')
        input_folder = (f'/Volumes/LauraDrive/Nanoindentation/NI_Data/Data/{group}/{foldergroup}/{settype}/{toscan}/{region}/matrix_scan01')

        output_folderCSV = f'/Volumes/LauraDrive/Nanoindentation/NI_Data/Output/CSV/{group}/{settype}/{region}'
        outputfilenameCSV = f"OutputExcel_{toscan}_linescan.csv" 
else:
    # input_folder = (f'/Users/lauraforster/Documents/Uni/3 - PhD/Nanoindentation/NI_Data/Data/{group}/{settype}/{toscan}/matrix_scan01')
    # output_folderCSV = f'/Users/lauraforster/Documents/Uni/3 - PhD/Nanoindentation/Data Analysis/Analysis2025/Output/CSV/{group}/{settype}/{foldergroup}'
    # outputfilenameCSV = f"OutputExcel_{toscan}_linescan.csv" 
    
    input_folder = (f'/Volumes/LauraDrive/Nanoindentation/NI_Data/Data/{group}/{settype}/{toscan}/matrix_scan01')
    output_folderCSV = f'/Volumes/LauraDrive/Nanoindentation/NI_Data/Output/CSV/{group}/{settype}/{foldergroup}'
    outputfilenameCSV = f"OutputExcel_{toscan}_linescan.csv" 
# ------------------------------------------------------------------------------------------------------------
csvfile, csv_writer = NIM.CreateCSV(output_folderCSV, outputfilenameCSV)
# ------------------------------------------------------------------------------------------------------------
# pdf_file_path = f'/Users/lauraforster/Documents/Uni/3 - PhD/Nanoindentation/Data Analysis/Analysis2025/Output/PDF/{group}/{settype}/{foldergroup}/OutputGraphs_{toscan}.pdf'
if group == 'Bleomycin':
    pdf_file_path = f'/Volumes/LauraDrive/Nanoindentation/NI_Data/Output/PDF/{group}/{settype}/{region}/OutputGraphs_{toscan}.pdf'
else:
    pdf_file_path = f'/Volumes/LauraDrive/Nanoindentation/NI_Data/Output/PDF/{group}/{settype}/OutputGraphs_{toscan}.pdf'

# ------------------------------------------------------------------------------------------------------------
filter_keys = None
# filter_keys = ["20_1", "34_1", "45_1", "60_1"]
# ------------------------------------------------------------------------------------------------------------
epsilon = 0.75
poissonratio = 0.5
Beta = 1
cutoff, fs, order = 0.5, 30.0, 6 #values for filtering
lowpass =  0.03
Nval = 1500
WSforthresholds = 50
WSforphases = 50
fixedmax = 0.5e-6
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------DATA IS EXTRACTED, SPLIT, FITTED AND OUTPUT----------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------
print(toscan)
# extract raw data from text files
data_dict = NIM.extract_from_txtfiles(input_folder, cutoff, fs, order, pdf_file_path, filter_keys)
# ------------------------------------------------------------------------------------------------------------
def plotdata(datax, datay, xy_key):
    plt.plot(datax, datay)
    # plt.xlim(1.5e-5,3e-5)
    # plt.ylim(-0.1e-6,0.15e-6)
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(NI_functions.format_micron))
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(NI_functions.format_nano))
    plt.xlabel('piezo (m)')
    plt.ylabel('Load (N)')
    plt.title(f'Test Plot: {xy_key}')
    plt.show()
    return
# ------------------------------------------------------------------------------------------------------------
# data is analysed
for xy_key in data_dict:
    print(xy_key)
# Filter the data
    # plotdata(data_dict[xy_key]["piezo_data"], data_dict[xy_key]["load_data"], xy_key)
    plotRawtoPDF = True
    data_dict = NIM.Filter(data_dict, xy_key, cutoff, fs, order, plotRawtoPDF)
    # plotdata(data_dict[xy_key]["piezo_data"], data_dict[xy_key]["load_data"], xy_key)
# ---------------------------------------------------  
# split raw data into various phases, including a lowPass filter and standardisation of the units
    plotSplittoPDF = True
    data_dict = NIM.SplittingHelper.Splitting(data_dict, xy_key, plotSplittoPDF, lowpass, WSforthresholds, WSforphases)
    if data_dict[xy_key]["split_failed"] == True:
        continue
    # plotdata(data_dict[xy_key]["piezo_data"], data_dict[xy_key]["load_data"], xy_key)
    # plt.plot(data_dict[xy_key]["unloading_piezo"], data_dict[xy_key]["unloading_load"]) 
    # plotdata(data_dict[xy_key]["whole_approach_piezo"], data_dict[xy_key]["whole_approach_load"], xy_key)
# ---------------------------------------------------
# Apply the Hertz model fit for contact point and Modulus
    SaveHertztoPDF = True
    data_dict = NIM.HertzContactPoint(data_dict, xy_key, poissonratio, SaveHertztoPDF)
    # ---------------------------------------------------
# Use the RoV for Contact Point 
    SaveRoVtoPDF = True
    data_dict = NIM.RoV(data_dict, xy_key, Nval, SaveRoVtoPDF)
# Apply the Hertz model fit for contact point and Modulus for a FIXED maximum
    SaveHertztoPDF = True
    # data_dict = NIM.HertzContactPoint_FixedMaximum(data_dict, xy_key, poissonratio, SaveHertztoPDF, fixedmax)
    # ---------------------------------------------------
# ---------------------------------------------------
# Apply the OliverPharr model for Modulus
    data_dict = NIM.OliverPharr(data_dict, xy_key, epsilon, Beta, poissonratio)
    SaveOPtoPDF = True
    NIM.PlotOliverpharr(data_dict, xy_key, SaveOPtoPDF)  
    
 # ---------------------------------------------------
 # Apply the Visco model for Modulus   
    SaveHoldtoPDF = True
    data_dict = NIM.Viscoelasticity(data_dict, xy_key, SaveHoldtoPDF)
    data_dict = NIM.ViscoelasticityHold(data_dict, xy_key, SaveHoldtoPDF)
    
# NEW analytic fit
    data_dict = NIM.ViscoAnalyticFit(data_dict, xy_key, tol_um=5.0, trim_s=0.0, SaveToPDF=True)   
    
    data_dict = NIM.StressStrain_Hertz_Loading_fromPiezo(data_dict, xy_key)
    NIM.PlotStressStrain_Loading(data_dict, xy_key, SaveToPDF=True)

# Output to CSV
    NIM.WritetoCSV(csv_writer, data_dict,xy_key)
    
    
# ------------------------------------------------------------------------------------------------------------    
# Print plots to PDF
pdf = data_dict[xy_key]["pdf"]
pdf.close()
csvfile.close()
# ------------------------------------------------------------------------------------------------------------
# Calculate the elapsed time
end_time = time.time()
print('\n')
elapsed_time = end_time - start_time

# Calculate minutes and seconds
minutes, seconds = divmod(elapsed_time, 60)

print(f"Elapsed Time: {elapsed_time:.2f} seconds")
os.system('afplay /Users/lauraforster/Desktop/chime-and-chomp-84419.mp3')


