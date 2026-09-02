#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:14:37 2025

@author: lauraforster
"""

import os
import csv
import numpy as np
from lmfit import Model, Parameters
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, savgol_filter
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
import sys
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import butter, filtfilt
import pandas as pd
from scipy.optimize import curve_fit
from lmfit import Model
from scipy.special import erfi

# --------------------------------------------------------------------------------------
# 2) Preprocessing
# --------------------------------------------------------------------------------------


class SplittingHelper:

    @staticmethod
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normalCutoff = cutoff / nyq
        b, a = butter(order, normalCutoff, btype='low', analog=False)
        return b, a

    @staticmethod
    def butter_lowpass_filter(data, cutoff, fs, order):
        b, a = SplittingHelper.butter_lowpass(cutoff, fs, order)
        y = filtfilt(b, a, data)
        return y

    @staticmethod
    def subbaseline(piezo_data, load_data, slope, y_intercept):
        piezo_data = np.array(piezo_data)
        load_baseline = slope * piezo_data + y_intercept
        return load_data - load_baseline

    @staticmethod
    def Drifting(rampup_load, loading_load, holding_load, unloading_load, rampdown_load,
                 rampup_piezo, loading_piezo, holding_piezo, unloading_piezo, rampdown_piezo,
                 whole_approach_load, whole_approach_piezo):
        if len(rampup_load) < 100:
            valuesx = int(len(loading_piezo) * 0.01)
            coefficients = np.polyfit(loading_piezo[:valuesx], loading_load[:valuesx], 1)
        else:
            valuesx = int(len(rampup_piezo) * 0.5)
            coefficients = np.polyfit(rampup_piezo[:valuesx], rampup_load[:valuesx], 1)
        slope, y_intercept = coefficients
        rampup_load = SplittingHelper.subbaseline(rampup_piezo, rampup_load, slope, y_intercept)
        loading_load = SplittingHelper.subbaseline(loading_piezo, loading_load, slope, y_intercept)
        holding_load = SplittingHelper.subbaseline(holding_piezo, holding_load, slope, y_intercept)
        unloading_load = SplittingHelper.subbaseline(unloading_piezo, unloading_load, slope, y_intercept)
        rampdown_load = SplittingHelper.subbaseline(rampdown_piezo, rampdown_load, slope, y_intercept)
        whole_approach_load = SplittingHelper.subbaseline(whole_approach_piezo, whole_approach_load, slope, y_intercept)
        return rampup_load, loading_load, holding_load, unloading_load, rampdown_load, whole_approach_load, whole_approach_piezo

    @staticmethod
    def LowPass(valuetofilt, lowpass):
        if len(valuetofilt) <= 12:
            # Handle short input vector case, return original or some other handling
            return valuetofilt
        b, a = signal.butter(3, lowpass)
        zi = signal.lfilter_zi(b, a)
        z, _ = signal.lfilter(b, a, valuetofilt, zi=zi*valuetofilt[0])
        z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
        valuetofilt = signal.filtfilt(b, a, valuetofilt)
        return valuetofilt

    @staticmethod
    def StandardUnits(data_dict, xy_key, *arrays):
        (holding_indent, holding_load, loading_indent, loading_load, unloading_indent, unloading_load,
         holding_time, loading_time, unloading_time,
         loading_piezo, unloading_piezo, holding_piezo,
         rampup_indent, rampdown_indent, rampup_load, rampdown_load,
         rampup_time, rampdown_time, rampup_piezo, rampdown_piezo,
         whole_approach_indentation, whole_approach_load, whole_approach_time, whole_approach_piezo) = arrays

        holding_indent /= 1e9
        holding_load /= 1e6
        loading_indent /= 1e9
        loading_load /= 1e6
        unloading_indent /= 1e9
        unloading_load /= 1e6
        loading_piezo /= 1e9
        unloading_piezo /= 1e9
        holding_piezo /= 1e9
        rampup_indent /= 1e9
        rampdown_indent /= 1e9
        rampup_load /= 1e6
        rampdown_load /= 1e6
        rampup_piezo /= 1e9
        rampdown_piezo /= 1e9
        whole_approach_indentation /= 1e9
        whole_approach_load /= 1e6
        whole_approach_piezo /= 1e9

        return (holding_indent, holding_load, loading_indent, loading_load, unloading_indent, unloading_load,
                holding_time, loading_time, unloading_time,
                loading_piezo, unloading_piezo, holding_piezo,
                rampup_indent, rampdown_indent, rampup_load, rampdown_load,
                rampup_time, rampdown_time, rampup_piezo, rampdown_piezo,
                whole_approach_indentation, whole_approach_load, whole_approach_time, whole_approach_piezo)

    @staticmethod
    def SVG(indentation_data):
        return savgol_filter(indentation_data, window_length=1000, polyorder=3)
    
    @staticmethod
    def updatedictwithsplit(data_dict, xy_key, *arrays):
        (holding_indent, holding_load, loading_indent, loading_load, unloading_indent, unloading_load,
         holding_time, loading_time, unloading_time,
         loading_piezo, unloading_piezo, holding_piezo,
         rampup_indent, rampdown_indent, rampup_load, rampdown_load,
         rampup_time, rampdown_time, rampup_piezo, rampdown_piezo,
         whole_approach_indentation, whole_approach_load, whole_approach_time, whole_approach_piezo) = arrays

        data_dict[xy_key].update({
            "rampup_indent": rampup_indent,
            "rampup_load": rampup_load,
            "rampup_time": rampup_time,
            "rampup_piezo": rampup_piezo,
            "loading_indent": loading_indent,
            "loading_load": loading_load,
            "loading_time": loading_time,
            "loading_piezo": loading_piezo,
            "holding_indent": holding_indent,
            "holding_load": holding_load,
            "holding_time": holding_time,
            "holding_piezo": holding_piezo,
            "unloading_indent": unloading_indent,
            "unloading_load": unloading_load,
            "unloading_time": unloading_time,
            "unloading_piezo": unloading_piezo,
            "rampdown_indent": rampdown_indent,
            "rampdown_load": rampdown_load,
            "rampdown_time": rampdown_time,
            "rampdown_piezo": rampdown_piezo,
            "whole_approach_indentation": whole_approach_indentation,
            "whole_approach_load": whole_approach_load,
            "whole_approach_time": whole_approach_time,
            "whole_approach_piezo": whole_approach_piezo,
        })
        return data_dict
        
    @staticmethod
    def detect_gradient_thresholds(data_dict, xy_key, windowsizeforthresholds):
        """
        Detects approximate gradient thresholds by computing gradients over a larger window
        and overlaying them on the graph with a Jet colormap.
        """
        piezo_data, load_data, time_data, indentation_data = data_dict[xy_key]["piezo_data"], data_dict[xy_key]["load_data"], data_dict[xy_key]["time_data"], data_dict[xy_key]["indentation_data"]
        WS = windowsizeforthresholds
        # Apply Savitzky-Golay filter for smoothing
        load_data_smooth = savgol_filter(load_data, 1000, 2)  
        
        # plt.plot(time_data, load_data)
        # plt.plot(time_data, load_data_smooth)
        # plt.show()
        # Compute gradients over sliding windows
        gradients = []
        midpoints = []
        for i in range(len(time_data) - WS):
            polyfit_coeffs = np.polyfit(time_data[i:i+WS], load_data_smooth[i:i+WS], 1)
            gradients.append(polyfit_coeffs[0])  
            midpoints.append(time_data[i + WS // 2]) 

        gradients = np.array(gradients)

        # Round gradients to 2 decimal places
        rounded_gradients = np.round(gradients, 2)

        # Identify clusters of similar gradients
        unique_vals, counts = np.unique(rounded_gradients, return_counts=True)

        # Define threshold for 'flat' gradients (hardcoded)
        flat_threshold = 0.001  

        flat_gradients = unique_vals[np.abs(unique_vals) <= flat_threshold]  
        pos_gradients = unique_vals[unique_vals > flat_threshold]  
        
        # Use mean values as representative thresholds
        flat_mean = np.mean(flat_gradients) if len(flat_gradients) > 0 else 0
        pos_mean = np.round(np.mean(pos_gradients) if len(pos_gradients) > 0 else np.max(gradients), 3)

        # Set dynamic tolerance (20% around detected means)
        pos_lower = np.round(pos_mean * 0.8, 3)
        pos_upper = np.round(pos_mean * 1.2, 3)
        # print(f"Flat Gradient Threshold: {flat_threshold}")
        # print(f"Positive Gradient: Mean = {pos_mean}, Range = {pos_lower} to {pos_upper}")
        
        WS = windowsizeforthresholds

        # Apply Savitzky-Golay filter for smoothing
        load_data_smooth = savgol_filter(load_data, 1000, 2)  
        load_data_forNeg= SplittingHelper.butter_lowpass_filter(load_data_smooth, cutoff = 0.1, fs = 100, order=4)

        # Compute gradients over sliding windows
        gradients = []
        midpoints = []
        for i in range(len(time_data) - WS):
            polyfit_coeffs = np.polyfit(time_data[i:i+WS], load_data_forNeg[i:i+WS], 1)
            gradients.append(polyfit_coeffs[0])  
            midpoints.append(time_data[i + WS // 2]) 

        gradients = np.array(gradients)

        # Round gradients to 2 decimal places
        rounded_gradients = np.round(gradients, 2)

        # Identify clusters of similar gradients
        unique_vals, counts = np.unique(rounded_gradients, return_counts=True)

        # Define threshold for 'flat' gradients (hardcoded)
        flat_threshold = 0.001  

        flat_gradients = unique_vals[np.abs(unique_vals) <= flat_threshold]  
        pos_gradients = unique_vals[unique_vals > flat_threshold]  
        neg_gradients = unique_vals[unique_vals < -flat_threshold]  
        # Instead of using all negative gradients, find the most negative 10% and use their average
        if len(neg_gradients) > 0:
            neg_sorted = np.sort(neg_gradients)  # Sort from most negative to least negative
            neg_top_10pct = neg_sorted[:max(1, len(neg_sorted) // 10)]  # Take the most negative 10%
            neg_mean = np.round(np.mean(neg_top_10pct), 3)
        else:
            neg_mean = np.round(np.min(gradients), 3)  # Fallback in case of missing values


        neg_lower = np.round(neg_mean * 0.4, 3)
        neg_upper = np.round(neg_mean * 1.6, 3)

        # print(f"Negative Gradient: Mean = {neg_mean}, Range = {neg_lower} to {neg_upper}")
        return flat_mean, pos_mean, neg_mean, pos_lower, pos_upper, neg_lower, neg_upper

    @staticmethod
    def detect_phases(data_dict, xy_key, plotSplittoPDF, flat_mean, pos_mean, neg_mean, pos_lower, pos_upper, neg_lower, neg_upper, WS2):
        """
        Walk over non-overlapping windows, detect where the gradient transitions into the loading phase,
        and split the data at that point.
        """
        FAILcounter = 0
        pdf = data_dict[xy_key]["pdf"]
        piezo_data, load_data, time_data, indentation_data = data_dict[xy_key]["piezo_data"], data_dict[xy_key]["load_data"], data_dict[xy_key]["time_data"], data_dict[xy_key]["indentation_data"]
        # Apply Savitzky-Golay filter for smoothing
        load_data_smooth = savgol_filter(load_data, 1000, 2) 
        # load_data_smooth2 = savgol_filter(load_data, 2000, 2) 
        
        # plt.plot(piezo_data, load_data)
        # plt.plot(piezo_data, load_data_smooth)
        # plt.plot(piezo_data, load_data_smooth2)
        # plt.show()

        
        plt.figure(figsize=(10, 5))
        plt.plot(time_data, load_data)
        plt.plot(time_data, load_data_smooth)
        
    ### ------------------------------------------------------------------------------------------------------------------------------------ ###
    ### ---- FIND LOADING PHASE START ---- ###
    ### ------------------------------------------------------------------------------------------------------------------------------------ ###

        # Compute gradients over non-overlapping windows
        transition_index = None
        for i in range(0, len(time_data) - WS2, WS2):  
            if i + WS2 >= len(time_data) - WS2:
                # print("Reached end of data without detecting unloading phase.")
                break
            polyfit_coeffs1 = np.polyfit(time_data[i:i+WS2], load_data_smooth[i:i+WS2], 1)
            gradient1 = np.round(polyfit_coeffs1[0], 3)
            if i + WS2 < len(time_data) - WS2:
                polyfit_coeffs2 = np.polyfit(time_data[i+WS2:i+2*WS2], load_data_smooth[i+WS2:i+2*WS2], 1)
                gradient2 = polyfit_coeffs2[0]  

                if (pos_lower <= gradient1 <= pos_upper) and (time_data[i + WS2 // 2]>1.5):
                    transition_index = i + WS2 // 2  # Center of the first detected transition window
                    # print(f"Loading Transition Detected at index {transition_index}, Time: {np.round(time_data[transition_index], 3)}s")
                    break  # Stop at the first detected transition
            
        if transition_index is None:
            target_timeFAILED = time_data[0] + 2
            transition_index = (np.abs(time_data - target_timeFAILED)).argmin() 
            FAILcounter = FAILcounter+1 
            # print(f"No Loading Index Found - splitting failed - Using index {transition_index} at Time: {np.round(time_data[transition_index], 2)}s instead") 
        # else:
            # print(f"Transition detected at index {transition_index}, Time: {np.round(time_data[transition_index], 2)}s")

        EndofApproachIndex = transition_index
            
        load_data_approach = load_data_smooth[:transition_index]
        time_data_approach = time_data[:transition_index]
        piezo_data_approach = piezo_data[:transition_index]
        
        load_data_segment2 = load_data_smooth[transition_index:]
        time_data_segment2 = time_data[transition_index:]
        piezo_data_segment2 = piezo_data[transition_index:]
            
        plt.plot(time_data_approach, load_data_approach, label="Approach Phase", color="blue")
        plt.plot(time_data_segment2, load_data_segment2, color="red")
        plt.axvline(x=time_data[transition_index], linestyle="--", color="blue")

    ### ------------------------------------------------------------------------------------------------------------------------------------ ###
    ### ---- FIND HOLDING PHASE START ---- ###
    ### ------------------------------------------------------------------------------------------------------------------------------------ ###

        WSHolding = 10
        # WSC = 20
        WS_coarse = 200
        transition_index_holding_coarse = None
        
        for i in range(0, len(time_data_segment2) - WS_coarse, WS_coarse):
            if i + WS_coarse >= len(time_data_segment2) - WS_coarse:
                # print("Reached end of data without detecting unloading phase.")
                break
            polyfit_coeffs1 = np.polyfit(time_data_segment2[i:i+WS_coarse], load_data_segment2[i:i+WS_coarse], 1)
            gradient_coarse = np.round(polyfit_coeffs1[0], 3)
            polyfit_coeffs2 = np.polyfit(time_data_segment2[i+WS_coarse:i+WS_coarse*2], load_data_segment2[i+WS_coarse:i+WS_coarse*2], 1)
            gradient_coarse2 = np.round(polyfit_coeffs2[0], 3)
            timeofloading = time_data_segment2[i + WS_coarse // 2] - time_data_segment2[0]

            # Look for the **first** transition where the gradient moves from positive to near-zero or negative
            if gradient_coarse <= 0 and gradient_coarse2 <= 0 and timeofloading>1:
                transition_index_holding_coarse = i + WS_coarse // 2
                break
        # plt.axvline(x=time_data_segment2[transition_index_holding_coarse-WS_coarse], linestyle="--", color="red")
        # plt.axvline(x=time_data_segment2[transition_index_holding_coarse+WS_coarse], linestyle="--", color="red")

        ### ---- REFINE WITH SMALLER WINDOW ---- ###
        transition_index_holding = None
        
        for i in range(transition_index_holding_coarse - WS_coarse, transition_index_holding_coarse + WS_coarse, WSHolding):
            if i < 0 or i + WSHolding > len(time_data_segment2):
                # print("Reached end of data without detecting unloading phase.")
                break
            polyfit_coeffs1 = np.polyfit(time_data_segment2[i:i+WSHolding], load_data_segment2[i:i+WSHolding], 1)
            gradient_fine = np.round(polyfit_coeffs1[0], 3)

            if gradient_fine <= 0:
                transition_index_holding = i - WSHolding // 2
                break
            
        if transition_index_holding is None:
            starttimesegment, endtimesegment = time_data_segment2[0], time_data_segment2[-1], 
            target_timeFAILED = ((endtimesegment+starttimesegment)/2)
            transition_index_holding = (np.abs(time_data_segment2 - target_timeFAILED)).argmin() 
            FAILcounter = FAILcounter+1
        #     print(f"No Holding Index Found - splitting failed - Using index {transition_index_holding} at Time: {np.round(time_data_segment2[transition_index_holding], 2)}s instead") 
        # else:
        #     print(f"Holding Transition detected at index {transition_index_holding}, Time: {np.round(time_data_segment2[transition_index_holding], 2)}s")
        
        piezo_data_loading = piezo_data_segment2[:transition_index_holding]
        if piezo_data_loading[-1] < np.max(piezo_data_loading):
            transition_index_holding = np.argmax(piezo_data_loading)
            
        EndOfLoadingPhase_Index = transition_index_holding + transition_index
        
        load_data_loading = load_data_segment2[:transition_index_holding]
        time_data_loading = time_data_segment2[:transition_index_holding]
        piezo_data_loading = piezo_data_segment2[:transition_index_holding]
            
        load_data_segment3 = load_data_segment2[transition_index_holding:]
        time_data_segment3 = time_data_segment2[transition_index_holding:]
        piezo_data_segment3 = piezo_data_segment2[transition_index_holding:]
        
        plt.axvline(x=time_data_segment2[transition_index_holding], linestyle="--", color="orange")
        plt.plot(time_data_loading, load_data_loading,  label="Loading Phase", color="orange")

    ### ------------------------------------------------------------------------------------------------------------------------------------ ###
    ### ---- FIND UNLOADING PHASE START ---- ###
    ### ------------------------------------------------------------------------------------------------------------------------------------ ###

        transition_index_unloading = None
        negative_count = 0
        
        try:
            # load_data_segment3=butter_lowpass_filter(load_data_segment3, cutoff = 0.1, fs = 100, order=4)
            target_time = time_data_segment3[0] + 2.5
            
            nearest_index = (np.abs(time_data_segment3 - target_time)).argmin()  # Index of the nearest element (find index of that point)
            time_data_segment3_spliced = time_data_segment3[nearest_index:]
            load_data_segment3_spliced = load_data_segment3[nearest_index:]
            piezo_data_segment3_spliced = piezo_data_segment3[nearest_index:]
        
            WSNEW = 10
            neg_upper = 1*neg_upper 
            neg_lower = 1*neg_lower
            bound1, bound2 = 30, 50
            testline, testline2 = bound1*WSNEW, bound2*WSNEW
            
            # plt.axvline(x=time_data_segment3_spliced[testline], linestyle="--", color="red")
            # plt.axvline(x=time_data_segment3_spliced[testline2], linestyle="--", color="red")
            
            for i in range(0, len(time_data_segment3) - WSNEW, WSNEW):
                if i + WSNEW >= len(time_data_segment3) - WSNEW:
                    # print("Reached end of data without detecting unloading phase.")
                    break
                polyfit_coeffs1 = np.polyfit(time_data_segment3_spliced[i:i+WSNEW], load_data_segment3_spliced[i:i+WSNEW], 1)
                gradient_unload = np.round(polyfit_coeffs1[0], 3)
                # Check if gradient remains strongly negative for 3 consecutive windows
                if neg_upper  < gradient_unload  and gradient_unload < neg_lower:
                    negative_count += 1
                    if negative_count >= 3:  # Require at least 3 consecutive windows
                        transition_index_unloading = i - WSNEW*5
                        break
                else:
                    negative_count = 0  # Reset count if a window isn't consistently negative
                
        except:
            target_time = time_data_segment3[0] + 0.5
            
            nearest_index = (np.abs(time_data_segment3 - target_time)).argmin()  # Index of the nearest element (find index of that point)
            time_data_segment3_spliced = time_data_segment3[nearest_index:]
            load_data_segment3_spliced = load_data_segment3[nearest_index:]
            piezo_data_segment3_spliced = piezo_data_segment3[nearest_index:]
            
            WSNEW = 10
            neg_upper = 1*neg_upper 
            neg_lower = 1*neg_lower
            bound1, bound2 = 30, 50
            testline, testline2 = bound1*WSNEW, bound2*WSNEW
            
            # plt.axvline(x=time_data_segment3_spliced[testline], linestyle="--", color="red")
            # plt.axvline(x=time_data_segment3_spliced[testline2], linestyle="--", color="red")

            for i in range(0, len(time_data_segment3_spliced) - WSNEW, WSNEW):
                if len(time_data_segment3_spliced[i:i+WSNEW]) == 0 or len(load_data_segment3_spliced[i:i+WSNEW]) ==0:
                    # print(f"Empty window at index {i}, breaking loop.")
                    break
                polyfit_coeffs1 = np.polyfit(time_data_segment3_spliced[i:i+WSNEW], load_data_segment3_spliced[i:i+WSNEW], 1)
                gradient_unload = np.round(polyfit_coeffs1[0], 3)
                # Check if gradient remains strongly negative for 3 consecutive windows
                if neg_upper  < gradient_unload  and gradient_unload < neg_lower:
                    negative_count += 1
                    if negative_count >= 3:  # Require at least 3 consecutive windows
                        transition_index_unloading = i - WSNEW*5
                        break
                else:
                    negative_count = 0  # Reset count if a window isn't consistently negative
                    # Exit if we've reached near the end and never found the transition
        
        if transition_index_unloading is None:
            starttimesegment, endtimesegment = time_data_segment3[0], time_data_segment3[-1], 
            target_timeFAILED = ((endtimesegment+starttimesegment)/2)
            transition_index_unloadingplus = (np.abs(time_data_segment3 - target_timeFAILED)).argmin() + nearest_index
            FAILcounter = FAILcounter+1
            # print(f"No Unloading Index Found - splitting failed - Using index {transition_index_unloadingplus} at Time: {np.round(time_data_segment3[transition_index_unloadingplus], 2)}s instead") 
            plt.axvline(x=time_data_segment3[transition_index_unloadingplus], linestyle="--", color="green")
        else:
            transition_index_unloadingplus = transition_index_unloading+nearest_index
            # print(f"Unloading Transition detected at index {transition_index_unloadingplus}, Time: {np.round(time_data_segment3[transition_index_unloadingplus], 2)}s")

        
        EndOfHoldingPhase_Index = transition_index_unloadingplus+ + transition_index_holding + transition_index
        
        load_data_holding = load_data_segment3[:transition_index_unloadingplus]
        time_data_holding = time_data_segment3[:transition_index_unloadingplus]
        piezo_data_holding = piezo_data_segment3[:transition_index_unloadingplus]
        
        load_data_segment4 = load_data_segment3[transition_index_unloadingplus:]
        time_data_segment4 = time_data_segment3[transition_index_unloadingplus:]
        piezo_data_segment4 = piezo_data_segment3[transition_index_unloadingplus:]

        plt.plot(time_data_holding, load_data_holding, label="Holding Phase", color="green")
        plt.axvline(x=time_data_segment3[transition_index_unloadingplus], linestyle="--", color="green")


    ### ------------------------------------------------------------------------------------------------------------------------------------ ###
    ### ---- FIND RETRACTION PHASE START ---- ###  
    ### ------------------------------------------------------------------------------------------------------------------------------------ ###

        WS3 = 10
        transition_indexretract = None

        for i in range(0, len(time_data_segment4) - WS3, WS3):  
            if i + WS3 >= len(time_data_segment4) - WS3:
                # print("Reached end of data without detecting unloading phase.")
                break
            polyfit_coeffs1 = np.polyfit(time_data_segment4[i:i+WS3], load_data_segment4[i:i+WS3], 1)
            gradient1 = np.round(polyfit_coeffs1[0], 3)
            
            if (0 <= gradient1):
                transition_indexretract = i+WS3  # Center of the first detected transition window
                break  # Stop at the first detected transition
            
        if transition_indexretract is None:
            starttimesegment, endtimesegment = time_data_segment4[0], time_data_segment4[-1] 
            target_timeFAILED = ((endtimesegment+starttimesegment)/2)
            transition_indexretract = (np.abs(time_data_segment4 - target_timeFAILED)).argmin() 
            FAILcounter = FAILcounter+1
            # print(f"No Retraction Index Found - splitting failed - Using index {transition_indexretract} at Time: {np.round(time_data_segment4[transition_indexretract], 2)}s instead") 
        # else:
            # print(f"Retraction Transition detected at index {transition_indexretract}, Time: {np.round(time_data_segment4[transition_indexretract], 2)}s")

            
        plt.axvline(x=time_data_segment4[transition_indexretract], linestyle="--", color="purple")
        
        EndOfUnloadingPhase_Index = transition_indexretract + transition_index_unloadingplus + transition_index_holding + transition_index  
        load_data_unloading = load_data_segment4[:transition_indexretract]
        time_data_unloading= time_data_segment4[:transition_indexretract]
        piezo_data_unloading = piezo_data_segment4[:transition_indexretract]
        
        # # Unloading phase data
        load_data_retraction = load_data_segment4[transition_indexretract:]
        time_data_retraction = time_data_segment4[transition_indexretract:]
        piezo_data_retraction = piezo_data_segment4[transition_indexretract:]
        
        plt.plot(time_data_unloading, load_data_unloading, label="Unloading Phase", color="purple")
        plt.plot(time_data_retraction, load_data_retraction, label="Retraction Phase", color="red")
        
        plt.xlabel("Time (s)")
        plt.ylabel("Load (N)")
        plt.title("Detected Loading Phase Transition")
        plt.legend(loc="upper left")
        # plt.xlim(0,10)
        # plt.ylim(-0.1, 0.1)
        
        if FAILcounter > 0: 
            print(f"Number of Segements failed to split: {FAILcounter}")
        
        if plotSplittoPDF == True:
            pdf.savefig()
            plt.close()
        else:
            plt.show()
            
        return EndofApproachIndex, EndOfLoadingPhase_Index, EndOfHoldingPhase_Index, EndOfUnloadingPhase_Index
    
    @staticmethod
    def score_transition_strength(index, data, window_size=10, num_windows=3, directional=False):
        """
        Scores how strong a gradient transition is at the given index.
        If directional=True, returns after_avg - before_avg.
        Otherwise, returns abs(after_avg - before_avg).
        """
        total_points = len(data)
        before_grads, after_grads = [], []

        # Gradients before
        for i in range(num_windows):
            start = index - (num_windows - i) * window_size
            end = start + window_size
            if start < 0 or end > total_points:
                continue
            grad = np.gradient(data[start:end])
            before_grads.append(np.mean(grad))

        # Gradients after
        for i in range(num_windows):
            start = index + i * window_size
            end = start + window_size
            if end > total_points:
                continue
            grad = np.gradient(data[start:end])
            after_grads.append(np.mean(grad))

        if len(before_grads) == 0 or len(after_grads) == 0:
            return 0  # Not enough data

        before_avg = np.mean(before_grads)
        after_avg = np.mean(after_grads)

        if directional:
            return after_avg - before_avg  # signed difference
        else:
            return abs(after_avg - before_avg)  # magnitude of change
    
    @staticmethod
    def is_valid_end_of_approach(index, load_data, time_data, max_fraction=0.2):
        if index is None:
            return False
        load_fraction = load_data[index] / np.max(load_data)
        time_fraction = time_data[index] / time_data[-1]
        return load_fraction < max_fraction and time_fraction < 0.5
    
    @staticmethod
    def is_valid_end_of_loading(index, load_data, time_data, tolerance=0.15):
        if index is None:
            return False
        load_fraction = load_data[index] / np.max(load_data)
        return abs(load_fraction - 1.0) < tolerance  # should be near max load
    
    @staticmethod
    def is_valid_end_of_holding(index, load_data, time_data, min_fraction=0.6):
        if index is None:
            return False
        time_fraction = time_data[index] / time_data[-1]
        return time_fraction > min_fraction and load_data[index] > 0.2 * np.max(load_data)
    
    @staticmethod
    def is_valid_end_of_unloading(index, load_data, time_data, min_fraction=0.7):
        if index is None:
            return False
        load_fraction = load_data[index] / np.max(load_data)
        time_fraction = time_data[index] / time_data[-1]
        return load_fraction < 0.3 and time_fraction > min_fraction
    
    @staticmethod
    def ValidateinIndexValues(time_data, load_data, EndofApproachIndex, EndOfLoadingPhase_Index, EndOfHoldingPhase_Index, EndOfUnloadingPhase_Index, EndofApproachIndex_filedef, EndOfLoadingPhase_Index_filedef, EndOfHoldingPhase_Index_filedef, EndOfUnloadingPhase_Index_filedef):
        load_data_smooth = savgol_filter(load_data, 1000, 2) 
        # plt.plot(time_data, load_data, color="black")
        # plt.plot(time_data, load_data_smooth, color="grey")
        # plt.axvline(x=time_data[EndofApproachIndex],color="red")
        # plt.axvline(x=time_data[EndOfLoadingPhase_Index], color="red")
        # plt.axvline(x=time_data[EndOfHoldingPhase_Index], color="red")
        # plt.axvline(x=time_data[EndOfUnloadingPhase_Index], color="red")
        
        # plt.axvline(x=time_data[EndofApproachIndex_filedef], color="orange")
        # plt.axvline(x=time_data[EndOfLoadingPhase_Index_filedef],color="orange")
        # plt.axvline(x=time_data[EndOfHoldingPhase_Index_filedef], color="orange")
        # plt.axvline(x=time_data[EndOfUnloadingPhase_Index_filedef], color="orange")
        
        # --- End of Approach: flat → increasing ---
        score_file_approach = SplittingHelper.score_transition_strength(EndofApproachIndex_filedef, load_data, directional=False)
        score_code_approach = SplittingHelper.score_transition_strength(EndofApproachIndex, load_data, directional=False)
        
        # valid_file = SplittingHelper.is_valid_end_of_approach(EndofApproachIndex_filedef, load_data, time_data)
        # valid_code = SplittingHelper.is_valid_end_of_approach(EndofApproachIndex, load_data, time_data)
          
        # Zero value pre-check
        if EndofApproachIndex_filedef == 0 and EndofApproachIndex > 0:
            EndofApproachIndex_final = EndofApproachIndex
        elif EndofApproachIndex == 0 and EndofApproachIndex_filedef > 0:
            EndofApproachIndex_final = EndofApproachIndex_filedef
        else:
            valid_file = SplittingHelper.is_valid_end_of_approach(EndofApproachIndex_filedef, load_data, time_data)
            valid_code = SplittingHelper.is_valid_end_of_approach(EndofApproachIndex, load_data, time_data)
        
            if not valid_file and not valid_code:
                if min(EndofApproachIndex_filedef, EndofApproachIndex) > 500:
                    EndofApproachIndex_final = min(EndofApproachIndex_filedef, EndofApproachIndex)
                elif EndofApproachIndex_filedef > 500:
                    EndofApproachIndex_final = EndofApproachIndex_filedef
                elif EndofApproachIndex > 500:
                    EndofApproachIndex_final = EndofApproachIndex
                else:
                    EndofApproachIndex_final = max(EndofApproachIndex_filedef, EndofApproachIndex)
            elif valid_file and not valid_code:
                EndofApproachIndex_final = EndofApproachIndex_filedef
            elif valid_code and not valid_file:
                EndofApproachIndex_final = EndofApproachIndex
            else:
                EndofApproachIndex_final = EndofApproachIndex_filedef if score_file_approach > score_code_approach else EndofApproachIndex
                       
        # --- End of Loading Phase: increasing → flat ---
        score_file_loading = SplittingHelper.score_transition_strength(EndOfLoadingPhase_Index_filedef, load_data, directional=False)
        score_code_loading = SplittingHelper.score_transition_strength(EndOfLoadingPhase_Index, load_data, directional=False)
        
        # valid_file = SplittingHelper.is_valid_end_of_loading(EndOfLoadingPhase_Index_filedef, load_data, time_data)
        # valid_code = SplittingHelper.is_valid_end_of_loading(EndOfLoadingPhase_Index, load_data, time_data)
        
        # First, zero catch
        if EndOfLoadingPhase_Index_filedef == 0 and EndOfLoadingPhase_Index > 0:
            EndOfLoadingPhase_Index_final = EndOfLoadingPhase_Index
        elif EndOfLoadingPhase_Index == 0 and EndOfLoadingPhase_Index_filedef > 0:
            EndOfLoadingPhase_Index_final = EndOfLoadingPhase_Index_filedef
        else:
            # Proceed with standard validation logic
            valid_file = SplittingHelper.is_valid_end_of_loading(EndOfLoadingPhase_Index_filedef, load_data, time_data)
            valid_code = SplittingHelper.is_valid_end_of_loading(EndOfLoadingPhase_Index, load_data, time_data)
        
            if not valid_file and not valid_code:
                EndOfLoadingPhase_Index_final = EndOfLoadingPhase_Index_filedef
            elif valid_file and not valid_code:
                EndOfLoadingPhase_Index_final = EndOfLoadingPhase_Index_filedef
            elif valid_code and not valid_file:
                EndOfLoadingPhase_Index_final = EndOfLoadingPhase_Index
            else:
                EndOfLoadingPhase_Index_final = EndOfLoadingPhase_Index_filedef if score_file_loading > score_code_loading else EndOfLoadingPhase_Index

        # --- End of Holding Phase: flat → sharply negative ---
        score_file_holding = SplittingHelper.score_transition_strength(EndOfHoldingPhase_Index_filedef, load_data, directional=True)
        score_code_holding = SplittingHelper.score_transition_strength(EndOfHoldingPhase_Index, load_data, directional=True)
        
        valid_file = SplittingHelper.is_valid_end_of_holding(EndOfHoldingPhase_Index_filedef, load_data, time_data)
        valid_code = SplittingHelper.is_valid_end_of_holding(EndOfHoldingPhase_Index, load_data, time_data)
        
        if not valid_file and not valid_code:
            EndOfHoldingPhase_Index_final = min(EndOfHoldingPhase_Index_filedef, EndOfHoldingPhase_Index)
        elif valid_file and not valid_code:
            EndOfHoldingPhase_Index_final = EndOfHoldingPhase_Index_filedef
        elif valid_code and not valid_file:
            EndOfHoldingPhase_Index_final = EndOfHoldingPhase_Index
        else:
            EndOfHoldingPhase_Index_final = EndOfHoldingPhase_Index_filedef if score_file_holding < score_code_holding else EndOfHoldingPhase_Index
        
        # --- End of Unloading Phase: sharply negative → flat ---
        score_file_unloading = SplittingHelper.score_transition_strength(EndOfUnloadingPhase_Index_filedef, load_data, directional=False)
        score_code_unloading = SplittingHelper.score_transition_strength(EndOfUnloadingPhase_Index, load_data, directional=False)
        
        valid_file = SplittingHelper.is_valid_end_of_unloading(EndOfUnloadingPhase_Index_filedef, load_data, time_data)
        valid_code = SplittingHelper.is_valid_end_of_unloading(EndOfUnloadingPhase_Index, load_data, time_data)
        
        if not valid_file and not valid_code:
            EndOfUnloadingPhase_Index_final = max(EndOfUnloadingPhase_Index_filedef, EndOfUnloadingPhase_Index)
        elif valid_file and not valid_code:
            EndOfUnloadingPhase_Index_final = EndOfUnloadingPhase_Index_filedef
        elif valid_code and not valid_file:
            EndOfUnloadingPhase_Index_final = EndOfUnloadingPhase_Index
        else:
            EndOfUnloadingPhase_Index_final = EndOfUnloadingPhase_Index_filedef if score_file_unloading > score_code_unloading else EndOfUnloadingPhase_Index

        # plt.axvline(x=time_data[EndofApproachIndex_final], linestyle="-.", color="green")
        # plt.axvline(x=time_data[EndOfLoadingPhase_Index_final], linestyle="-.", color="green")
        # plt.axvline(x=time_data[EndOfHoldingPhase_Index_final], linestyle="-.", color="green")
        # plt.axvline(x=time_data[EndOfUnloadingPhase_Index_final], linestyle="-.", color="green")
        
        # plt.show()
        return EndofApproachIndex_final, EndOfLoadingPhase_Index_final, EndOfHoldingPhase_Index_final, EndOfUnloadingPhase_Index_final
    
    @staticmethod
    def plotSplit(data_dict, xy_key, i, ij, ijk, ijkl, plotSplittoPDF):
        piezo_data,load_data, time_data, indentation_data = data_dict[xy_key]["piezo_data"], data_dict[xy_key]["load_data"], data_dict[xy_key]["time_data"], data_dict[xy_key]["indentation_data"]
        whole_approach_piezo, whole_approach_load, loading_piezo, loading_load, holding_piezo, holding_load, unloading_piezo, unloading_load, rampup_piezo, rampup_load, rampdown_piezo, rampdown_load = data_dict[xy_key]["whole_approach_piezo"], data_dict[xy_key]["whole_approach_load"], data_dict[xy_key]["loading_piezo"],data_dict[xy_key]["loading_load"], data_dict[xy_key]["holding_piezo"],data_dict[xy_key]["holding_load"],data_dict[xy_key]["unloading_piezo"],data_dict[xy_key]["unloading_load"], data_dict[xy_key]["rampup_piezo"],data_dict[xy_key]["rampup_load"],data_dict[xy_key]["rampdown_piezo"],data_dict[xy_key]["rampdown_load"]
        pdf = data_dict[xy_key]["pdf"]
        plt.plot(whole_approach_piezo, whole_approach_load)
        plt.plot(loading_piezo, loading_load)
        plt.plot(holding_piezo, holding_load)
        plt.plot(unloading_piezo, unloading_load)
        plt.plot(rampup_piezo, rampup_load)
        plt.plot(rampdown_piezo, rampdown_load)
        plt.xlabel('Piezo (m)')
        plt.ylabel('Load (N)')
        plt.title(f'{xy_key} - Data Split into approximate regions')
        
        if plotSplittoPDF == True:
            pdf.savefig()
            plt.close()
        else:
            plt.show()
        return
    
    @staticmethod
    def CantileverAdjustment(cantilever_data, piezo_data, indexval):
        # # Subtract the piezo value at the given index from all piezo values
        # corr_piezo = piezo_data - piezo_data[indexval]
        # # Subtract the cantilever value at the same index from all cantilever values
        # corr_cant = cantilever_data - cantilever_data[indexval]
        # # Final correction: piezo minus cantilever
        # corrected_piezo = corr_piezo - corr_cant
        corrected_piezo = piezo_data - cantilever_data
        return corrected_piezo

    @staticmethod
    def InitialIndetation(indentation_data):
        # Find the index of the first non-zero value in indentation_data
        nonzero_indices = np.nonzero(indentation_data)[0]
        if len(nonzero_indices) == 0:
            return None  # Or raise an exception if you prefer
        indexstart_indentation = nonzero_indices[0]
        return indexstart_indentation
    
    @staticmethod
    def Splitting(data_dict, xy_key, plotSplittoPDF, lowpass, WSforthresholds, WSforphases):
        SR = data_dict[xy_key]["splittingrequired"]
        piezo_data, load_data, time_data, indentation_data, cantilever_data = data_dict[xy_key]["piezo_data"], data_dict[xy_key]["load_data"], data_dict[xy_key]["time_data"], data_dict[xy_key]["indentation_data"], data_dict[xy_key]["cantilever_data"]
        start1, start2, start3,end3 = data_dict[xy_key]["start1"], data_dict[xy_key]["start2"], data_dict[xy_key]["start3"], data_dict[xy_key]["end3"]
    
        try:# find the thresholds of gradients and find the indexs of the data when the gradients change
            flat_mean, pos_mean, neg_mean, pos_lower, pos_upper, neg_lower, neg_upper = SplittingHelper.detect_gradient_thresholds(data_dict, xy_key, WSforthresholds)
            EndofApproachIndex, EndOfLoadingPhase_Index, EndOfHoldingPhase_Index, EndOfUnloadingPhase_Index = SplittingHelper.detect_phases(data_dict, xy_key, plotSplittoPDF, flat_mean, pos_mean, neg_mean, pos_lower, pos_upper, neg_lower, neg_upper, WSforphases)
            ManualSplitFailed = False
          # Split the data based on these indexs
        except:
            ManualSplitFailed = True
        EndofApproachIndex_filedef, EndOfLoadingPhase_Index_filedef, EndOfHoldingPhase_Index_filedef, EndOfUnloadingPhase_Index_filedef= np.argmin(np.abs(time_data - start1)), np.argmin(np.abs(time_data - start2)), np.argmin(np.abs(time_data - start3)),np.argmin(np.abs(time_data - end3))
        if SR == False and ManualSplitFailed == False:
            data_dict[xy_key]["split_failed"] = False
            # best case scenario - Manual splitting worked and there are values in the file  - choose the best
            EndofApproachIndex_final, EndOfLoadingPhase_Index_final, EndOfHoldingPhase_Index_final, EndOfUnloadingPhase_Index_final = SplittingHelper.ValidateinIndexValues(time_data, load_data, EndofApproachIndex, EndOfLoadingPhase_Index, EndOfHoldingPhase_Index, EndOfUnloadingPhase_Index, EndofApproachIndex_filedef, EndOfLoadingPhase_Index_filedef, EndOfHoldingPhase_Index_filedef, EndOfUnloadingPhase_Index_filedef)
        elif SR == False and ManualSplitFailed == True:
            data_dict[xy_key]["split_failed"] = False
            # Manual splitting failed but there are values in the file so we just use the file defined values
            EndofApproachIndex_final, EndOfLoadingPhase_Index_final, EndOfHoldingPhase_Index_final, EndOfUnloadingPhase_Index_final = EndofApproachIndex_filedef, EndOfLoadingPhase_Index_filedef, EndOfHoldingPhase_Index_filedef, EndOfUnloadingPhase_Index_filedef
        elif SR == True and ManualSplitFailed == False:
            data_dict[xy_key]["split_failed"] = False
            # Manual splitting worked but there are no values in the file so we just use the manual defined values
            EndofApproachIndex_final, EndOfLoadingPhase_Index_final, EndOfHoldingPhase_Index_final, EndOfUnloadingPhase_Index_final = EndofApproachIndex, EndOfLoadingPhase_Index, EndOfHoldingPhase_Index, EndOfUnloadingPhase_Index
        elif SR == True and ManualSplitFailed == True:
            # worst case scenario Manual splitting failed but there no values in the file so we just use the file defined values
            print('Manual Splitting Failed and no fall back values. Returning')
            data_dict[xy_key]["split_failed"] = True
            return data_dict
        
        data_dict[xy_key].update ({"EndofApproachIndex": (EndofApproachIndex_final),"EndOfLoadingPhase_Index": (EndOfLoadingPhase_Index_final),"EndOfHoldingPhase_Index": (EndOfHoldingPhase_Index_final),"EndOfUnloadingPhase_Index": (EndOfUnloadingPhase_Index_final)})
        i, ij, ijk, ijkl = EndofApproachIndex_final, EndOfLoadingPhase_Index_final, EndOfHoldingPhase_Index_final, EndOfUnloadingPhase_Index_final
        # indexstart_indentation = SplittingHelper.InitialIndetation(indentation_data)
        indexstart_indentation = EndofApproachIndex_final
        # print(indexstart_indentation, time_data[indexstart_indentation])
        # print(indexstart_indentation, cantilever_data[indexstart_indentation])
        # print(indexstart_indentation, piezo_data[indexstart_indentation])
        # print(indexstart_indentation, indentation_data[indexstart_indentation])

        piezo_data = SplittingHelper.CantileverAdjustment(cantilever_data, piezo_data, indexstart_indentation)
        
        loading_indent, holding_indent, unloading_indent, rampup_indent, rampdown_indent = np.array(indentation_data[i:ij]), np.array(indentation_data[ij:ijk]), np.array(indentation_data[ijk:ijkl]), np.array(indentation_data[:i]), np.array(indentation_data[ijkl:])
        loading_load, holding_load, unloading_load, rampup_load, rampdown_load = np.array(load_data[i:ij]), np.array(load_data[ij:ijk]), np.array(load_data[ijk:ijkl]), np.array(load_data[:i]), np.array(load_data[ijkl:])
        loading_time, holding_time, unloading_time, rampup_time, rampdown_time = np.array(time_data[i:ij]), np.array(time_data[ij:ijk]), np.array(time_data[ijk:ijkl]), np.array(time_data[:i]), np.array(time_data[ijkl:])
        loading_piezo, holding_piezo, unloading_piezo, rampup_piezo, rampdown_piezo = np.array(piezo_data[i:ij]), np.array(piezo_data[ij:ijk]), np.array(piezo_data[ijk:ijkl]), np.array(piezo_data[:i]), np.array(piezo_data[ijkl:])
        whole_approach_indentation, whole_approach_load, whole_approach_time, whole_approach_piezo = np.array(indentation_data[:ij]), np.array(load_data[:ij]), np.array(time_data[:ij]), np.array(piezo_data[:ij])
       
    # Check if drift needs implemented
        try:
            if len(rampup_load) > 50:
                if rampup_load[0] and rampup_load[50] < 0: 
                    rampup_load, loading_load, holding_load, unloading_load, rampdown_load, whole_approach_load, whole_approach_piezo = SplittingHelper.Drifting(
                        rampup_load, loading_load, holding_load, unloading_load, rampdown_load,
                        rampup_piezo, loading_piezo, holding_piezo, unloading_piezo, rampdown_piezo,
                        whole_approach_load, whole_approach_piezo)
            elif len(loading_load) > 50:
                if loading_load[0] and loading_load[50] < 0: 
                    rampup_load, loading_load, holding_load, unloading_load, rampdown_load, whole_approach_load, whole_approach_piezo = SplittingHelper.Drifting(
                        rampup_load, loading_load, holding_load, unloading_load, rampdown_load,
                        rampup_piezo, loading_piezo, holding_piezo, unloading_piezo, rampdown_piezo,
                        whole_approach_load, whole_approach_piezo)
            else:
                print(f"Skipping drift correction due to error:")
                data_dict[xy_key]["split_failed"] = True
                # Replace relevant arrays with zeros of the correct shape
                for key in ["rampup_load", "loading_load", "holding_load", "unloading_load", "rampdown_load", "whole_approach_load"]:
                    data_dict[xy_key][key] = np.zeros_like(data_dict[xy_key].get(key, np.array([0])))
                for key in ["rampup_piezo", "loading_piezo", "holding_piezo", "unloading_piezo", "rampdown_piezo", "whole_approach_piezo"]:
                    data_dict[xy_key][key] = np.zeros_like(data_dict[xy_key].get(key, np.array([0])))
        except Exception as e:
            print(f"Skipping drift correction due to error: {e}")
            data_dict[xy_key]["split_failed"] = True
            # Replace relevant arrays with zeros of the correct shape
            for key in ["rampup_load", "loading_load", "holding_load", "unloading_load", "rampdown_load", "whole_approach_load"]:
                data_dict[xy_key][key] = np.zeros_like(data_dict[xy_key].get(key, np.array([0])))
            for key in ["rampup_piezo", "loading_piezo", "holding_piezo", "unloading_piezo", "rampdown_piezo", "whole_approach_piezo"]:
                data_dict[xy_key][key] = np.zeros_like(data_dict[xy_key].get(key, np.array([0])))
            return data_dict

    # standardise the units, lowpass filter then add to dictionary
        # SI unit
        holding_indent, holding_load, loading_indent, loading_load, unloading_indent, unloading_load, holding_time, loading_time, unloading_time, loading_piezo, unloading_piezo, holding_piezo, rampup_indent, rampdown_indent, rampup_load, rampdown_load, rampup_time, rampdown_time, rampup_piezo, rampdown_piezo, whole_approach_indentation, whole_approach_load, whole_approach_time, whole_approach_piezo = SplittingHelper.StandardUnits(data_dict, xy_key, holding_indent, holding_load, loading_indent, loading_load, unloading_indent, unloading_load, holding_time, loading_time, unloading_time, loading_piezo, unloading_piezo, holding_piezo, rampup_indent, rampdown_indent, rampup_load, rampdown_load, rampup_time, rampdown_time, rampup_piezo, rampdown_piezo, whole_approach_indentation, whole_approach_load, whole_approach_time, whole_approach_piezo)
        # lowpass filter to smooth the data
        holding_load, loading_load, unloading_load, rampdown_load, rampup_load, whole_approach_load = SplittingHelper.LowPass(holding_load, lowpass), SplittingHelper.LowPass(loading_load, lowpass), SplittingHelper.LowPass(unloading_load, lowpass), SplittingHelper.LowPass(rampdown_load, lowpass), SplittingHelper.LowPass(rampup_load, lowpass), SplittingHelper.LowPass(whole_approach_load, lowpass)
        # dictionary update
        data_dict = SplittingHelper.updatedictwithsplit(data_dict, xy_key, holding_indent, holding_load, loading_indent, loading_load, unloading_indent, unloading_load, holding_time, loading_time, unloading_time, loading_piezo, unloading_piezo, holding_piezo, rampup_indent, rampdown_indent, rampup_load, rampdown_load, rampup_time, rampdown_time, rampup_piezo, rampdown_piezo, whole_approach_indentation, whole_approach_load, whole_approach_time, whole_approach_piezo)
        SplittingHelper.plotSplit(data_dict, xy_key, i, ij, ijk, ijkl, plotSplittoPDF)
        return data_dict

def Filter(data_dict, xy_key, cutoff, fs, order, plotRawtoPDF):
    piezo_data,load_data, time_data, indentation_data, cantilever_data = data_dict[xy_key]["piezo_data"], data_dict[xy_key]["load_data"], data_dict[xy_key]["time_data"], data_dict[xy_key]["indentation_data"], data_dict[xy_key]["cantilever_data"]

    
    pdf = data_dict[xy_key]["pdf"]
    # Lowpass filters
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    piezo_data_filt = lfilter(b, a, piezo_data)
    load_data_filt = lfilter(b, a, load_data)
    time_data_filt = lfilter(b, a, time_data)
    cantilever_data_filt = lfilter(b, a, cantilever_data)
    indentation_data_filt = lfilter(b, a, indentation_data)
    data_dict[xy_key]["piezo_data"], data_dict[xy_key]["load_data"], data_dict[xy_key]["time_data"], data_dict[xy_key]["indentation_data"], data_dict[xy_key]["cantilever_data"] = piezo_data_filt,load_data_filt, time_data_filt, indentation_data_filt, cantilever_data_filt
    
    plt.plot(piezo_data, load_data, label='Raw Data')
    plt.plot(piezo_data_filt, load_data_filt, label='filtered data')
    plt.title(f'{xy_key} - Raw and Filtered Data')
    plt.legend(loc='upper left')
    plt.xlabel('Piezo (nm)')
    plt.ylabel('Load (uN)')
    if plotRawtoPDF == True:
        pdf.savefig()
        plt.close()
    else:
        plt.show()
    
    return data_dict

def find_nearest_index(array, value):
    """Find index of the nearest to the given value in array."""
    array = np.array(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def format_micron(value, _):
    return f"{value * 1e6:.1f}" 

def format_nano(value, _):
    return f"{value * 1e9:.1f}" 

def extract_header(header):#Extract the x and y numbers from the header.
    lines = header.split('\n')
    scan_line = lines[2]  
    _, _, _, x_number, _, y_number, _, _ = scan_line.split('\t')
    radius_line = lines[11]
    radius = radius_line.split('\t')[1]
    start1, start2, start3, end1, end2, end3 = 0,0,0,0,0,0
    lines2 = header.split("\n")
    splittingrequired = True
    for line in lines2:
        if "E[eff] (Pa)" in line:
            value = line.split(":")[-1].strip()
            Eff = float(value.split("\t")[1])
        if "E[v=0.500]" in line:
            value = line.split(":")[-1].strip()
            E = float(value.split("\t")[1])
        if "Step absolute start times" in line:
            value = line.split()[5]
            start1 = float(value.split(',')[0])
            start2 = float(value.split(',')[1])
            start3 = float(value.split(',')[2])
            splittingrequired = False
        if "Step absolute end times" in line:
            value = line.split()[5]
            end1 = float(value.split(',')[0])
            end2 = float(value.split(',')[1])
            end3 = float(value.split(',')[2])
            splittingrequired = False
    # if splittingrequired == True:
        # print('Manual Splitting will be required for this file')
    return int(x_number), int(y_number), float(radius), float(lines[22].split('\t')[1]), Eff, E, start1, start2, start3, end1, end2, end3, splittingrequired

def extract_xy_key(file_name):
    parts = file_name.split(' ')
    xy_parts = [part for part in parts if part.startswith('X-') or part.startswith('Y-')]
    x, y = (0, 0)  # Default values if X or Y is not found
    for part in xy_parts:
        if part.startswith('X-'):
            x = int(part.split('-')[1])
        elif part.startswith('Y-'):
            y = int(part.split('-')[1])
    return x, y

def extract_from_txtfiles(input_folder, cutoff, fs, order, pdf_file_path, filter_keys):
    file_list = os.listdir(input_folder)
    sorted_file_list = sorted(file_list, key=extract_xy_key)
    data_dict = {}  
    pdf = PdfPages(pdf_file_path)

    for file_name in sorted_file_list:
        if file_name.endswith(".txt") and "position" not in file_name:
            x, y = extract_xy_key(file_name)
            # Create key for dictionary
            xy_key = f"{x}_{y}"
    
            # If filter_keys is specified, skip if not in list
            if filter_keys and xy_key not in filter_keys:
                continue
            
            file_path = os.path.join(input_folder, file_name)
            with open(file_path, "r") as file:
                lines = file.readlines()
            header = ""
            data_rows = []
            in_data_section = False
            
            for line in lines:
                if line.strip() == "Time (s)\tLoad (uN)\tIndentation (nm)\tCantilever (nm)\tPiezo (nm)\tAuxiliary":
                    in_data_section = True
                    continue
                if in_data_section:
                    data_rows.append(line.strip())
                else:
                    header += line

            x_number, y_number, radius, h_max, eff_modulus, fileYM, start1, start2, start3, end1, end2, end3, splittingrequired = extract_header(header)

            radius = radius / 1e6
            if x_number is None or y_number is None:
                continue  # Skip files where extraction failed
            
            load_data, indentation_data, time_data, piezo_data, cantilever_data = [], [], [], [],[]
            
            for row in data_rows:
                row_values = row.split("\t")
                load_data.append(float(row_values[1]))
                indentation_data.append(float(row_values[2]))
                time_data.append(float(row_values[0]))
                piezo_data.append(float(row_values[4]))
                cantilever_data.append(float(row_values[3]))
            # print(xy_key)
            data_dict[xy_key] = {
                "file_name": file_name,
                "PDF_file_name":pdf_file_path, 
                "x_number": x_number,
                "y_number": y_number,
                "eff_modulus": eff_modulus,
                "fileYM": fileYM,
                "radius": radius,
                "indentation_data": np.array(indentation_data),
                "cantilever_data": np.array(cantilever_data),
                "time_data": np.array(time_data),
                "load_data": np.array(load_data),
                "piezo_data": np.array(piezo_data),
                "pdf": pdf,
                "start1" : start1,
                "start2" : start2,
                "start3" : start3,
                "end1" : end1,
                "end2" : end2,
                "end3" : end3,
                "splittingrequired" : splittingrequired
            }
            
            file_name = os.path.basename(file_path)
    # plt.plot(time_data, piezo_data)
    # plt.plot(time_data, cantilever_data)
    # plt.show()

    return data_dict

# --------------------------------------------------------------------------------------
# 2) Hertz
# --------------------------------------------------------------------------------------


def ModHertz(x, a, d0):
    fval = np.where(x>d0, a*((x-d0)**(3/2)), 0)
    return fval
    # hyperlink pubmed paper here
    
def HertzContactPoint(data_dict, xy_key, poissonratio, SaveHertztoPDF):
    if data_dict[xy_key].get("split_failed", False) or len(data_dict[xy_key]["whole_approach_piezo"])==0:
        print(f"[{xy_key}] Skipping HertzContactPoint: split failed.")
        data_dict[xy_key].update({
            "fitresult": None,
            "Ind_mod_Hertz": np.nan,
            "ContactPoint_Hertz": np.nan,
            "Rsq_Hertz": np.nan
        })
        return data_dict
    piezo_data_tomax, load_data_tomax = data_dict[xy_key]["whole_approach_piezo"], data_dict[xy_key]["whole_approach_load"]
    indentation_data_tomax = data_dict[xy_key]["whole_approach_indentation"]
    pdf = data_dict[xy_key]["pdf"]
    radius  = data_dict[xy_key]["radius"]
    fileYM = data_dict[xy_key]["fileYM"]
    
    plt.plot(piezo_data_tomax, load_data_tomax, label='Calculated indentation value from code')
    plt.plot(indentation_data_tomax, load_data_tomax, label='Indentation column from .txt file')
    plt.xlabel('Indentation (m)')
    plt.ylabel('Load (N)')
    plt.legend(loc = 'upper left')
    plt.title(f'{xy_key} - Compared Loading Curves')
    if SaveHertztoPDF == True:
        pdf.savefig()
        plt.close()
    else:
        plt.show()

    positive_load = np.array(load_data_tomax[int((len(piezo_data_tomax)*0.1)):])
    positive_piezo = np.array(piezo_data_tomax[int((len(piezo_data_tomax)*0.1)):])
    
    modHertz = Model(ModHertz)
    params = Parameters()
    
    initial_a = fileYM if fileYM > 0 else 1e3
    params.add('a', value=initial_a, min = 0)
    
    # hardcoded = 14e-6
    try:
        guessedd0 = piezo_data_tomax[(data_dict[xy_key]["EndofApproachIndex"])-1]
    except:
        guessedd0 = piezo_data_tomax[int((len(piezo_data_tomax)*0.25))]
    params.add('d0', value = guessedd0, min = 1e-6,  max = (np.max(piezo_data_tomax))  )
    try:
        fitresult = modHertz.fit(positive_load, params=params, x=positive_piezo)
    except ValueError as e:
        print(f"Fit error: {e}")
        data_dict[xy_key].update ({
            "fitresult": None,
            "Ind_mod_Hertz": np.nan,
            "ContactPoint_Hertz": np.nan,
            "Rsq_Hertz": np.nan
        })
        return data_dict
    
    yeval = modHertz.eval(fitresult.params, x = positive_piezo)
    F= np.max(fitresult.best_fit)
    ContactPoint =np.round((fitresult.params['d0'].value),8)
    # print("Hertz", ContactPoint)
    d = (np.max(positive_piezo) - fitresult.params['d0'].value)
    
    pbracket = (1-(poissonratio**2))
    fraction =4/3
    rad = (np.sqrt(radius))
    dval = d**(3/2)
    Ind_mod = (F*pbracket)/(fraction*rad*dval)
    
    Rsq = round(fitresult.rsquared, 2)  
    Ind_mod = round(Ind_mod,2)
    
    # print('orig', fileYM, Ind_mod)

    plt.axvline(x=ContactPoint, label='Contact Point', color="black")
    plt.plot(positive_piezo, yeval, color='red', label='Initial Hertz Fit')
    plt.plot(positive_piezo, positive_load, label= 'Loading Phase Curve')

    plt.title(f'{xy_key} - Hertz Model Fit to Loading Curve')
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(format_micron))
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(format_nano))
    plt.xlabel('Indentation (m)')
    plt.ylabel('Load (N)')
    plt.legend(loc = 'upper left')
    # plt.ylim(-0.05e-6,2e-6)
    # plt.xlim(0,40e-6)
    
    if SaveHertztoPDF == True:
        pdf.savefig()
        plt.close()
    else:
        plt.show()

    data_dict[xy_key].update ({
        "fitresult": fitresult,
        "Ind_mod_Hertz": Ind_mod,
        "ContactPoint_Hertz": ContactPoint,
        "Rsq_Hertz": Rsq
        })
    
    return data_dict
 
def RoV(data_dict, xy_key, N, SaveRoVtoPDF):
    if data_dict[xy_key].get("split_failed", False) or len(data_dict[xy_key]["whole_approach_piezo"]) == 0:
        print(f"[{xy_key}] Skipping RoV: split failed.")
        data_dict[xy_key].update({
            "ContactPoint_RoV": np.nan,
        })
        return data_dict
    piezo_data_tomax, load_data_tomax = data_dict[xy_key]["whole_approach_piezo"], data_dict[xy_key]["whole_approach_load"]
    pdf = data_dict[xy_key]["pdf"]
    RoV_values = []
    # ROV for real data
    for i in range(N, len(piezo_data_tomax) - N):
        numerator = np.var(load_data_tomax[i + 1 : i + N])
        denominator = np.var(load_data_tomax[i - N : i - 1])
        RoV_values.append(numerator / denominator if denominator != 0 else np.nan)
    try:
        Argmax = np.argmax(RoV_values)
    except ValueError as e:
        print(f"Fit error: {e}")
        data_dict[xy_key].update({
            "ContactPoint_RoV": np.nan,
            })
        return data_dict
    
    ContactPointRoV = np.round(piezo_data_tomax[Argmax + N],8)
    # print("RoV", ContactPointRoV)
    cplab = np.round(ContactPointRoV, 8)
    
    plt.plot([ContactPointRoV, ContactPointRoV], [min(load_data_tomax), max(load_data_tomax)], linestyle='--', color='red', label=f'Contact Point = {cplab} µm')
    plt.plot(piezo_data_tomax, load_data_tomax)
    plt.xlabel("Piezo (µm)")
    plt.ylabel("Ratio of Variances")
    plt.title(f"{xy_key} - RoV vs. Piezo Position")
    plt.legend(loc='upper left')
    if SaveRoVtoPDF == True:
        pdf.savefig()
        plt.close()
    else:
        plt.show()
    
    data_dict[xy_key].update({
        "ContactPoint_RoV": ContactPointRoV,
        })

    return data_dict

def HertzContactPoint_FixedMaximum(data_dict, xy_key, poissonratio, SaveHertztoPDF, fixedmax):
    if data_dict[xy_key].get("split_failed", False) or len(data_dict[xy_key]["whole_approach_piezo"]) == 0:
        print(f"[{xy_key}] Skipping HertzContactPoint: split failed.")
        data_dict[xy_key].update({
            "fitresult": None,
            "Ind_mod_Hertz": np.nan,
            "ContactPoint_Hertz": np.nan,
            "Rsq_Hertz": np.nan
        })
        return data_dict
    piezo_data_tomax, load_data_tomax = data_dict[xy_key]["whole_approach_piezo"], data_dict[xy_key]["whole_approach_load"]
    ContactPoint = data_dict[xy_key]["ContactPoint_Hertz"]
    
    fixedmaxadjusted = ContactPoint + fixedmax
    pdf = data_dict[xy_key]["pdf"]
    radius  = data_dict[xy_key]["radius"]
    fileYM = data_dict[xy_key]["fileYM"]
    
    positive_load = np.array(load_data_tomax[int((len(piezo_data_tomax)*0.1)):])
    positive_piezo = np.array(piezo_data_tomax[int((len(piezo_data_tomax)*0.1)):])
    
    mask = positive_piezo <= fixedmaxadjusted
    fit_piezo = positive_piezo[mask]
    fit_load = positive_load[mask]
    
    modHertz = Model(ModHertz)
    params = Parameters()
    
    initial_a = fileYM if fileYM > 0 else 1e3
    params.add('a', value=initial_a, min = 0)
    
    params.add('d0', value = ContactPoint, vary=False )

    try:
        fitresult_fix = modHertz.fit(fit_load, params=params, x=fit_piezo)
    except:
        print("Fit error:")
        data_dict[xy_key].update ({
            "fitresult": None,
            "Ind_mod_Hertz": np.nan,
            "ContactPoint_Hertz": np.nan,
            "Rsq_Hertz": np.nan
        })
        return data_dict
    
    yeval = modHertz.eval(fitresult_fix.params, x = fit_piezo)
    F_fix= np.max(fitresult_fix.best_fit)
    ContactPoint_fix =np.round((fitresult_fix.params['d0'].value),8)
    # print("Hertz", ContactPoint)
    d_fix = (np.max(fit_piezo) - fitresult_fix.params['d0'].value)
    
    pbracket = (1-(poissonratio**2))
    fraction =4/3
    rad = (np.sqrt(radius))
    dval = d_fix**(3/2)
    Ind_mod_fix = (F_fix*pbracket)/(fraction*rad*dval)
    
    Rsq_fix = round(fitresult_fix.rsquared, 2)  
    Ind_mod_fix = round(Ind_mod_fix,2)
    
    # print('fix', Ind_mod_fix)

    
    plt.axvline(x=ContactPoint_fix, label='Contact Point', color="black")
    plt.plot(fit_piezo, yeval, color='red', label='Initial Hertz Fit')
    plt.plot(fit_piezo, fit_load, label= 'Loading Phase Curve')

    plt.title(f'{xy_key} - Hertz Model Fit to Loading Curve')
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(format_micron))
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(format_nano))
    plt.xlabel('Indentation (m)')
    plt.ylabel('Load (N)')
    # plt.ylim(-0.05e-6,2e-6)
    # plt.xlim(0,40e-6)
    plt.legend(loc = 'upper left')
    
    if SaveHertztoPDF == True:
        pdf.savefig()
        plt.close()
    else:
        plt.show()

    data_dict[xy_key].update ({
        "fitresult_fixed": fitresult_fix,
        "Ind_mod_Hertz_fixed": Ind_mod_fix,
        "ContactPoint_Hertz_fixed": ContactPoint_fix,
        "Rsq_Hertz_fixed": Rsq_fix
        })
    
    return data_dict


# --------------------------------------------------------------------------------------
# 2)OP
# --------------------------------------------------------------------------------------



def OliverPharr(data_dict, xy_key, epsilon, Beta, poissonratio):
    if data_dict[xy_key].get("split_failed", False) or len(data_dict[xy_key]["unloading_piezo"]) == 0:
        print(f"[{xy_key}] Skipping OliverPharr: split failed.")
        data_dict[xy_key].update({
            "OliverPharr_YM": np.nan,
            "OliverPharr_Rsq": np.nan,
            "OliverPharr_slope": np.zeros(1)  # or np.nan or empty array depending on usage
        })
        return data_dict
    piezo_OP, load_OP = data_dict[xy_key]["unloading_piezo"], data_dict[xy_key]["unloading_load"]
    radius =data_dict[xy_key]["radius"]
    
    # perform a second order poly fit to the top 50% of the unloading data
    newperc = int(len(load_OP) * 0.5)
    c2, c1,c0 = np.polyfit(piezo_OP[:newperc], load_OP[:newperc], 2)
    load_pred = (c2*piezo_OP[:newperc]**2) + (c1*piezo_OP[:newperc]) +c0
    
    fiftyperc = int(len(load_pred) * 0.5)
    
    # Perform initial fit on the first 50% of the polyfit
    S, plus_C = np.polyfit(piezo_OP[:fiftyperc], load_pred[:fiftyperc], 1)
    slope = S * piezo_OP[:fiftyperc] + plus_C
    P_max = np.max(load_pred)
    h_max = np.max(piezo_OP) - np.min(piezo_OP)
    h_c = h_max - (epsilon * (P_max / S))
    A = np.pi * ((2 * radius * h_c) - (h_c**2))
    Eff = np.round((S / Beta) * (np.sqrt(np.pi) / 2) / np.sqrt(A), 2)
    youngs_modulus = (1 - poissonratio**2) / (1 / Eff)
    OP_YM = np.round(youngs_modulus, 2)

    # Calculate R-squared for the linear fit on the first 10% of the predicted data
    residuals = load_OP[:fiftyperc] - slope
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((load_pred[:fiftyperc] - np.mean(load_pred[:fiftyperc]))**2)
    r_squared = np.round(1 - (ss_res / ss_tot) if ss_tot != 0 else 0, 3)
    data_dict[xy_key].update({
        "OliverPharr_YM": OP_YM,
        "OliverPharr_Rsq": r_squared,
        "OliverPharr_slope": slope
        })

    return data_dict

def PlotOliverpharr(data_dict, xy_key, SaveOPtoPDF):
    try:
        piezo_OP, load_OP, slope = data_dict[xy_key]["unloading_piezo"], data_dict[xy_key]["unloading_load"], data_dict[xy_key]["OliverPharr_slope"]
        pdf = data_dict[xy_key]["pdf"]
        newperc = int(len(load_OP) * 0.5)
        c2, c1,c0 = np.polyfit(piezo_OP[:newperc], load_OP[:newperc], 2)
        load_pred = (c2*piezo_OP[:newperc]**2) + (c1*piezo_OP[:newperc]) +c0
        fiftyperc = int(len(load_pred) * 0.5)
    
        plt.plot(piezo_OP, load_OP, label='Unloading Phase Curve', color='#3b719f', linewidth=2)
        plt.plot(piezo_OP[:newperc], load_pred,  label='Initial Second order PolyFit', color='orange', linewidth=3, linestyle='dotted')
        plt.plot(piezo_OP[:fiftyperc], slope, label='Linear Fit to PolyFit', color='red')
        
        plt.title(f'{xy_key} - OliverPharr Model ')
        plt.gca().xaxis.set_major_formatter(FuncFormatter(format_micron))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_nano))
        plt.legend(loc='upper left')
        plt.xlabel('Indentation (µm)')
        plt.ylabel('Load (nN)')
        
        
        if SaveOPtoPDF == True:
            pdf.savefig()
            plt.close()
        else:
            plt.show()
    except:
        print('Plot OP failed')
    return
    

# --------------------------------------------------------------------------------------
# 2) Viscoelasticity
# --------------------------------------------------------------------------------------



def Viscoelasticity(data_dict, xy_key, SaveHoldtoPDF=True):
    """
    Simple holding-phase metrics:
      - Hold_LoadDrop = max(load_hold) - min(load_hold)
      - Hold_TimeHeld = max(time_hold) - min(time_hold)

    Stores results into data_dict[xy_key] and optionally saves a plot to the PDF.
    """
    if data_dict[xy_key].get("split_failed", False):
        print(f"[{xy_key}] Skipping Viscoelasticity: split failed.")
        data_dict[xy_key].update({
            # "Hold_LoadDrop": np.nan,
            "Hold_TimeHeld": np.nan
        })
        return data_dict

    d = data_dict[xy_key]

    # Try common holding keys (keeps your code resilient to naming)
    piezo = d.get("holding_piezo", d.get("hold_piezo", d.get("holding_phase_piezo", None)))
    load  = d.get("holding_load",  d.get("hold_load",  d.get("holding_phase_load",  None)))
    time  = d.get("holding_time",  d.get("hold_time",  d.get("holding_phase_time",  None)))

    # If time wasn't stored explicitly, fall back to full time indices if available
    if time is None:
        time = d.get("time_data", None)

    if piezo is None or load is None or len(load) == 0:
        print(f"[{xy_key}] Skipping Viscoelasticity: no holding data found.")
        data_dict[xy_key].update({
            # "Hold_LoadDrop": np.nan,
            "Hold_TimeHeld": np.nan
        })
        return data_dict

    piezo = np.asarray(piezo)
    load  = np.asarray(load)
    time  = np.asarray(time) if time is not None else None

    hold_load_drop = float(np.nanmax(load) - np.nanmin(load))
    hold_time_held = float(np.nanmax(time) - np.nanmin(time)) if time is not None and len(time) == len(load) else np.nan
    # print(hold_load_drop, hold_time_held)
    data_dict[xy_key].update({
        # "Hold_LoadDrop": hold_load_drop,
        "Hold_TimeHeld": hold_time_held
    })

    PlotHoldingPhase(data_dict, xy_key, SaveHoldtoPDF=SaveHoldtoPDF)
    
    # print("drop in Load(N)", hold_load_drop,"time held at max Load(s)",  hold_time_held)

    return data_dict

def PlotHoldingPhase(data_dict, xy_key, SaveHoldtoPDF=True):
    """
    Plot holding phase: load vs piezo (and optionally annotate load drop/time held).
    """
    try:
        d = data_dict[xy_key]
        pdf = d.get("pdf", None)

        piezo = d.get("holding_piezo", d.get("hold_piezo", d.get("holding_phase_piezo", None)))
        load  = d.get("holding_load",  d.get("hold_load",  d.get("holding_phase_load",  None)))
        time = d.get("holding_time", d.get("hold_time", d.get("holding_phase_time", None)))
        
        if piezo is None or load is None or len(load) == 0:
            print(f"[{xy_key}] PlotHoldingPhase: no holding data found.")
            return

        piezo = np.asarray(piezo)
        load  = np.asarray(load)
        time  = np.asarray(time)

        plt.plot(time, load, linewidth=2)


        title_bits = [f"{xy_key} - Holding phase"]
        ld = d.get("Hold_LoadDrop", None)
        th = d.get("Hold_TimeHeld", None)
        if ld is not None and not np.isnan(ld):
            title_bits.append(f"LoadDrop={ld:.3g} N")
        if th is not None and not np.isnan(th):
            title_bits.append(f"TimeHeld={th:.3g} s")

        plt.title(" | ".join(title_bits))
        plt.xlabel("time (s)")
        plt.ylabel("Load (N)")



        if SaveHoldtoPDF and pdf is not None:
            pdf.savefig()
            plt.close()
        else:
            plt.show()
            
        

    except Exception as e:
        print(f"[{xy_key}] PlotHoldingPhase failed: {e}")

    return

def ViscoelasticityHold(data_dict, xy_key, SaveHoldtoPDF=False):
    d = data_dict[xy_key]
    if d.get("split_failed", False) or len(d.get("holding_time", [])) < 10:
        d.update({
            "Hold_TimeHeld": np.nan,
            "Hold_F0": np.nan,
            "Hold_Fend": np.nan,
            "Hold_RelaxFrac": np.nan,
            # "Hold_dFdt0": np.nan,
            # "Hold_norm_dFdt0": np.nan,
            # "Hold_tau": np.nan,
            # "Hold_AUC_norm": np.nan
        })
        return data_dict

    t = np.asarray(d["holding_time"], dtype=float)
    F = np.asarray(d["holding_load"], dtype=float)

    # make time start at 0 for fitting
    t0 = t[0]
    t = t - t0

    # Basic durations
    T = float(t[-1] - t[0])
    d["Hold_TimeHeld"] = T

    # Use robust estimates: mean of first/last 10% (less noise)
    n = len(F)
    k = max(3, int(0.1*n))
    F0 = float(np.mean(F[:k]))
    Fend = float(np.mean(F[-k:]))

    d["Hold_F0"] = F0
    d["Hold_Fend"] = Fend
    d["Hold_RelaxFrac"] = float((F0 - Fend) / F0) if F0 != 0 else np.nan
    
    # print("RelaxFrac", float((F0 - Fend) / F0) )
    # Initial rate over first 0.4 s (or first 20% if very short)
    # t_rate = 0.4
    # m = t <= min(t_rate, 0.2*T if T > 0 else t_rate)
    # if np.sum(m) >= 5:
    #     slope, intercept = np.polyfit(t[m], F[m], 1)
    #     dFdt0 = float(slope)  # N/s (likely negative)
    #     d["Hold_dFdt0"] = dFdt0
    #     d["Hold_norm_dFdt0"] = float(dFdt0 / F0) if F0 != 0 else np.nan
    # else:
    #     d["Hold_dFdt0"] = np.nan
    #     d["Hold_norm_dFdt0"] = np.nan

    # Single exponential fit for tau
    def exp1(t, Finf, A, tau):
        return Finf + A*np.exp(-t/tau)

    tau = np.nan
    try:
        # initial guesses
        Finf0 = Fend
        A0 = F0 - Fend
        tau0 = max(0.2, 0.3*T)
        popt, _ = curve_fit(
            exp1, t, F,
            p0=[Finf0, A0, tau0],
            bounds=([-np.inf, -np.inf, 1e-4], [np.inf, np.inf, np.inf]),
            maxfev=5000
        )
        Finf_fit, A_fit, tau_fit = popt
        tau = float(tau_fit)
        d["Hold_tau"] = tau
    except Exception:
        d["Hold_tau"] = np.nan

    # # Normalised AUC
    # if F0 != 0:
    #     d["Hold_AUC_norm"] = float(np.trapezoid(F/F0, t))
    # else:
    #     d["Hold_AUC_norm"] = np.nan

    # (Optional plotting can be added later)
    return data_dict


def visco_P_analytic(t, G0, tau, G1, R, v0, t_ramp):
    t = np.asarray(t, float)
    tau = max(float(tau), 1e-6)

    a = np.maximum(np.minimum(t, float(t_ramp)), 0.0)
    a_safe = np.maximum(a, 1e-15)

    x = np.sqrt(a_safe / tau)
    bracket = np.sqrt(a_safe / tau) * np.exp(a_safe / tau) - (np.sqrt(np.pi) / 2.0) * erfi(x)

    pref = (16.0 * np.sqrt(R) / 3.0) * (3.0 / 2.0) * (v0 ** 1.5)
    P = pref * (G0 * (tau ** 1.5) * np.exp(-t / tau) * bracket + (2.0 / 3.0) * G1 * (a_safe ** 1.5))

    P = np.asarray(P, float)
    P[~np.isfinite(P)] = 1e30
    return P

def ViscoAnalyticFit(data_dict, xy_key, tol_um=5.0, trim_s=0.15, SaveToPDF=True):
    """
    Uses Hertz CP unless |Hertz-RoV| > tol_um, then uses RoV CP.
    Builds fit window = (approach+loading from CP onward) + holding.
    Fits analytic model and stores results in data_dict[xy_key].
    Assumes your preprocessing already handled baseline / drift / units.
    """
    d = data_dict[xy_key]
    if d.get("split_failed", False):
        d.update({"ViscoAna_success": False})
        return data_dict

    # ---------- CP selection (Hertz-first, RoV only if catastrophic mismatch) ----------
    cp_h = d.get("ContactPoint_Hertz", np.nan)  # m
    cp_r = d.get("ContactPoint_RoV",   np.nan)  # m
    cp_used, cp_src, cp_diff_um = np.nan, "None", np.nan

    h_ok, r_ok = np.isfinite(cp_h), np.isfinite(cp_r)
    if h_ok and not r_ok:
        cp_used, cp_src = float(cp_h), "Hertz"
    elif r_ok and not h_ok:
        cp_used, cp_src = float(cp_r), "RoV"
    elif h_ok and r_ok:
        cp_diff_um = abs(float(cp_h) - float(cp_r)) * 1e6
        cp_used, cp_src = (float(cp_r), "RoV") if cp_diff_um > float(tol_um) else (float(cp_h), "Hertz")

    d.update({"CP_used_m": cp_used, "CP_used_method": cp_src, "CP_diff_um": cp_diff_um})
    if not np.isfinite(cp_used):
        d.update({"ViscoAna_success": False})
        return data_dict

    # ---------- Build arrays: (whole_approach to end of loading) is already stored as whole_approach_* ----------
    # whole_approach_* goes from start to end of loading (your split uses [:ij])
    piezoA = np.asarray(d["whole_approach_piezo"], float)
    tA     = np.asarray(d["whole_approach_time"],  float)
    P_A    = np.asarray(d["whole_approach_load"],  float)

    tH     = np.asarray(d["holding_time"], float)
    P_H    = np.asarray(d["holding_load"], float)

    # nearest index in whole_approach to CP
    i0 = int(np.nanargmin(np.abs(piezoA - cp_used))) if piezoA.size else 0

    # fit window: from CP through end-of-loading + holding
    t_abs = np.concatenate([tA[i0:], tH])
    P     = np.concatenate([P_A[i0:], P_H])

    # time relative to CP-start
    t = t_abs - float(t_abs[0])

    # ---------- Ramp parameters: use profile interpretation (DZ1,t1) if present; else use loading arrays ----------
    R = float(d["radius"])  # metres

    # time at CP in the whole_approach arrays
    t_cp = float(tA[i0])
    
    # ---- Robust ramp duration ----
    lt = np.asarray(d.get("loading_time", []), float)
       
    if lt.size >= 2 and np.isfinite(lt).all():
        # if loading_time is relative (common), duration is just last-first
        lt0, lt1 = float(lt[0]), float(lt[-1])
        if abs(lt0) < 1e-6:   # relative time starting ~0
            t_ramp = max(lt1 - lt0, 1e-6)
        else:
            # absolute time array
            t_ramp = max(lt1 - t_cp, 1e-6)
    else:
        # fallback: whole_approach goes to end-of-loading
        if tA.size >= 2 and np.isfinite(tA).all():
            t_ramp = max(float(tA[-1]) - t_cp, 1e-6)
        else:
            d.update({"ViscoAna_success": False})
            return data_dict
    # indentation at CP from whole_approach indentation
    indA = np.asarray(d["whole_approach_indentation"], float)
    ind_cp = float(indA[i0])
    
    # max indentation at end of loading (use loading_indent)
    li = np.asarray(d["loading_indent"], float)
    ind_end = float(np.nanmax(li)) if li.size else float(np.nanmax(indA))
    
    DZ = max(ind_end - ind_cp, 0.0)
    v0 = max(DZ / t_ramp, 1e-12)

    # ---------- Fit ----------
    model = Model(visco_P_analytic, independent_vars=["t"])
    params = model.make_params(G0=500.0, tau=0.6, G1=200.0)

    params["tau"].set(value=0.5, min=1e-3, max=20.0)
    params["G0"].set(value=500.0, min=0.0, max=2e5)
    params["G1"].set(value=200.0, min=0.0, max=2e5)

    params.add("R", value=R, vary=False)
    params.add("v0", value=v0, vary=False)
    params.add("t_ramp", value=t_ramp, vary=False)

    # trim early transient (optional)
    mask = t >= float(trim_s)
    out = model.fit(P[mask], params, t=t[mask], nan_policy="omit", method="powell")

    # ---------- store outputs ----------
    nu = 0.5
    G0 = float(out.params["G0"].value)
    G1 = float(out.params["G1"].value)
    tau = float(out.params["tau"].value)
    E0 = 2.0 * (G0 + G1) * (1.0 + nu)
    Einf = 2.0 * G1 * (1.0 + nu)

    d.update({
        "ViscoAna_G0": G0, "ViscoAna_G1": G1, "ViscoAna_tau": tau,
        "ViscoAna_E0": E0, "ViscoAna_Einf": Einf,
        "ViscoAna_success": bool(out.success),
        "ViscoAna_chisq": float(out.chisqr),
    })
    
    # --- Fit metrics ---
    y = np.asarray(P[mask], float)          # data that was fit
    yhat = np.asarray(out.best_fit, float)  # model on those points
    
    resid = y - yhat                        # same as out.residual
    rmse = float(np.sqrt(np.nanmean(resid**2)))
    redchi = float(out.redchi) if out.redchi is not None else np.nan
    
    ss_res = float(np.nansum((y - yhat)**2))
    ss_tot = float(np.nansum((y - np.nanmean(y))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    
    d.update({
        "ViscoAna_redchi": redchi,
        "ViscoAna_rmse": rmse,
        "ViscoAna_r2": r2,
    })
    
    # --- Prediction uncertainty band (1σ) ---
    try:
        P_sigma = out.eval_uncertainty(t=t)
    except Exception:
        P_sigma = np.full_like(out.best_fit, np.nan, dtype=float)
    
    d["ViscoAna_pred_sigma_mean"] = float(np.nanmean(P_sigma)) if np.isfinite(P_sigma).any() else np.nan

    # ---------- PDF plot ----------
    if SaveToPDF and d.get("pdf", None) is not None:
        P_fit = out.eval(t=t)
        fig, ax = plt.subplots()
        # approach pre-CP (for context)
        ax.plot(tA, P_A, label="Approach+Loading (raw)")
        ax.axvline(float(tA[i0]) if tA.size else 0.0, linestyle="--", label=f"CP used ({cp_src})")
        # fit window + fit
        ax.plot(t_abs, P, label="Load+Hold (from CP)")
        ax.plot(t_abs, P_fit, "--", linewidth=2.0, label="Analytic fit")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Load (N)")
        ax.set_title(f"{xy_key} - Analytic Visco Fit")
        ax.grid(True)
        ax.legend(loc="best")
        plt.tight_layout()
        # plt.show()
        d["pdf"].savefig(fig)
        plt.close(fig)

    return data_dict





# --------------------------------------------------------------------------------------
# 2) Write to CSV the outputs
# --------------------------------------------------------------------------------------


def WritetoCSV(csv_writer, data_dict, xy_key):
    d = data_dict[xy_key]

    row_data = [
        d.get("file_name", ""),
        d.get("x_number", np.nan),
        d.get("y_number", np.nan),

        d.get("eff_modulus", np.nan),
        d.get("fileYM", np.nan),

        d.get("ContactPoint_Hertz", np.nan),
        d.get("Ind_mod_Hertz", np.nan),
        d.get("Rsq_Hertz", np.nan),

        d.get("ContactPoint_RoV", np.nan),
        d.get("OliverPharr_YM", np.nan),
        d.get("OliverPharr_Rsq", np.nan),

        d.get("Hold_TimeHeld", np.nan),
        d.get("Hold_F0", np.nan),
        d.get("Hold_Fend", np.nan),
        d.get("Hold_RelaxFrac", np.nan),

        d.get("ViscoAna_tau", np.nan),
        d.get("ViscoAna_G0", np.nan),
        d.get("ViscoAna_G1", np.nan),
        d.get("ViscoAna_E0", np.nan),
        d.get("ViscoAna_Einf", np.nan),

        d.get("ViscoAna_redchi", np.nan),
        d.get("ViscoAna_rmse", np.nan),
        d.get("ViscoAna_r2", np.nan),
    ]

    csv_writer.writerow(row_data)
    return
    
def CreateCSV(output_folderCSV, outputfilenameCSV):
    if not os.path.exists(output_folderCSV):
        os.makedirs(output_folderCSV)

    csvfile = open(os.path.join(output_folderCSV, outputfilenameCSV), "w", newline="")
    csv_writer = csv.writer(csvfile)

    headers = [
        "File Name", "x", "y",
        "Eff modulus from file", "modulus from file",

        "Hertz - Contact Point",
        "Hertz - Modulus(Pa) fit",
        "Hertz - Rsq",

        "RoV - Contact Point",
        "OP - Modulus",
        "OP - Rsq",

        "Holding - Time Held (s)",

        "Hold - Load Start",
        "Hold - Load End",
        "Hold - Relaxation Fraction",
        
        "Visco (Analytic) - tau (s)",
        "Visco (Analytic) - G0 (Pa)",
        "Visco (Analytic) - G1 (Pa)",
        "Visco (Analytic) - E0 (Pa)",
        "Visco (Analytic) - E_inf (Pa)",
        
        "ViscoAna_redchi",
        "ViscoAna_rmse",
        "ViscoAna_r2",
    ]

    csv_writer.writerow(headers)
    return csvfile, csv_writer
    












# --------------------------------------------------------------------------------------
# 2) Hertz-style "stress–strain" using piezo (loading phase) + Hertz CP
# --------------------------------------------------------------------------------------
def StressStrain_Hertz_Loading_fromPiezo(data_dict, xy_key):
    """
    Option A: indentation-derived stress–strain using piezo depth:
      delta = loading_piezo - ContactPoint_Hertz  (post-contact only)
      A = pi * R * delta
      stress = P / A   (Pa)
      strain = delta / R  (dimensionless)

    Stores:
      SS_strain_loading, SS_stress_loading
    """
    if data_dict[xy_key].get("split_failed", False):
        data_dict[xy_key].update({
            "SS_strain_loading": np.array([]),
            "SS_stress_loading": np.array([])
        })
        return data_dict

    d = data_dict[xy_key]
    R = float(d.get("radius", np.nan))  # m
    z = np.asarray(d.get("loading_piezo", []), dtype=float)  # m
    P = np.asarray(d.get("loading_load",  []), dtype=float)  # N
    z_cp = d.get("ContactPoint_Hertz", np.nan)               # m

    if not (np.isfinite(R) and np.isfinite(z_cp)) or len(z) == 0 or len(P) == 0:
        d.update({"SS_strain_loading": np.array([]), "SS_stress_loading": np.array([])})
        return data_dict

    # Contact depth from piezo
    delta = z - float(z_cp)

    # Keep only post-contact points
    m = delta > 0
    delta = delta[m]
    P = P[m]

    if len(delta) == 0:
        d.update({"SS_strain_loading": np.array([]), "SS_stress_loading": np.array([])})
        return data_dict

    A = np.pi * R * delta     # m^2
    stress = P / A            # Pa
    strain = delta / R        # dimensionless

    d.update({
        "SS_strain_loading": strain,
        "SS_stress_loading": stress
    })
    return data_dict


# --------------------------------------------------------------------------------------
# 3) Plot stress–strain (loading)
# --------------------------------------------------------------------------------------
def PlotStressStrain_Loading(data_dict, xy_key, SaveToPDF=True):
    """
    Plots SS_stress_loading vs SS_strain_loading.
    """
    d = data_dict[xy_key]
    strain = d.get("SS_strain_loading", None)
    stress = d.get("SS_stress_loading", None)

    if strain is None or stress is None or len(strain) == 0:
        print(f"[{xy_key}] Stress–strain plot skipped (no data).")
        return

    plt.plot(strain, stress, linewidth=2)
    plt.xlabel("Indentation strain (δ/R)")
    plt.ylabel("Mean contact pressure (Pa)")
    plt.title(f"{xy_key} - Loading stress–strain (piezo + Hertz CP)")

    pdf = d.get("pdf", None)
    if SaveToPDF and pdf is not None:
        pdf.savefig()
        plt.close()
    else:
        plt.show()

    return







