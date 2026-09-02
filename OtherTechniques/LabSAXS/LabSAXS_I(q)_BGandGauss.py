#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:56:08 2024

@author: lauraforster
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d
from lmfit.models import ExponentialModel, ConstantModel, GaussianModel
import sys
import pandas as pd

# Specify the base data directory
data_directory = '/Users/lauraforster/Documents/Uni/3 - PhD/Other Techniques/Nano-InXider/data/realData/Elis'

# Function to read data and header from a file
def read_data(file_path):
    with open(file_path, 'r') as file:
        contents = file.readlines()

    header_lines = [line.strip() for line in contents if line.strip().startswith('#')]
    data_lines = [line.strip() for line in contents if not line.strip().startswith('#')]

    return data_lines, header_lines

# Function to extract the Intensity1 value from the header
def extract_intensity1(header_lines):
    for line in header_lines:
        if line.startswith('# Intensity1'):
            return float(line.split()[2])
    return 1.0  # Default value if Intensity1 is not found

def exponential(x, A, b, d):
    return A * np.exp(x * b) + d

def Cutting(x, y, order, x_peak_wid, xborderLL, xborderLR, xborderRL, xborderRR, fileno):
    x_peak = x[(x >= order - x_peak_wid) & (x < order + x_peak_wid)]
    x_L = x[(x >= order - xborderLL) & (x < order - xborderLR)]
    x_R = x[(x >= order + xborderRL) & (x < order + xborderRR)]

    y_peak = y[(x >= order - x_peak_wid) & (x < order + x_peak_wid)]
    y_L = y[(x >= order - xborderLL) & (x < order - xborderLR)]
    y_R = y[(x >= order + xborderRL) & (x < order + xborderRR)]

    x_sides = np.concatenate((x_L, x_R))
    y_sides = np.concatenate((y_L, y_R))
    x_whole = np.concatenate((x_L, x_peak, x_R))
    y_whole = np.concatenate((y_L, y_peak, y_R))
    
    plt.scatter(x_whole, y_whole, color='green')
    plt.plot(x_L, y_L, color='red')
    plt.plot(x_R, y_R, color='red')
    plt.title(f': {fileno}')
    
    if np.any(y_whole > 0):
        plt.yscale('log')
    plt.show()

    return x_whole, y_whole, x_sides, y_sides

def BackgroundSubtraction(x_sides, y_sides, x_whole, y_whole, fileno):
    try:
        if len(x_sides) == 0 or len(y_sides) == 0:
            raise ValueError("Empty sequence for background subtraction.")
            
        eModel = ExponentialModel()
        cModel = ConstantModel()
        pars = eModel.guess(y_sides, x_sides)
        pars['decay'].min = 0.0001  # Ensure decay parameter is positive and small enough to avoid concavity
        pars += cModel.make_params(c=np.min(y_sides))
        fullModel = eModel + cModel
        result = fullModel.fit(y_sides, params=pars, x=x_sides)
        y_bgr = fullModel.eval(result.params, x=x_whole)

        plt.plot(x_whole, y_whole, color='green', label='whole')
        plt.plot(x_sides, y_sides, color='red', label='sides')
        plt.plot(x_whole, y_bgr, label='fit', color='blue')
        
        plt.axvline(x=0.3, color='r', linestyle=':')
        if np.any(y_whole > 0):
            plt.yscale('log')
        plt.xlabel('q (nm⁻¹)')
        plt.ylabel('I(q)_transmitted')
        plt.title(f'background subtraction: {fileno}')
        plt.legend()
        plt.show()
        
        y_BG_sub = y_whole - y_bgr
        passed = 1

    except Exception as e:
        print(f"Error in background subtraction: {e}")
        y_BG_sub, y_bgr, passed = 0, 0, 0
        plt.plot(x_whole, y_whole, label='No Background Fit')

    return y_BG_sub, y_bgr, passed

def GaussianFit(peak_prominance, x_whole, y_BG_sub, order, fileno):
    if peak_prominance > 0:
        peak_position, D_period, peak_width, peak_amplitude, peak_height = 0, 0, 0, 0, 0
        gmodel_BGfit = GaussianModel()
        initparams = gmodel_BGfit.guess(y_BG_sub, x=x_whole)
        initparams['center'].max = 0.305
        initparams['center'].min = 0.275
        result_BGfit = gmodel_BGfit.fit(y_BG_sub, x=x_whole, params=initparams)
        bgr_gauss = result_BGfit.eval(params=result_BGfit.params, x=x_whole)
        
        peak_position = x_whole[bgr_gauss.argmax()]
        D_period = (2 * np.pi) / (peak_position / 3)
        peak_width = PeakWidth(peak_position, result_BGfit, x_whole, bgr_gauss)
        peak_amplitude = result_BGfit.params['amplitude'].value
        peak_height = np.max(bgr_gauss)
        
        plt.plot(x_whole, y_BG_sub)
        plt.plot(x_whole, bgr_gauss)
        plt.title(f'Third order peak with Gaussian fit: {fileno}')
        plt.show()
    else:
        peak_position, D_period, peak_width, peak_amplitude, peak_height = 0, 0, 0, 0, 0
    return peak_position, D_period, peak_width, peak_amplitude, peak_height

def PeakWidth(peak_position, result_BGfit, x_BG_sub, bgr_gauss):
    sigma = result_BGfit.params['sigma'].value
    fwhm = 2 * sigma * np.sqrt(2 * np.log(2))

    x_left, x_right = peak_position - fwhm, peak_position + fwhm
    peak_width = x_right - x_left
    return peak_width

def smooth_data(y, sigma=0.7):
    return gaussian_filter1d(y, sigma=sigma)

# Function to process data and plot
# def process_and_plot(data_directory):
resolution_dict = {}
results=[]

# Find all .dat files in the directory
dat_files = [f for f in sorted(os.listdir(data_directory)) if f.endswith('.dat')]

# Loop through each .dat file and process it
for dat_file in dat_files:
    print(dat_file)
    file_path = os.path.join(data_directory, dat_file)
    data, header = read_data(file_path)
    if data is not None:
        intensity1 = extract_intensity1(header)
        
        data_array = []
        for line in data[1:]:  # Skip the first line as it is the header
            parts = line.split()
            data_array.append([float(part) for part in parts])

        data2 = np.array(data_array)
        q_angstroms = data2[:, 0]
        I_q = data2[:, 1]
        Sig_q = data2[:, 2]

        # Convert q from inverse Angstroms to inverse nanometers
        q = q_angstroms * 10
        
        # Apply the Intensity1 factor to I(q)
        I_q_transmitted = I_q * intensity1
        
        # Parse the file name to get sample name, resolution, and exposure time
        sample_name, n, fileno = dat_file[:-4].split('_')

        if n not in resolution_dict:
            resolution_dict[n] = {}
        if fileno not in resolution_dict[n]:
            resolution_dict[n][fileno] = []

        resolution_dict[n][fileno].append((q, I_q_transmitted, dat_file))
   
        # Background subtraction around third order peak
        order = 0.29
        x_peak_wid = 0.05  # Updated peak width
        xborderLL, xborderLR = 0.11, 0.012  # Updated left borders
        xborderRL, xborderRR = 0.022, 0.11
        
        # area_thresh = 0.000000000001
        area_thresh = 1e-12
        order_no = 3
        
        total_SAXS_area = np.sum(I_q_transmitted)
        x_whole, y_whole, x_sides, y_sides = Cutting(q, I_q_transmitted, order, x_peak_wid, xborderLL, xborderLR, xborderRL, xborderRR, fileno)
        I_q_transmitted = smooth_data(I_q_transmitted)
        y_BG_sub, y_bgr, passed = BackgroundSubtraction(x_sides, y_sides, x_whole, y_whole, fileno)
        
        if passed == 0:
            area = 0
        else:
            area = np.trapz(y=y_BG_sub, x=x_whole)

        if area > area_thresh:
            peak_prominence = 1
        else:
            peak_prominence = 0
        
            
        peak_position, D_period, peak_width, peak_amplitude, peak_height = GaussianFit(peak_prominence, x_whole, y_BG_sub, order_no, fileno)
        print('Dperiod', D_period)
        print('height', peak_height)

        results.append({
                            'filename': dat_file,
                            'filenumber': fileno,
                            'total_SAXS_intensity': total_SAXS_area,
                            'area_under_third_order_curve': area,
                            'peak_position_third': peak_position,
                            'D_period_third': D_period,
                            'peak_width_third': peak_width,
                            'peak_amplitude_third': peak_amplitude,
                            'peak_height_third': peak_height
                        })
        
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(data_directory, 'SAXS_analysis_results23july.csv'), index=False)

# process_and_plot(data_directory)
