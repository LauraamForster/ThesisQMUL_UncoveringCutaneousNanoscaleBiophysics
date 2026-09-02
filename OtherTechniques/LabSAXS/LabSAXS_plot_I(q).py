#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:56:08 2024

@author: lauraforster
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from lmfit.models import ExponentialModel, ConstantModel

# Specify the base data directory
data_directory = '/Users/lauraforster/Documents/Uni/3 - PhD/Other Techniques/Nano-InXider/data/realData/16july'

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

def PlotSettings(dat_file):
    # plt.ylim(20e-10, 10e-9)
    # plt.xlim(0.2,0.4)
    # plt.axvline(x=0.1, color='g', linestyle=':')
    # plt.axvline(x=0.3, color='r', linestyle=':')
    # plt.axvline(x=0.49, color='r', linestyle=':')
    # plt.axvline(x=0.7, color='g', linestyle=':')
    # plt.axvline(x=1.55, color='black', linestyle=':')
    plt.yscale('log')
    plt.xlabel('q (nm⁻¹)')
    plt.ylabel('I(q)_transmitted')
    plt.title(f'{dat_file}')
    # plt.legend()
    return

# Function to process data and plot
def process_and_plot(data_directory):
    resolution_dict = {}
    
    # Find all .dat files in the directory
    dat_files = [f for f in os.listdir(data_directory) if f.endswith('.dat')]

    # Loop through each .dat file and process it
    for dat_file in dat_files:
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
            sample_name, resolution, exposure_time = dat_file[:-4].split('_')
            
            if resolution not in resolution_dict:
                resolution_dict[resolution] = {}
            if exposure_time not in resolution_dict[resolution]:
                resolution_dict[resolution][exposure_time] = []

            resolution_dict[resolution][exposure_time].append((q, I_q_transmitted, dat_file))

    # Plot the data
    for resolution, exposure_dict in resolution_dict.items():
        plt.figure(figsize=(10, 8))
        for exposure_time, data_list in exposure_dict.items():
            for q, I_q_transmitted, dat_file in data_list:
                plt.plot(q, I_q_transmitted, label=f'{dat_file}')
        
        PlotSettings(dat_file)
        plt.show()



process_and_plot(data_directory)
