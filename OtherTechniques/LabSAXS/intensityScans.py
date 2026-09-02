#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:54:18 2024

@author: lauraforster
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d
import pandas as pd

data_directory = '/Users/lauraforster/Documents/Uni/3 - PhD/Other Techniques/Nano-InXider/data/Practice data/intensity scans/'

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

# Function to process data and plot
def process_and_plot(data_directory):
    file_dict = {}
    
    # Find all .dat files in the directory
    dat_files = [f for f in sorted(os.listdir(data_directory)) if f.endswith('.txt')]
    print(dat_files)
    # Group files by sample_name and fileno
    for dat_file in dat_files:

        file_path = os.path.join(data_directory, dat_file)
        data, header = read_data(file_path)
        
        data_array = []
        for line in data[1:]:  # Skip the first line as it is the header
            parts = line.split()
            data_array.append([float(part) for part in parts])

        data2 = np.array(data_array)
        q_angstroms = data2[:, 0]
        I_q = data2[:, 1]


        # Convert q from inverse Angstroms to inverse nanometers
        q = q_angstroms * 10
        
        # Apply the Intensity1 factor to I(q)
        I_q_transmitted = I_q 
      
        # Plot the data
        plt.plot(q, I_q_transmitted, label=f': {dat_file}')
        
        plt.xlabel('q (nm⁻¹)')
        plt.ylabel('I(q)_transmitted')
        # plt.xlim(0.01,20)
        # plt.ylim(1e-11, 1e-8)
        # plt.yscale('log')
        # plt.title(f'{sample_name} - File {fileno}')
        # plt.legend(lo)
        plt.show()

process_and_plot(data_directory)

