#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 12:04:35 2024

@author: lauraforster
"""

import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np

def extract_values(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the line containing X and Y values
    xy_line = [line.strip() for line in lines if "Scan (#)" in line][0]


    # Extract X and Y values
    x_value = float(xy_line.split('\t')[3])
    y_value = float(xy_line.split('\t')[5])


    # Find the line containing Z position
    z_position_line = [line.strip() for line in lines if "Z-position (um)" in line][0]

    # Extract Z position value
    z_position = float(z_position_line.split('\t')[1])

    return x_value, z_position

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

def plot_data(input_folder):
    # Get a list of all files in the folder
    file_list = os.listdir(input_folder)
    sorted_file_list = sorted(file_list, key=extract_xy_key)
    
    # Create empty lists to store X and Z values
    x_values = []
    z_values = []
        
    for file_name in sorted_file_list:
        if file_name.endswith(".txt") and "position" not in file_name and "Y-2" not in file_name:
            file_path = os.path.join(input_folder, file_name)


            # Extract X and Z values from the file
            x_value, z_value = extract_values(file_path)
    
            if x_value> 0:
                # Append values to the lists
                x_values.append(x_value)
                z_values.append(z_value)
        
    x_values = np.linspace(0, 100, len(x_values))
    print(z_values)
    maxy, miny =np.max(z_values), np.min(z_values)
    maxylim = maxy-miny
    z_values = maxy - z_values
    averagepos = np.average(z_values)

    plt.plot(np.linspace(np.max(x_values), np.min(x_values), 2), [averagepos, averagepos])
    plt.plot(np.linspace(np.max(x_values), np.min(x_values), 2), [averagepos-30, averagepos-30])
    plt.plot(np.linspace(np.max(x_values), np.min(x_values), 2), [averagepos+30, averagepos+30])
    a, b = 8, 9
    # print(len(z_values))
    plt.scatter(x_values, z_values, color='b', s=10)
    plt.scatter(x_values[a:-b], z_values[a:-b], color='orange', s=10)
    plt.xlabel('X Position')
    plt.ylabel('Z Position (um)')
    plt.title(f'{label} X vs Z Position')
    # plt.ylim(-10, maxylim+10)
    plt.show()

# Replace 'path_to_folder' with the actual path to your folder containing the files
folder = '/Users/lauraforster/Documents/Uni/3 - PhD/Nanoindentation/NI_Data/Data/Paper/Thicknesses/'
label = '60um 1'
end = '/matrix_scan01'
input_folder = folder + label + end
plot_data(input_folder)


