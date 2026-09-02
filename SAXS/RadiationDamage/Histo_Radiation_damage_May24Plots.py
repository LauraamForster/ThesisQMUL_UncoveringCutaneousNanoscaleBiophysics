#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 22:08:35 2024

@author: lauraforster
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
})

def parse_filename(filename):
    parts = filename.split()
    file_id = parts[0]
    exposure = parts[1].replace('s', '')
    repeat = parts[3]
    return file_id, exposure, repeat

def read_csv_files(folder, file_list):
    data_dict = {}
    for file_info in file_list:
        file_id, exposure, repeat = parse_filename(file_info)
        
        csv_file = os.path.join(folder, f"{file_id} IQ_smooth.csv")
        
        if os.path.exists(csv_file):
            # Read the CSV file with correct delimiter and header
            df = pd.read_csv(csv_file, delimiter=',')
            if exposure not in data_dict:
                data_dict[exposure] = {}
            if repeat not in data_dict[exposure]:
                data_dict[exposure][repeat] = {}
            for line_number in df['y'].unique():
                line_data = df[df['y'] == line_number]
                data_dict[exposure][repeat][line_number] = line_data

    return data_dict

def plot_param(data_dict, param, exposure_time, lines, color_map):
    plt.figure(figsize=(10, 6))
    exposure_str = str(exposure_time)
    if exposure_str in data_dict:
        for repeat in data_dict[exposure_str]:
            for line in lines:
                if line in data_dict[exposure_str][repeat]:
                    df = data_dict[exposure_str][repeat][line]
                    # Replace 0 values with NaN in the specified column
                    parameter = df[param].replace(0, np.nan)
                    color = color_map.get((int(repeat), int(line)))
                    plt.plot(df['x'], parameter, color=color, label=f'{exposure_time}s repeat {repeat} line {line}')
                    plt.scatter(df['x'], parameter, color=color, s=10)
    plt.xlabel('X Position')
    plt.ylabel(param)
    plt.title(f'{param} for {exposure_time}s Exposure Time')
    plt.legend()
    plt.show()
    
def plot_param_histogram(data_dict, param, exposure_time, lines, color_map):
    plt.figure(figsize=(10, 6))
    exposure_str = str(exposure_time)
    bin_width = 0.2  # Bin width of 0.2 nm
    min_value = 63  # Adjust these as needed based on your data range
    max_value = 67  # Adjust these as needed based on your data range
    bins = np.arange(min_value, max_value + bin_width, bin_width)

    if exposure_str in data_dict:
        for repeat in data_dict[exposure_str]:
            combined_data = []
            for line in lines:
                if line in data_dict[exposure_str][repeat]:
                    df = data_dict[exposure_str][repeat][line]
                    # Replace 0 values with NaN in the specified column
                    parameter = df[param].replace(0, np.nan)
                    combined_data.extend(parameter.dropna().values)
            color = color_map.get((int(repeat), 0))  # Use the color for line 0 as the repeat color
            plt.hist(combined_data, bins=bins, edgecolor='black', alpha=0.5, color=color, label=f'Repeat {repeat}')
    plt.xlabel(param)
    # plt.xlim(min_value, max_value)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {param} for {exposure_time}s Exposure Time')
    plt.legend()
    plt.show()



def plot_param_avg_std(data_dict, param, exposure_time, lines, color_map):
    averages = []
    std_devs = []
    labels = []
    
    exposure_str = str(exposure_time)
    if exposure_str in data_dict:
        for repeat in data_dict[exposure_str]:
            for line in lines:
                if line in data_dict[exposure_str][repeat]:
                    df = data_dict[exposure_str][repeat][line]
                    # Replace 0 values with NaN in the specified column
                    parameter = df[param].replace(0, np.nan)
                    avg = np.mean(parameter)
                    std = np.std(parameter)
                    averages.append(avg)
                    std_devs.append(std)
                    labels.append(f'Repeat {repeat} Line {line}')
    
    # Plot the averages with error bars
    plt.figure(figsize=(10, 6))
    # plt.ylim(60,70)
    plt.bar(labels, averages, yerr=std_devs, capsize=5, color=[color_map[(int(repeat), int(line))] for repeat in range(1, 4) for line in lines])
    plt.xlabel('Condition')
    plt.ylabel(param)
    plt.title(f'Average and Standard Deviation of {param} for {exposure_time}s Exposure Time')
    plt.xticks(rotation=45)
    plt.show()

def plot_combined_histograms(data_dict, param, lines, color_map):
    plt.figure(figsize=(12, 8))
    
    for exposure_time in data_dict:
        for repeat in data_dict[exposure_time]:
            for line in lines:
                if line in data_dict[exposure_time][repeat]:
                    df = data_dict[exposure_time][repeat][line]
                    parameter = df[param].replace(0, np.nan)
                    color = color_map.get((int(repeat), int(line)), None)
                    plt.hist(parameter.dropna().values, bins=30, edgecolor='black', alpha=0.5, color=color, label=f'{exposure_time}s repeat {repeat} line {line}', histtype='stepfilled')

    plt.xlabel(param)
    plt.ylabel('Frequency')
    plt.title(f'Combined Histogram of {param} for All Exposure Times and Repeats')
    plt.legend()
    plt.show()
    
# Folder path
folder = '/Volumes/Expansion/Documents/3 - PhD/SAXS/Radiation damage/'

# File list
file_list = [
    "753799 0.5s repeat 1",
    "753800 0.5s repeat 2",
    "753801 0.5s repeat 3",
    "753802 1s repeat 1",
    "753803 1s repeat 2",
    "753804 1s repeat 3",
    "753805 2s repeat 1",
    "753806 2s repeat 2",
    "753807 2s repeat 3",
    "753808 5s repeat 1",
    "753809 5s repeat 2",
]

color_map = {
    (1, 0): 'darkblue',
    (1, 1): 'darkblue',
    (2, 0): 'red',
    (2, 1): 'red',
    (3, 0): 'darkgreen',
    (3, 1): 'darkgreen'
}

# Read data into dictionary
data_dict = read_csv_files(folder, file_list)

# Parameters to choose from
# 'total SAXS intensity' 
# 'area under third order curve'
# 'D_period_third'
# 'peak_width_third'
# 'peak_height_third'
# 'scaled_peak_amp'
# 'scaled_peak_area'
# 'area_scaled_nonparam'

# Specify parameter, exposure time, and lines
# param = 'D_period_third'
# param='total SAXS intensity' 
param = 'area under third order curve'
# 'peak_width_third'
exposure = 5
lines = [1]

# Plot specified parameter for given exposure time and lines
plot_param(data_dict, param, exposure, lines, color_map)

# Plot histogram of the specified parameter for given exposure time and lines
# plot_param_histogram(data_dict, param, exposure, lines, color_map)
# plot_param_avg_std(data_dict, param, exposure, lines, color_map)

# # Plot combined histogram of the specified parameter for all exposure times and repeats
# plot_combined_histograms(data_dict, param, lines, color_map)
