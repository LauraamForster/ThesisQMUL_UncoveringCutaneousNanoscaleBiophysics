import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    plt.xlabel('X Position')
    plt.ylabel(param)
    plt.title(f'{param} for {exposure_time}s Exposure Time')
    plt.legend()
    plt.show()

def plot_param_histogram(data_dict, param, exposure_time, lines, color_map):

    
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
                    plt.hist(parameter.dropna().values, bins=30, edgecolor='black', alpha=0.5, color=color, label=f'Repeat {repeat} Line {line}')
    plt.xlabel(param)
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
    plt.ylim(60,70)
    plt.bar(labels, averages, yerr=std_devs, capsize=5, color=[color_map[(int(repeat), int(line))] for repeat in range(1, 4) for line in lines])
    plt.xlabel('Condition')
    plt.ylabel(param)
    plt.title(f'Average and Standard Deviation of {param} for {exposure_time}s Exposure Time')
    plt.xticks(rotation=45)
    plt.show()
    
    
# Folder path
folder = '/Users/lauraforster/Documents/Uni/3 - PhD/SAXS/DLS visits/May24/Radiation damage/'

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
    (1, 0): 'lightblue',
    (1, 1): 'darkblue',
    (2, 0): 'orange',
    (2, 1): 'red',
    (3, 0): 'lightgreen',
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
param = 'D_period_third'
# 'peak_width_third'
exposure = 0.5
lines = [0, 1]

# Plot specified parameter for given exposure time and lines
plot_param(data_dict, param, exposure, lines, color_map)

# Plot histogram of the specified parameter for given exposure time and lines
plot_param_histogram(data_dict, param, exposure, lines, color_map)
plot_param_avg_std(data_dict, param, exposure, lines, color_map)
