import numpy as np
import matplotlib.pyplot as plt
import os
from lmfit.models import ExponentialModel, ConstantModel

# Specify the base data directory
base_data_directory = '/Users/lauraforster/Documents/Uni/3 - PhD/Other Techniques/Nano-InXider/data/realData/Comparison of Settings'
# parnika - change base_data_directory to where you save the data folder

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
def extract_sumforint(header_lines):
    for line in header_lines:
        if line.startswith('# SumForIntensity'):
            return float(line.split()[2])
    return 1.0  # Default value if Intensity1 is not found


def PlotSettings(listtoplot, xlim, ylim):
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlim(xlim)
    plt.axvline(x=0.29, color='r', linestyle=':')
    plt.axvline(x=0.48, color='r', linestyle=':')
    plt.axvline(x=0.67, color='g', linestyle=':')
    plt.axvline(x=1.55, color='black', linestyle=':')
    plt.yscale('log')
    plt.xlabel('q (nm⁻¹)')
    plt.ylabel('I(q)_transmitted')
    plt.title(f'SettingConst: {listtoplot}')
    plt.title('Hello Parnika')
    plt.legend()
    return

# Function to process BG data
def BG(listtoplot, xlim, ylim):
    for folder in listtoplot:
        process_and_plot(folder, 'BG', listtoplot, xlim, ylim)

# Function to process S1 data
def S1(listtoplot, xlim, ylim):
    for folder in listtoplot:
        process_and_plot(folder, 'S1', listtoplot, xlim, ylim)

# Function to process S2 data
def S2(listtoplot, xlim, ylim):
    for folder in listtoplot:
        process_and_plot(folder, 'S2', listtoplot, xlim, ylim)
        
# Function to process S2 data
def S3(listtoplot, xlim, ylim):
    for folder in listtoplot:
        process_and_plot(folder, 'S3', listtoplot, xlim, ylim)
        
# Function to process data and plot
def process_and_plot(folder_name, subfolder_name, listtoplot, xlim, ylim):
    resolution_dict = {}
    data_directory = os.path.join(base_data_directory, folder_name, subfolder_name)
    
    # Find all .dat files in the directory
    dat_files = [f for f in os.listdir(data_directory) if f.endswith('.dat')]

    # Loop through each .dat file and process it
    for dat_file in dat_files:
        file_path = os.path.join(data_directory, dat_file)
        data, header = read_data(file_path)
        print(dat_file)
        
        if data is not None:
            intensity1 = extract_intensity1(header)
            sumforint = extract_sumforint(header)
            
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
            print(intensity1, sumforint)
            I_q_transmitted = I_q * sumforint
            # I_q_transmitted = I_q 
            
            # Parse the file name to get sample name, resolution, and exposure time
            sample_name, resolution, exposure_time = dat_file[:-4].split('_')
            
            if resolution not in resolution_dict:
                resolution_dict[resolution] = {}
            if exposure_time not in resolution_dict[resolution]:
                resolution_dict[resolution][exposure_time] = []

            resolution_dict[resolution][exposure_time].append((q, I_q_transmitted, dat_file))

    # Plot the data
    for resolution, exposure_dict in resolution_dict.items():
        # plt.figure(figsize=(10, 8))
        for exposure_time, data_list in exposure_dict.items():
            for q, I_q_transmitted, dat_file in data_list:
                plt.plot(q, I_q_transmitted, label=f'{dat_file}')
                
                # PlotSettings(listtoplot, xlim, ylim)
                # plt.show()
            
listtoplot=['100s'] #parnika - you can change this to change what is being plotted (options are below)
# ['100s', '300s', '450s', '600s', 'HR', 'MR', 'VHR']:

# ylim = (10e-12, 10e-8)#scaled  
# ylim = (10e-3, 10e-0)#unscaled
ylim=(10e6, 1e12) #parnika - you can change this to change the y scale
xlim=(0.1,0.7) #parnika - you can change this to change the x scale (but you shouldnt need to) 

# below is a list of samples and background
# BG(listtoplot, xlim, ylim)
S1(listtoplot, xlim, ylim)
# S2(listtoplot, xlim, ylim)
# S3(listtoplot, xlim, ylim)

PlotSettings(listtoplot, xlim, ylim)
plt.show()


