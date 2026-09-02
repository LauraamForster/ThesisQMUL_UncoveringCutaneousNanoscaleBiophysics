import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d
import pandas as pd

# Specify the base data directory
data_directory = '/Users/lauraforster/Documents/Uni/3 - PhD/Other Techniques/Nano-InXider/data analysis/datawaxs'

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

def smooth_data(y, sigma=0.7):
    return gaussian_filter1d(y, sigma=sigma)

# Function to process data and plot
def process_and_plot(data_directory):
    file_dict = {}
    
    # Find all .dat files in the directory
    dat_files = [f for f in sorted(os.listdir(data_directory)) if f.endswith('.dat')]

    # Group files by sample_name and fileno
    for dat_file in dat_files:
        sample_name, n, fileno = dat_file[:-4].split('_')
        if (sample_name, fileno) not in file_dict:
            file_dict[(sample_name, fileno)] = {}
        file_dict[(sample_name, fileno)][n] = dat_file

    # Loop through each group and process the files
    for (sample_name, fileno), files in file_dict.items():
        if '0' in files and '1' in files:  # Only process if both SAXS and WAXS files are present
            plt.figure(figsize=(10, 8))
            for n, dat_file in files.items():
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

                    # Smooth the data
                    I_q_transmitted_smoothed = smooth_data(I_q_transmitted)

                    # Plot the data
                    label = 'SAXS' if n == '0' else 'WAXS'
                    plt.plot(q, I_q_transmitted_smoothed, label=f'{label}: {dat_file}')
            
            plt.xlabel('q (nm⁻¹)')
            plt.ylabel('I(q)_transmitted')
            plt.xlim(0.01,20)
            plt.ylim(1e-11, 1e-8)
            plt.yscale('log')
            plt.title(f'{sample_name} - File {fileno}')
            plt.legend()
            plt.show()

process_and_plot(data_directory)











