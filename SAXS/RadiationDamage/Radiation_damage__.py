#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:29:50 2023

@author: lauraforster
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd


file_path = '/Users/lauraforster/Documents/Uni/3 - PhD/SAXS/DLS visits/July23/csv/mia+other'
directory = os.chdir(file_path)


def plot_peak_height(file_path, label):
    # Read the Excel file into a Pandas DataFrame
    df = pd.read_csv(file_path)  # Change to pd.read_csv if it's a CSV file

    # Extract peak height column (adjust the column name)
    # peak_height = df['area under third order curve']
    x_coords = df['x']
    y_coords = df['y']

    x_min = 15
    x_max = 16
    
    y_min = 25
    y_max = 30
    
    central_data = df[(x_coords >= x_min) & (x_coords <= x_max)]
    central_data = df[(y_coords >= y_min) & (y_coords <= y_max)]
    peak_height = central_data['area under third order curve']
    

    # Create a range of x-values (assuming one x-value per row)
    x_values = range(1, len(peak_height) + 1)

    # Plot peak height
    plt.plot(x_values, peak_height, label=label)

# List of file paths and corresponding labels
file_paths = ["778 IQ_smooth.csv", "779 IQ_smooth.csv"]
# file_paths = ["776 IQ_smooth.csv", "777 IQ_smooth.csv", "778 IQ_smooth.csv", "779 IQ_smooth.csv"]
labels = ["File 1", "File 2"]
# labels = ["File 1", "File 2", "File 3", "File 4"]

# Loop through the files and plot peak heights
for file_path, label in zip(file_paths, labels):
    plot_peak_height(file_path, label)


# Customize your plot

plt.xlabel('Data Point')
plt.ylabel('intensity')
plt.title('Peak Height Comparison')
plt.legend()
plt.grid()

# Show or save the plot
plt.show()  # To display the plot
# plt.savefig('peak_height_comparison.png')  # To save the plot to a file

