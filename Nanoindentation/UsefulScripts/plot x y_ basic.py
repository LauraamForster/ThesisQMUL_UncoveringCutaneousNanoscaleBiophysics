#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 23:32:04 2024

@author: lauraforster
"""

import matplotlib.pyplot as plt
import numpy as np

# Define the file path
file_path = '/Users/lauraforster/Desktop/xyplot/data5.txt'

# Function to check if a string can be converted to a float
def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Read the data from the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Remove the header lines
lines = lines[2:]

# Check if the file has one or two columns
if '\t' in lines[0]:
    # Two columns case
    y = []
    for line in lines:
        values = line.strip().split('\t')
        for value in values:
            if value.lower() == 'nan' or not is_float(value):
                y.append(np.nan)
            else:
                y.append(float(value))
else:
    # One column case
    y = []
    for line in lines:
        value = line.strip()
        if value.lower() == 'nan' or not is_float(value):
            y.append(np.nan)
        else:
            y.append(float(value))

# Replace values over 100000 and negative values with np.nan
y = [np.nan if val > 100000 or val < 0 else val for val in y]

# Reverse the y values
y = y[::-1]

# Convert y values from Pa to kPa
y = [val / 1000 if val is not np.nan else val for val in y]

# Generate x values as the position indices
x = list(range(len(y)))

# Create the plot
plt.plot(x, y, linestyle='-', color='b')
plt.title("Effective Young's Modulus over $x$ Position")
plt.xlabel('$x$ Position')
plt.ylabel('Eff Young\'s Modulus (kPa)')
plt.show()
