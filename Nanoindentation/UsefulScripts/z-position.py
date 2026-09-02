#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:28:44 2023

@author: lauraforster
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

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

# Define the folder where your files are located
folder_path = '/Users/lauraforster/Documents/Uni/3 - PhD/Nanoindentation/NI_Data/Data/Paper/Thicknesses/60um 1/matrix_scan01'
# folder_path = '/Users/lauraforster/Documents/Uni/3 - PhD/Nanoindentation/NI_Data/Data/Paper/Protocol Testing/line scan epidermis 2'
file_list = os.listdir(folder_path)
sorted_file_list = sorted(file_list, key=extract_xy_key)
# Initialize lists to store data
x_cord, y_cord, x_poss, y_poss, z_poss, z_surf, piez_surf = [], [],[],[],[],[],[]


# Loop through each .txt file in the folder
for file_name in sorted_file_list:
    if file_name.endswith('.txt') and "position" not in file_name:
        file_path = os.path.join(folder_path, file_name)
        
        # Read the file into a DataFrame, skipping rows until the Z-position line is found
        with open(file_path, 'r') as f:
            for line in f:
                lines = f.readlines()
                
                x_cord.append(int(lines[1].split()[5]))
                y_cord.append(int(lines[1].split()[8]))
                x_poss.append(float(lines[3].split()[2]))
                y_poss.append(float(lines[4].split()[2]))
                z_poss.append(float(lines[5].split()[2]))
                z_surf.append(float(lines[6].split()[3]))
                piez_surf.append(float(lines[7].split()[4]))
                
 # Create a common x-axis from 0 to the maximum row length

where, where2  = 2, 3
specific_cols = [1]
specific_rows = [1,2,3,4,5,6,7,8, 9]


# PLOT STRAIGHT LINES
xline, xline2, yline = np.linspace(where,where, 100), np.linspace(where2,where2, 100), np.linspace(min(z_poss),max(z_poss),100)
# plt.plot(xline,yline, 'blue')
# plt.plot(xline2,yline, 'blue')               
                
# Plot SPECIFIC COLUMNS
unique_x_coords = list(set(x_cord))
num_cols = len(specific_cols)
max_col_length = max([x_cord.count(x) for x in unique_x_coords])
x_axis = np.arange(max_col_length)
col_data = np.zeros((num_cols, max_col_length))

for i, x_coord in enumerate(unique_x_coords):
    if x_coord in specific_cols:
        cols = [j for j, x in enumerate(x_cord) if x == x_coord]
        z_values = [z_poss[j] for j in cols]
        col_data[specific_cols.index(x_coord), :len(z_values)] = z_values

for i in range(num_cols):
    plt.plot(x_axis, col_data[i, :], marker='o', label=f'Col {specific_cols[i]}')
# plt.legend()
# plt.ylim(1450,1650)
plt.title("Columns")
plt.show()

# # Plot SPECIFIC ROWS
# unique_y_coords = list(set(y_cord))
# num_rows = len(specific_rows)
# max_row_length = max([y_cord.count(y) for y in unique_y_coords])
# y_axis = np.arange(max_row_length)
# row_data = np.zeros((num_rows, max_row_length))

# for i, y_coord in enumerate(unique_y_coords):
#     if y_coord in specific_rows:
#         rows = [j for j, y in enumerate(y_cord) if y == y_coord]
#         z_values = [z_poss[j] for j in rows]
#         row_data[specific_rows.index(y_coord), :len(z_values)] = z_values

# for i in range(num_rows):
#     plt.plot(y_axis, row_data[i, :], marker='o', label=f'Row {specific_rows[i]}')
# plt.legend()
# # plt.ylim(1450,1650)
# plt.title("Rows")
# plt.show()


# Horizontal or Vertical Lines
# upsidedown= []
# which = x_cord
# maxx = np.max(which)
# for x in range(len(z_poss)):
#     upsidedown.append(maxx - z_poss[x])

# leng = len(which)
# points = (int(leng/2))+1
# plt.plot(which, upsidedown)
# plt.scatter(which, z_poss)
# plt.xticks(np.linspace(1,leng,points))
# plt.show()
       

# All the ROWS:

# Find unique X-coordinates
# unique_x_coords = list(set(x_cord))
# # Plot each unique X-coordinate as a row
# for x_coord in unique_x_coords:
#     rows = [i for i, x in enumerate(x_cord) if x == x_coord]
#     z_values = [z_poss[i] for i in rows]
#     plt.plot(rows, z_values, marker='o', label=f'Z Values for Row {x_coord}')
#     plt.legend()
#     plt.show()


# # All the COLUMNS:

# # Find unique Y-coordinates
# unique_y_coords = list(set(y_cord))

# # Plot each unique Y-coordinate as a column
# for y_coord in unique_y_coords:
#     columns = [i for i, y in enumerate(y_cord) if y == y_coord]
#     z_values = [z_poss[i] for i in columns]
#     plt.plot(columns, z_values, marker='o', label=f'Z Values for Column {y_coord}')
#     plt.legend()
#     plt.show()