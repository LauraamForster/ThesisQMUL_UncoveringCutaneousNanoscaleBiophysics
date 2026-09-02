import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import matplotlib.colorbar as cbar
import math
import sys

# Set the directory path
directorypath = '/Users/lauraforster/Documents/Uni/3 - PhD/SAXS/DLS visits/July23/csv/mia+other'
directory = os.chdir(directorypath)

# Read the two CSV files
df_776 = pd.read_csv('776 IQ_smooth.csv')
df_777 = pd.read_csv('777 IQ_smooth.csv')

df_776 = pd.read_csv('776 IQ_smooth.csv')
df_777 = pd.read_csv('777 IQ_smooth.csv')

# Extract X, Y, and intensity values
x_776 = df_776['x'].values
y_776 = df_776['y'].values
intensity_776 = df_776['area under third order curve'].values

x_777 = df_777['x'].values
y_777 = df_777['y'].values
intensity_777 = df_777['area under third order curve'].values

# Reshape intensity values into a 36 by 71 grid
grid_776 = intensity_776.reshape((71, 36))
grid_777 = intensity_777.reshape((71, 36))

grid_777_2 = grid_777[8:, :]
grid_776_2 = grid_776[:-8, :]

grid_777_2 = grid_777_2[:, 1:]
grid_776_2 = grid_776_2[:, :-1]

# Create updated X and Y coordinates for the new grid
x_776_2 = np.tile(np.arange(0, grid_776_2.shape[1]), grid_776_2.shape[0])
y_776_2 = np.repeat(np.arange(0, grid_776_2.shape[0]), grid_776_2.shape[1])
x_777_2 = np.tile(np.arange(0, grid_777_2.shape[1]), grid_777_2.shape[0])
y_777_2 = np.repeat(np.arange(0, grid_777_2.shape[0]), grid_777_2.shape[1])

# Flatten the updated grid_776_2
intensity_776_2 = grid_776_2.flatten()
intensity_777_2 = grid_777_2.flatten()

# Plot histogram for grid_776_2
plt.hist(intensity_777_2, bins=50, color='blue',  label='Grid 776_2')

# Plot histogram for grid_777_2 on the same plot
plt.hist(intensity_776_2, bins=50, color='orange', alpha=0.5, label='Grid 777_2')

# Set plot labels and title
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.xlim(0.001, 0.005)
plt.ylim(0, 150)
plt.title('Histogram of Intensity Values in Grids 776_2 and 777_2')
plt.legend()

# Show the plot
plt.show()
sys.exit()

# Create a DataFrame for the new grid_776_2
df_776_2 = pd.DataFrame({'x': x_776_2, 'y': y_776_2, 'intensity': intensity_776_2})
df_777_2 = pd.DataFrame({'x': x_777_2, 'y': y_777_2, 'intensity': intensity_777_2})

# Save the new DataFrame to a new CSV file
df_776_2.to_csv('776_2 IQ_smooth.csv', index=False)
df_777_2.to_csv('777_2 IQ_smooth.csv', index=False)

# Plot the original grid_776 using imshow
plt.imshow(grid_776_2, cmap='jet', origin='lower', aspect='auto')
plt.colorbar(label='Intensity')
plt.title('Original Grid 776')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# # Plot the original grid_776 using imshow
plt.imshow(grid_777_2, cmap='jet', origin='lower', aspect='auto')
plt.colorbar(label='Intensity')
plt.title('Original Grid 777')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()