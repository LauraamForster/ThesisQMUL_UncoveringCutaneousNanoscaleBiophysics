import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import matplotlib.colorbar as cbar
import math

# Set the directory path
directorypath = '/Users/lauraforster/Documents/Uni/3 - PhD/SAXS/DLS visits/July23/csv/mia+other'
directory = os.chdir(directorypath)

# Read the two CSV files
df_776 = pd.read_csv('776 IQ_smooth.csv')
df_777 = pd.read_csv('777 IQ_smooth.csv')

# Size of the array in mm
size_x_mm = 0.7
size_y_mm = 1.4

# Offsets in mm
offset_x = 0
offset_y = 0.15


# Number of points in each direction
num_points_x = 36
num_points_y = 71

# Calculate the corresponding shifts in terms of points
shift_x = int((offset_x / size_x_mm) * num_points_x)
shift_y = int((offset_y / size_y_mm) * num_points_y)

# Apply shifts to df_777
df_777['x'] += shift_x
df_777['y'] += shift_y

# Apply shifts to df_776
df_776['x'] += shift_x
df_776['y'] += shift_y

# Extract common region based on adjusted coordinates
common_ix = np.intersect1d(df_776['x'], df_777['x'])
common_iy = np.intersect1d(df_776['y'], df_777['y'])


# Filter data for the common region
df_776_common = df_776[(df_776['x'].isin(common_ix)) & (df_776['y'].isin(common_iy))]
df_777_common = df_777[(df_777['x'].isin(common_ix)) & (df_777['y'].isin(common_iy))]

# Extract data for plotting
ix_common = np.array(df_776_common['x'])
iy_common = np.array(df_776_common['y'])
WhatPlot_776 = np.array(df_776_common['area under third order curve'])
WhatPlot_777 = np.array(df_777_common['area under third order curve'])
# WhatPlot_776 = np.array(df_776_common['D_period_third'])
# WhatPlot_777 = np.array(df_777_common['D_period_third'])
# WhatPlot_776 = np.array(df_776_common['total SAXS intensity'])
# WhatPlot_777 = np.array(df_777_common['total SAXS intensity'])


zmin,zmax  = 0, 0.005 #area_arr
# zmin,zmax  = 64, 65.5 #DPeriod
# zmin,zmax  = 0, 1600 #total area

WhatPlot_776[WhatPlot_776 < zmin] = zmin
WhatPlot_777[WhatPlot_777 > zmax] = zmax
WhatPlot_777[WhatPlot_777 < zmin] = zmin
WhatPlot_776[WhatPlot_776 > zmax] = zmax

# --------------funcs -----------------------------------------
def rectangleplot(x, y, z, zmin, zmax):
    dx, dy = 1, 1
    if z < zmin or z > zmax:
        return
    else:
        normal = plt.Normalize(zmin, zmax)  # change zmin zmax
        colors = plt.cm.jet(normal(z))
        currentAxis = plt.gca()
        currentAxis.add_patch(Rectangle((x - dx / 2, y - dy / 2), dx, dy, color=colors))

def render_2D_map(axs, x, y, z, ldata, zmin, zmax, tick_spac_x, tick_spac_y, ticklabel_size):
    axs.xaxis.set_major_locator(ticker.MultipleLocator(tick_spac_x))
    axs.yaxis.set_major_locator(ticker.MultipleLocator(tick_spac_y))
    axs.xaxis.set_tick_params(labelsize=ticklabel_size)
    axs.yaxis.set_tick_params(labelsize=ticklabel_size)
    for i in range(ldata):
        myx, myy, myz = x[i], y[i], z[i]
        if not math.isnan(myz):
            rectangleplot(myx, myy, myz, zmin, zmax)
    normal = plt.Normalize(zmin, zmax)  # change zmin zmax
    caxs, _ = cbar.make_axes(axs, shrink=0.53)
    cb2 = cbar.ColorbarBase(caxs, cmap=plt.cm.jet, norm=normal)
    cb2.ax.tick_params(labelsize=ticklabel_size)
    return

def set_size(w, h, ax=None):
    """ w, h: width, height in inches """
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)

# # --------------heatmap -----------------------------------------
fig1 = plt.figure()
ax = fig1.add_subplot(111)
tick_label_size = 8
tick_spacing_x, tick_spacing_y = 5, 5
width, height = 10.0, 12.5
set_size(width, height)
plt.axis([-1, len(common_ix), len(common_iy), -1])
fig1.set_figheight(6)
fig1.set_figwidth(5)
ax.set_aspect(1)

render_2D_map(ax, ix_common, iy_common, WhatPlot_776, len(WhatPlot_776), zmin, zmax,
              tick_spacing_x, tick_spacing_y, tick_label_size)

plt.show()


# Set the directory path
directorypath = '/Users/lauraforster/Documents/Uni/3 - PhD/SAXS/DLS visits/July23/csv/mia+other'
directory = os.chdir(directorypath)

# Read the two CSV files
df_776 = pd.read_csv('776 IQ_smooth.csv')
df_777 = pd.read_csv('777 IQ_smooth.csv')

# Size of the array in mm
size_x_mm = 0.7
size_y_mm = 1.4

# Offsets in mm
offset_x = 0
offset_y = -0.15


# Number of points in each direction
num_points_x = 36
num_points_y = 71

# Calculate the corresponding shifts in terms of points
shift_x = int((offset_x / size_x_mm) * num_points_x)
shift_y = int((offset_y / size_y_mm) * num_points_y)

# Apply shifts to df_777
df_777['x'] += shift_x
df_777['y'] += shift_y

# Apply shifts to df_776
df_776['x'] += shift_x
df_776['y'] += shift_y

# Extract common region based on adjusted coordinates
common_ix = np.intersect1d(df_776['x'], df_777['x'])
common_iy = np.intersect1d(df_776['y'], df_777['y'])


# Filter data for the common region
df_776_common = df_776[(df_776['x'].isin(common_ix)) & (df_776['y'].isin(common_iy))]
df_777_common = df_777[(df_777['x'].isin(common_ix)) & (df_777['y'].isin(common_iy))]

# Extract data for plotting
ix_common = np.array(df_776_common['x'])
iy_common = np.array(df_776_common['y'])
WhatPlot_776 = np.array(df_776_common['area under third order curve'])
WhatPlot_777 = np.array(df_777_common['area under third order curve'])
# WhatPlot_776 = np.array(df_776_common['D_period_third'])
# WhatPlot_777 = np.array(df_777_common['D_period_third'])
# WhatPlot_776 = np.array(df_776_common['total SAXS intensity'])
# WhatPlot_777 = np.array(df_777_common['total SAXS intensity'])


zmin,zmax  = 0, 0.005 #area_arr
# zmin,zmax  = 64, 65.5 #DPeriod
# zmin,zmax  = 0, 1600 #total area

WhatPlot_776[WhatPlot_776 < zmin] = zmin
WhatPlot_777[WhatPlot_777 > zmax] = zmax
WhatPlot_777[WhatPlot_777 < zmin] = zmin
WhatPlot_776[WhatPlot_776 > zmax] = zmax

# # --------------heatmap -----------------------------------------
fig1 = plt.figure()
ax = fig1.add_subplot(111)
tick_label_size = 8
tick_spacing_x, tick_spacing_y = 5, 5
width, height = 10.0, 12.5
set_size(width, height)
plt.axis([-1, len(common_ix), len(common_iy), -1])
fig1.set_figheight(6)
fig1.set_figwidth(5)
ax.set_aspect(1)

# render_2D_map(ax, ix_common, iy_common, WhatPlot_776, len(WhatPlot_776), zmin, zmax,
#               tick_spacing_x, tick_spacing_y, tick_label_size)

# plt.show()


render_2D_map(ax, ix_common, iy_common, WhatPlot_777, len(WhatPlot_777), zmin, zmax,
              tick_spacing_x, tick_spacing_y, tick_label_size)

plt.show()