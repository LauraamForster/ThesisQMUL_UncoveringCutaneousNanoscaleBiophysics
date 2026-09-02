#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:50:44 2026

@author: lauraforster
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- input file ---
file_path = "/Users/lauraforster/Documents/Uni/3 - PhD/SAXS/DLS visits/July23/CSVs/692800 IQ_fitting.csv"

# Column used for thresholding
threshold_col = "total_SAXS_norm_0_1"

# Column actually plotted
plot_col = "collagen_third_norm_0_1"
# plot_col = "area under third order curve"
# plot_col = "total SAXS intensity"

threshold_pct = 25

df = pd.read_csv(file_path, sep=None, engine="python")
df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

cols = ["x", "y", threshold_col, plot_col]
df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

# Optional cropping
df = df[~df["y"].between(125, 200)]
df = df[~df["x"].between(0, 15)]

# df = df[~df["y"].between(145, 200)]
# df = df[~df["y"].between(0, 45)]
# df = df[~df["x"].between(55, 70)]

# Threshold based only on threshold_col
threshold = np.nanpercentile(df[threshold_col], threshold_pct)

# Keep only rows where threshold_col survives
df.loc[df[threshold_col] < threshold, plot_col] = np.nan

# Renormalise plotted column only on surviving points
vmin = df[plot_col].min(skipna=True)
vmax = df[plot_col].max(skipna=True)

if pd.notna(vmin) and pd.notna(vmax) and vmax > vmin:
    df[plot_col] = (df[plot_col] - vmin) / (vmax - vmin)
else:
    df[plot_col] = np.nan

# Make heatmap
heatmap_data = df.pivot(index="y", columns="x", values=plot_col)
rotated = np.rot90(heatmap_data.values, k=3)

plt.figure(figsize=(9, 4))
plt.imshow(
    rotated,
    origin="lower",
    aspect="auto",
    cmap="jet",
    extent=[
        heatmap_data.index.min(), heatmap_data.index.max(),
        heatmap_data.columns.max(), heatmap_data.columns.min()
    ],
    vmin=0,
    vmax=1
)
plt.colorbar(label=f"{plot_col} (thresholded by {threshold_col})")
plt.xlabel("y")
plt.ylabel("x")
plt.title(f"2D Heatmap of {plot_col}")
plt.show()