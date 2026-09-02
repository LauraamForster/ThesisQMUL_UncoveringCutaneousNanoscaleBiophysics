#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 12:23:48 2026

@author: lauraforster
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone SAXS heatmap with IChi orientation overlay.

Reads:
    692800 IQ_fitting.csv
    692800 IChi_fitting.csv

Plots:
    total SAXS intensity normalised 0-1
    fibril orientation overlay
    line length scaled by dominant IChi peak height
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# =============================================================================
# User settings
# =============================================================================

IQ_CSV = Path(
    "/Users/lauraforster/Documents/Uni/3 - PhD/SAXS/DLS visits/July23/CSVs/692798 IQ_fitting.csv"
)

ICHI_CSV = Path(
    "/Users/lauraforster/Documents/Uni/3 - PhD/SAXS/DLS visits/July23/CSVs/692798 IChi_fitting.csv"
)

SAVE_FIG = True
SAVE_PATH = IQ_CSV.parent / "692800_total_SAXS_with_IChi_orientation.png"

FONT = "Arial"

HEATMAP_COL = "total_SAXS_norm_0_1"
TOTAL_SAXS_COL = "total SAXS intensity"

# IChi orientation columns
PEAK1_POS_COL = "wm1_p1"
PEAK2_POS_COL = "wm1_p2"
PEAK1_HEIGHT_COL = "peak_height"
PEAK2_HEIGHT_COL = "peak_height2"

# Optional quality filtering
MIN_RSQ = None          # e.g. 0.3, or None to ignore
RSQ_COL = "rsq_gaussian_fit"

# Line scaling
MIN_LINE_LENGTH = 0.0
MAX_LINE_LENGTH = 4

# If your plotted SAXS orientation needs rotating by 90 degrees,
# change this to 90.
ANGLE_OFFSET_DEG = 0

LINE_COLOUR = "black"
LINE_WIDTH = 0.8

# # Crop settings
# CROP_X_MIN = 20
# CROP_X_MAX = 70
# CROP_Y_MIN = 0
# CROP_Y_MAX = 100

# Crop settings
CROP_X_MIN = 0
CROP_X_MAX = 40
CROP_Y_MIN = 0
CROP_Y_MAX = 60
# =============================================================================
# Helper functions
# =============================================================================

def read_csv_checked(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found:\n{path}")
    return pd.read_csv(path)


def normalise_0_1(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)

    norm = np.full_like(values, np.nan, dtype=float)

    if valid.sum() == 0:
        return norm

    vmin = np.nanmin(values[valid])
    vmax = np.nanmax(values[valid])

    if vmax > vmin:
        norm[valid] = (values[valid] - vmin) / (vmax - vmin)
    else:
        norm[valid] = 0.0

    return norm


def make_heatmap_array(df, value_col):
    x = df["x"].astype(int)
    y = df["y"].astype(int)

    min_x = x.min()
    max_x = x.max() + 1
    min_y = y.min()
    max_y = y.max() + 1

    image = np.full((max_y - min_y, max_x - min_x), np.nan)

    for xi, yi, value in zip(x, y, df[value_col]):
        image[yi - min_y, xi - min_x] = value

    extent = (min_x, max_x, min_y, max_y)

    return image, min_x, max_x, min_y, max_y, extent


def get_dominant_angle_and_height(row):
    p1 = row.get(PEAK1_POS_COL, np.nan)
    p2 = row.get(PEAK2_POS_COL, np.nan)

    h1 = row.get(PEAK1_HEIGHT_COL, np.nan)
    h2 = row.get(PEAK2_HEIGHT_COL, np.nan)

    p1 = pd.to_numeric(p1, errors="coerce")
    p2 = pd.to_numeric(p2, errors="coerce")
    h1 = pd.to_numeric(h1, errors="coerce")
    h2 = pd.to_numeric(h2, errors="coerce")

    if np.isfinite(h1) and np.isfinite(h2):
        if h1 >= h2:
            return p1, h1
        return p2, h2

    if np.isfinite(h1):
        return p1, h1

    if np.isfinite(h2):
        return p2, h2

    return np.nan, np.nan


def add_orientation_overlay(ax, ichi_df, min_x, max_x, min_y, max_y):
    required = {"x", "y", PEAK1_POS_COL, PEAK2_POS_COL, PEAK1_HEIGHT_COL, PEAK2_HEIGHT_COL}
    missing = [col for col in required if col not in ichi_df.columns]

    if missing:
        raise ValueError(
            f"IChi CSV is missing required columns:\n{missing}\n\n"
            f"Available columns are:\n{ichi_df.columns.tolist()}"
        )

    rows = []

    for _, row in ichi_df.iterrows():
        xi = int(row["x"])
        yi = int(row["y"])

        if not (min_x <= xi < max_x and min_y <= yi < max_y):
            continue

        if MIN_RSQ is not None and RSQ_COL in ichi_df.columns:
            rsq = pd.to_numeric(row.get(RSQ_COL, np.nan), errors="coerce")
            if not np.isfinite(rsq) or rsq < MIN_RSQ:
                continue

        angle_deg, peak_height = get_dominant_angle_and_height(row)

        if not np.isfinite(angle_deg) or not np.isfinite(peak_height):
            continue

        rows.append((xi, yi, angle_deg % 360, peak_height))

    if not rows:
        print("No valid IChi orientation rows found.")
        return

    peak_heights = np.array([r[3] for r in rows], dtype=float)
    peak_height_norm = normalise_0_1(peak_heights)

    lengths = MIN_LINE_LENGTH + peak_height_norm * (MAX_LINE_LENGTH - MIN_LINE_LENGTH)

    angles = np.deg2rad(np.array([r[2] for r in rows]) + ANGLE_OFFSET_DEG)

    # Same orientation convention as your previous script
    dx = lengths * np.sin(angles)
    dy = lengths * np.cos(angles)

    xs = np.array([r[0] for r in rows], dtype=float) + 0.5
    ys = np.array([r[1] for r in rows], dtype=float) + 0.5

    x0 = xs - dx / 2
    y0 = ys - dy / 2
    x1 = xs + dx / 2
    y1 = ys + dy / 2

    segments = np.stack(
        [np.column_stack([x0, y0]), np.column_stack([x1, y1])],
        axis=1,
    )

    line_collection = LineCollection(
        segments,
        colors=LINE_COLOUR,
        linewidths=LINE_WIDTH,
        zorder=10,
    )

    ax.add_collection(line_collection)

    print(f"Orientation lines plotted: {len(rows)}")
    print(f"Peak height min: {np.nanmin(peak_heights):.4g}")
    print(f"Peak height max: {np.nanmax(peak_heights):.4g}")
    print(f"Line length min: {np.nanmin(lengths):.4g}")
    print(f"Line length max: {np.nanmax(lengths):.4g}")


# =============================================================================
# Main script
# =============================================================================

plt.rcParams["font.family"] = FONT
plt.rcParams["font.size"] = 12

iq_df = read_csv_checked(IQ_CSV)
ichi_df = read_csv_checked(ICHI_CSV)

# =============================================================================
# Crop data
# =============================================================================

iq_df = iq_df[
    (iq_df["x"] >= CROP_X_MIN) &
    (iq_df["x"] <= CROP_X_MAX) &
    (iq_df["y"] >= CROP_Y_MIN) &
    (iq_df["y"] <= CROP_Y_MAX)
].copy()

ichi_df = ichi_df[
    (ichi_df["x"] >= CROP_X_MIN) &
    (ichi_df["x"] <= CROP_X_MAX) &
    (ichi_df["y"] >= CROP_Y_MIN) &
    (ichi_df["y"] <= CROP_Y_MAX)
].copy()

print(f"Cropped IQ rows:   {len(iq_df)}")
print(f"Cropped IChi rows: {len(ichi_df)}")

print(f"IQ rows:   {len(iq_df)}")
print(f"IChi rows: {len(ichi_df)}")

if not {"x", "y"}.issubset(iq_df.columns):
    raise ValueError("IQ CSV must contain 'x' and 'y' columns.")

if HEATMAP_COL not in iq_df.columns:
    if TOTAL_SAXS_COL not in iq_df.columns:
        raise ValueError(
            f"Could not find '{HEATMAP_COL}' or '{TOTAL_SAXS_COL}' in IQ CSV."
        )

    print(f"'{HEATMAP_COL}' not found. Normalising '{TOTAL_SAXS_COL}' manually.")
    iq_df[HEATMAP_COL] = normalise_0_1(iq_df[TOTAL_SAXS_COL])

# =============================================================================
# Build cropped heatmap image
# =============================================================================

image, min_x, max_x, min_y, max_y, extent = make_heatmap_array(iq_df, HEATMAP_COL)

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(
    image,
    origin="lower",
    cmap="jet",
    vmin=0,
    vmax=1,
    extent=extent,
    aspect="equal",
)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Normalised total SAXS intensity")

# Add IChi fibril orientation overlay
add_orientation_overlay(
    ax=ax,
    ichi_df=ichi_df,
    min_x=min_x,
    max_x=max_x,
    min_y=min_y,
    max_y=max_y,
)

# Force axis limits to crop range
ax.set_xlim(CROP_X_MIN, CROP_X_MAX)
ax.set_ylim(CROP_Y_MIN, CROP_Y_MAX)

# Clean ticks
ax.set_xticks(np.arange(CROP_X_MIN, CROP_X_MAX + 1, 10))
ax.set_yticks(np.arange(CROP_Y_MIN, CROP_Y_MAX + 1, 20))

ax.set_xlabel("x scan position")
ax.set_ylabel("y scan position")

plt.tight_layout()

if SAVE_FIG:
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved figure to:\n{SAVE_PATH}")

plt.show()