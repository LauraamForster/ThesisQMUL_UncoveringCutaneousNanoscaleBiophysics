#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 11:58:07 2025

@author: lauraforster
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Force an interactive GUI backend (pick the first one that works)
import matplotlib
for candidate in ("QtAgg", "Qt5Agg", "TkAgg", "MacOSX"):
    try:
        matplotlib.use(candidate, force=True)
        break
    except Exception:
        pass

import matplotlib.pyplot as plt
from matplotlib.path import Path


# ----------------------------
# Angular overlay (your helper)
# ----------------------------
def overlay_orientation_arrows(
    Filenumber, Output_directoryCSV, ax, max_x, max_y,
    min_rsq=0.2,
    min_len=0.7,
    max_len=2.0
):
    ichi_csv = os.path.join(Output_directoryCSV, f"{Filenumber} IChi_fitting.csv")
    if not os.path.exists(ichi_csv):
        print(f"[AngularOverlay] IChi file not found: {ichi_csv}")
        return

    df = pd.read_csv(ichi_csv)
    if len(df) < 5:
        print(f"[AngularOverlay] IChi file has fewer than 5 rows. Skipping overlay.")
        return

    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    col_p1 = pick('peak_position', 'peak_position_first')
    col_p2 = pick('peak_position2', 'peak_position_second')
    col_h1 = pick('peak_height', 'peak_height_first')
    col_h2 = pick('peak_height2', 'peak_height_second')
    if col_p1 is None and col_p2 is None:
        print("[AngularOverlay] No peak position columns found.")
        return

    rows = []
    for _, r in df.iterrows():
        xi, yi = int(r['x']), int(r['y'])
        if not (0 <= xi < max_x and 0 <= yi < max_y):
            continue

        a1 = r[col_p1] if col_p1 and pd.notna(r.get(col_p1, np.nan)) else np.nan
        a2 = r[col_p2] if col_p2 and pd.notna(r.get(col_p2, np.nan)) else np.nan
        if np.isnan(a1) and np.isnan(a2):
            continue

        if np.isnan(a1):
            angle_deg, which = float(a2) % 360.0, 2
        elif np.isnan(a2):
            angle_deg, which = float(a1) % 360.0, 1
        else:
            angle_deg, which = (float(a1), 1) if float(a1) <= float(a2) else (float(a2), 2)
            angle_deg %= 360.0

        h = np.nan
        if which == 1 and col_h1:
            h = r.get(col_h1, np.nan)
            if pd.isna(h) and col_h2:
                h = r.get(col_h2, np.nan)
        elif which == 2 and col_h2:
            h = r.get(col_h2, np.nan)
            if pd.isna(h) and col_h1:
                h = r.get(col_h1, np.nan)
        if pd.isna(h):
            continue

        rsq = r.get("rsq_gaussian_fit", 1.0)
        if pd.isna(rsq) or rsq < min_rsq:
            continue

        rows.append((xi, yi, angle_deg, float(h)))

    if not rows:
        print("[AngularOverlay] No valid orientation/height data to display.")
        return

    heights = np.array([h for *_, h in rows], float)
    hmin, hmax = np.nanmin(heights), np.nanmax(heights)
    if hmax > hmin:
        hnorm = (heights - hmin) / (hmax - hmin)
    else:
        hnorm = np.full_like(heights, 0.5)
    lengths = min_len + hnorm * (max_len - min_len)

    angles = np.deg2rad([ang for _, _, ang, _ in rows])
    dx = lengths * np.sin(angles)
    dy = lengths * np.cos(angles)

    xs = np.array([x for x, *_ in rows], float) + 0.5
    ys = np.array([y for _, y, *_ in rows], float) + 0.5
    x0 = xs - dx / 2.0
    y0 = ys - dy / 2.0
    x1 = xs + dx / 2.0
    y1 = ys + dy / 2.0

    segments = np.stack(
        [np.column_stack([x0, y0]), np.column_stack([x1, y1])],
        axis=1
    )

    lc = LineCollection(segments, colors='white', linewidths=0.8, zorder=10)
    ax.add_collection(lc)

    print(
        f"[AngularOverlay] Plotted {len(rows)} orientation lines "
        f"(len ∈ [{lengths.min():.2f},{lengths.max():.2f}], "
        f"h ∈ [{hmin:.3g},{hmax:.3g}])."
    )

# ----------------------------
# Save raw 3-panel heatmap
# ----------------------------
def save_raw_heatmaps_png(
    Filenumber,
    images,
    out_dir,
    cmap='jet'
):
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{Filenumber}_raw.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    titles = {
        'curvearea_norm': 'curvearea_norm (collagen_third_norm_0_1)',
        'SAXS_norm': 'SAXS_norm (total_SAXS_norm_0_1)',
        'D_period': 'D_period (D_period_third)'
    }
    labels = {
        'curvearea_norm': 'collagen_third_norm_0_1',
        'SAXS_norm': 'total_SAXS_norm_0_1',
        'D_period': 'D_period'
    }

    for ax, key in zip(axes, ['curvearea_norm', 'SAXS_norm', 'D_period']):
        img, (zmin, zmax), _, _ = images[key]
        im = ax.imshow(
            img,
            origin='lower',
            cmap=cmap,
            vmin=zmin,
            vmax=zmax,
            extent=(0, img.shape[1], 0, img.shape[0])
        )
        ax.set_title(titles[key])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=labels[key])

    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved raw heatmap PNG to:\n{out_png}")

def save_raw_heatmaps_with_overlay_png(
    Filenumber,
    Output_directoryCSV,
    images,
    out_dir,
    cmap='jet'
):
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{Filenumber}_raw_with_overlay.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    titles = {
        'curvearea_norm': 'curvearea_norm (collagen_third_norm_0_1)',
        'SAXS_norm': 'SAXS_norm (total_SAXS_norm_0_1)',
        'D_period': 'D_period (D_period_third)'
    }
    labels = {
        'curvearea_norm': 'collagen_third_norm_0_1',
        'SAXS_norm': 'total_SAXS_norm_0_1',
        'D_period': 'D_period'
    }

    max_x_all = max(images[k][2] for k in images)
    max_y_all = max(images[k][3] for k in images)

    for ax, key in zip(axes, ['curvearea_norm', 'SAXS_norm', 'D_period']):
        img, (zmin, zmax), _, _ = images[key]
        im = ax.imshow(
            img,
            origin='lower',
            cmap=cmap,
            vmin=zmin,
            vmax=zmax,
            extent=(0, img.shape[1], 0, img.shape[0])
        )
        ax.set_title(titles[key])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=labels[key])

        overlay_orientation_arrows(
            Filenumber=Filenumber,
            Output_directoryCSV=Output_directoryCSV,
            ax=ax,
            max_x=max_x_all,
            max_y=max_y_all
        )

    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved raw heatmap PNG with angular overlay to:\n{out_png}")
# ----------------------------
# Simple wound/control selector
# ----------------------------
def choose_sample_type():
    """
    Returns 'wound' or 'control'.
    Uses a small tkinter popup with mutually exclusive radio buttons.
    """
    import tkinter as tk
    from tkinter import messagebox

    result = {"value": None}

    root = tk.Tk()
    root.title("Sample type")
    root.geometry("260x140")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    var = tk.StringVar(value="")

    tk.Label(root, text="Select sample type:", font=("Arial", 11)).pack(pady=(12, 8))
    tk.Radiobutton(root, text="Wound", variable=var, value="wound").pack(anchor="w", padx=30)
    tk.Radiobutton(root, text="Control", variable=var, value="control").pack(anchor="w", padx=30)

    def confirm():
        val = var.get()
        if val not in {"wound", "control"}:
            messagebox.showwarning("Selection required", "Please select either Wound or Control.")
            return
        result["value"] = val
        root.destroy()

    def cancel():
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=12)

    tk.Button(btn_frame, text="OK", width=10, command=confirm).pack(side="left", padx=5)
    tk.Button(btn_frame, text="Cancel", width=10, command=cancel).pack(side="left", padx=5)

    root.mainloop()

    if result["value"] is None:
        raise RuntimeError("Sample type selection cancelled.")

    print(f"\nSelected sample type: {result['value']}")
    return result["value"]

def choose_mode():
    """
    Returns 'define_regions' or 'orient_sample'.
    """
    import tkinter as tk
    from tkinter import messagebox

    result = {"value": None}

    root = tk.Tk()
    root.title("ROI tool mode")
    root.geometry("300x150")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    var = tk.StringVar(value="")

    tk.Label(root, text="Choose mode:", font=("Arial", 11)).pack(pady=(12, 8))
    tk.Radiobutton(root, text="Define regions", variable=var, value="define_regions").pack(anchor="w", padx=30)
    tk.Radiobutton(root, text="Orient sample", variable=var, value="orient_sample").pack(anchor="w", padx=30)

    def confirm():
        val = var.get()
        if val not in {"define_regions", "orient_sample"}:
            messagebox.showwarning("Selection required", "Please select a mode.")
            return
        result["value"] = val
        root.destroy()

    def cancel():
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=12)

    tk.Button(btn_frame, text="OK", width=10, command=confirm).pack(side="left", padx=5)
    tk.Button(btn_frame, text="Cancel", width=10, command=cancel).pack(side="left", padx=5)

    root.mainloop()

    if result["value"] is None:
        raise RuntimeError("Mode selection cancelled.")

    print(f"\nSelected mode: {result['value']}")
    return result["value"]
# -----------------------------------
# Heatmap builder & interactive click
# -----------------------------------
def load_saved_regions_from_excel(excel_path, Experiment, Filenumber):
    """
    Returns:
        sample_type, regions
    where regions is a dict with keys sample/dermis/wound and list of (x,y).
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"ROI Excel file not found:\n{excel_path}")

    df = pd.read_excel(excel_path, sheet_name="ROIs")
    df["experiment"] = df["experiment"].astype(str)
    df["Filenumber"] = df["Filenumber"].astype(str)

    row = df[
        (df["experiment"] == str(Experiment)) &
        (df["Filenumber"] == str(Filenumber))
    ]

    if row.empty:
        raise ValueError(f"No saved ROI row found for {Filenumber} / {Experiment}")

    row = row.iloc[0]
    sample_type = str(row.get("sample_type", "wound"))

    all_region_specs = [
        ("sample", 8),
        ("dermis", 8),
        ("wound", 8),
    ]

    regions = {}
    for name, npts in all_region_specs:
        pts = []
        for i in range(1, npts + 1):
            xcol = f"{name}_p{i}_x"
            ycol = f"{name}_p{i}_y"
            x = row.get(xcol, np.nan)
            y = row.get(ycol, np.nan)
            if pd.notna(x) and pd.notna(y):
                pts.append((float(x), float(y)))
        regions[name] = pts

    return sample_type, regions

def update_orientation_in_excel(excel_path, Experiment, Filenumber, epidermis_pt, subcutaneous_pt):
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"ROI Excel file not found:\n{excel_path}")

    df = pd.read_excel(excel_path, sheet_name="ROIs")
    df["experiment"] = df["experiment"].astype(str)
    df["Filenumber"] = df["Filenumber"].astype(str)

    mask = (
        (df["experiment"] == str(Experiment)) &
        (df["Filenumber"] == str(Filenumber))
    )

    if not mask.any():
        raise ValueError(f"No matching ROI row found for {Filenumber} / {Experiment}")

    df.loc[mask, "epidermis_x"] = epidermis_pt[0]
    df.loc[mask, "epidermis_y"] = epidermis_pt[1]
    df.loc[mask, "subcutaneous_x"] = subcutaneous_pt[0]
    df.loc[mask, "subcutaneous_y"] = subcutaneous_pt[1]

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="ROIs", index=False)

    print(f"\nUpdated orientation points in:\n{excel_path}")

def orient_sample_from_existing_regions(
    Filenumber,
    Output_directoryCSV,
    Experiment,
    images,
    excel_out_path,
    show_angular_overlay=True,
    cmap='jet'
):
    sample_type, regions = load_saved_regions_from_excel(excel_out_path, Experiment, Filenumber)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    titles = {
        'curvearea_norm': 'curvearea_norm (collagen_third_norm_0_1)',
        'SAXS_norm': 'SAXS_norm (total_SAXS_norm_0_1)',
        'D_period': 'D_period'
    }

    max_x_all = max(images[k][2] for k in images)
    max_y_all = max(images[k][3] for k in images)

    for ax, key in zip(axes, ['curvearea_norm', 'SAXS_norm', 'D_period']):
        img, (zmin, zmax), _, _ = images[key]
        im = ax.imshow(
            img,
            origin='lower',
            cmap=cmap,
            vmin=zmin,
            vmax=zmax,
            extent=(0, img.shape[1], 0, img.shape[0])
        )
        ax.set_title(titles[key])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(False)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if show_angular_overlay:
        for ax in axes:
            overlay_orientation_arrows(Filenumber, Output_directoryCSV, ax, max_x_all, max_y_all)

    draw_saved_regions_on_axes(axes, regions)

    points = {
        "epidermis": None,
        "subcutaneous": None
    }
    artists = {
        "epidermis": [],
        "subcutaneous": []
    }

    def draw_marker_all(name, x, y):
        marker = 'x' if name == "epidermis" else 'o'
        color = 'magenta' if name == "epidermis" else 'cyan'
        size = 90
        for ax in axes:
            sc = ax.scatter([x], [y], marker=marker, s=size, color=color, zorder=25)
            artists[name].append(sc)

    def clear_marker(name):
        for h in artists[name]:
            try:
                h.remove()
            except Exception:
                pass
        artists[name].clear()
        points[name] = None

    def on_click(event):
        if event.inaxes not in axes or event.xdata is None or event.ydata is None:
            return

        x, y = float(event.xdata), float(event.ydata)

        if points["epidermis"] is None:
            clear_marker("epidermis")
            points["epidermis"] = (x, y)
            draw_marker_all("epidermis", x, y)
            print(f"Epidermis point: x={x:.2f}, y={y:.2f}")
        elif points["subcutaneous"] is None:
            clear_marker("subcutaneous")
            points["subcutaneous"] = (x, y)
            draw_marker_all("subcutaneous", x, y)
            print(f"Subcutaneous point: x={x:.2f}, y={y:.2f}")
            print("\nBoth orientation points selected. Press q to save, or e/s to reset one.")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'e':
            clear_marker("epidermis")
            print("Reset epidermis point.")
        elif event.key == 's':
            clear_marker("subcutaneous")
            print("Reset subcutaneous point.")
        elif event.key == 'q':
            if points["epidermis"] is None or points["subcutaneous"] is None:
                print("Please select both epidermis and subcutaneous points before saving.")
                return
            update_orientation_in_excel(
                excel_path=excel_out_path,
                Experiment=Experiment,
                Filenumber=Filenumber,
                epidermis_pt=points["epidermis"],
                subcutaneous_pt=points["subcutaneous"]
            )
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    print("\nOrientation mode:")
    print("  1st click = epidermis (x)")
    print("  2nd click = subcutaneous (o)")
    print("  e = reset epidermis | s = reset subcutaneous | q = save and close")

    plt.show()

def _build_image_from_df(df, value_col, default_z):
    """Return (image, zlims, max_x, max_y) from df selection."""
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in CSV.")

    if df.empty:
        raise ValueError(f"No points left after thresholding for '{value_col}'")

    x = df['x'].astype(int)
    y = df['y'].astype(int)
    v = df[value_col]

    max_x = x.max() + 1
    max_y = y.max() + 1

    image = np.full((max_y, max_x), np.nan)
    for xi, yi, vi in zip(x, y, v):
        image[yi, xi] = vi

    if default_z is None:
        v_clean = v[np.isfinite(v)]
        if len(v_clean) == 0:
            zmin, zmax = 0, 1
        else:
            mean = np.mean(v_clean)
            std = np.std(v_clean)
            if std == 0:
                zmin = mean - 1e-6
                zmax = mean + 1e-6
            else:
                zmin = mean - 3 * std
                zmax = mean + 3 * std
    else:
        zmin, zmax = default_z

    return image, (zmin, zmax), max_x, max_y


def heatmaps_with_rois(
    Filenumber,
    Output_directoryCSV,
    Experiment,
    show_angular_overlay=True,
    thresholds=None,
    zlims=None,
    cmap='jet',
    excel_out_path=None
):
    """
    3 heatmaps + ROI picking.

    Mode popup asks whether to:
      - define_regions
      - orient_sample

    If define_regions:
      Popup asks whether sample is:
        - control -> sample (8), dermis (8)
        - wound   -> sample (8), dermis (8), wound (8)

      Exports all vertices to an Excel file (one row per Filenumber/Experiment).
      Missing wound points for control are written as NaN.

    If orient_sample:
      Loads previously saved regions from Excel, redraws them,
      then lets you click:
        - epidermis point (x)
        - subcutaneous point (o)
      and updates the same Excel row with those extra columns.
    """
    print(f"\nBeginning triple heatmap + ROI picking for File {Filenumber} ({Experiment})...")

    mode = choose_mode()
    sample_type = None

    iq_path = os.path.join(Output_directoryCSV, f"{Filenumber} IQ_fitting.csv")
    if not os.path.isfile(iq_path):
        print(f"No IQ CSV found at: {iq_path}")
        return

    df = pd.read_csv(iq_path)
    if len(df) <= 5:
        print(f"IQ CSV found but has insufficient data ({len(df)} rows).")
        return
    if not {'x', 'y'}.issubset(df.columns):
        print("IQ CSV missing required 'x' or 'y' columns.")
        return

    if thresholds is None:
        thresholds = {
            'curvearea_norm': 0.0,
            'SAXS_norm': 0.0,
            'D_period': 64.6
        }
    if zlims is None:
        zlims = {
            'curvearea_norm': (0.0, 1.0),
            'SAXS_norm': (0.0, 1.0),
            'D_period': (64.6, 67.0)
        }

    colmap = {
        'curvearea_norm': 'collagen_third_norm_0_1',
        'SAXS_norm': 'total_SAXS_norm_0_1',
        'D_period': 'D_period'
    }

    c_curve = colmap["curvearea_norm"]
    c_saxs = colmap["SAXS_norm"]
    c_dp = colmap["D_period"]

    df_saxs = df[np.isfinite(df[c_saxs])].copy()

    shared = df.copy()
    shared = shared[
        np.isfinite(shared[c_curve]) &
        np.isfinite(shared[c_saxs]) &
        np.isfinite(shared[c_dp])
    ]
    shared = shared[
        (shared[c_curve] > thresholds.get("curvearea_norm", 0.0)) &
        (shared[c_saxs] > thresholds.get("SAXS_norm", 0.0)) &
        (shared[c_dp] > thresholds.get("D_period", -np.inf))
    ]

    if shared.empty:
        raise ValueError("No points survive the shared threshold mask (curvearea+saxs+D_period thresholds).")

    images = {}
    max_x_all, max_y_all = 0, 0

    for key in ["curvearea_norm", "D_period"]:
        img, zpair, mx, my = _build_image_from_df(shared, colmap[key], zlims.get(key, (0, 1)))
        images[key] = (img, zpair, mx, my)
        max_x_all = max(max_x_all, mx)
        max_y_all = max(max_y_all, my)

    img, zpair, mx, my = _build_image_from_df(
        df_saxs,
        colmap["SAXS_norm"],
        zlims.get("SAXS_norm", (0, 1))
    )
    images["SAXS_norm"] = (img, zpair, mx, my)
    max_x_all = max(max_x_all, mx)
    max_y_all = max(max_y_all, my)

    excel_master_path = (
        excel_out_path
        if excel_out_path is not None
        else "/Volumes/LauraDrive/SAXS/Presentations and notes/Manifests/ROI_points_simplified.xlsx"
    )

    if mode == "orient_sample":
        orient_sample_from_existing_regions(
            Filenumber=Filenumber,
            Output_directoryCSV=Output_directoryCSV,
            Experiment=Experiment,
            images=images,
            excel_out_path=excel_master_path,
            show_angular_overlay=show_angular_overlay,
            cmap=cmap
        )
        return

    sample_type = choose_sample_type()

    # ---------------------------------
    # Save raw 3-panel image FIRST
    # ---------------------------------
    raw_out_dir = "/Volumes/LauraDrive/SAXS/Presentations and notes/Images/DefineRegions/RAW"
    save_raw_heatmaps_png(
        Filenumber=str(Filenumber),
        images=images,
        out_dir=raw_out_dir,
        cmap=cmap
    )

    save_raw_heatmaps_with_overlay_png(
        Filenumber=str(Filenumber),
        Output_directoryCSV=Output_directoryCSV,
        images=images,
        out_dir=raw_out_dir,
        cmap=cmap
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    titles = {
        'curvearea_norm': 'curvearea_norm (collagen_third_norm_0_1)',
        'SAXS_norm': 'SAXS_norm (total_SAXS_norm_0_1)',
        'D_period': 'D_period'
    }

    for ax, key in zip(axes, ['curvearea_norm', 'SAXS_norm', 'D_period']):
        img, (zmin, zmax), _, _ = images[key]
        im = ax.imshow(
            img,
            origin='lower',
            cmap=cmap,
            vmin=zmin,
            vmax=zmax,
            extent=(0, img.shape[1], 0, img.shape[0])
        )
        ax.set_title(titles[key])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(False)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colmap[key])

    if show_angular_overlay:
        for ax in axes:
            overlay_orientation_arrows(Filenumber, Output_directoryCSV, ax, max_x_all, max_y_all)

    # ---------------------------------
    # ROI definitions based on sample type
    # ---------------------------------
    if sample_type == "control":
        region_specs = [
            ("sample", 8),
            ("dermis", 8),
        ]
    else:
        region_specs = [
            ("sample", 8),
            ("dermis", 8),
            ("wound", 8),
        ]

    # Always write these regions to Excel, even if control leaves wound empty
    all_region_specs = [
        ("sample", 8),
        ("dermis", 8),
        ("wound", 8),
    ]

    regions = {name: [] for name, _ in all_region_specs}
    region_artists = {name: {"markers": [], "lines": []} for name, _ in all_region_specs}
    nonlocal_current_idx = [0]

    def _current():
        return region_specs[nonlocal_current_idx[0]]

    def _prompt():
        name, npts = _current()
        got = len(regions[name])
        print(f"\nNow picking: {name}  ({got}/{npts})")
        print(f"  → click {npts} points for the {name} polygon.")
        print("Controls: r = reset CURRENT region | R = reset ALL | q = finish/save")

    def _draw_region(name):
        pts = regions[name]

        for ln in region_artists[name]["lines"]:
            try:
                ln.remove()
            except Exception:
                pass
        region_artists[name]["lines"].clear()

        if len(pts) < 2:
            fig.canvas.draw_idle()
            return

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        required_lookup = dict(all_region_specs)
        required = required_lookup[name]
        if len(pts) >= required:
            xs = xs + [xs[0]]
            ys = ys + [ys[0]]

        for ax in axes:
            ln, = ax.plot(xs, ys, linewidth=1.2, zorder=19)
            region_artists[name]["lines"].append(ln)

        fig.canvas.draw_idle()

    def _add_marker_all(x, y, name):
        marker = 'x' if name == "sample" else 'o'
        size = 80 if name == "sample" else 55
        for ax in axes:
            sc = ax.scatter([x], [y], marker=marker, s=size, zorder=20)
            region_artists[name]["markers"].append(sc)

    def _reset_region(name):
        for h in region_artists[name]["markers"] + region_artists[name]["lines"]:
            try:
                h.remove()
            except Exception:
                pass
        region_artists[name]["markers"].clear()
        region_artists[name]["lines"].clear()
        regions[name] = []
        fig.canvas.draw_idle()
        print(f"\nReset region: {name}")

    def _advance_if_complete():
        idx = nonlocal_current_idx[0]
        name, npts = region_specs[idx]
        if len(regions[name]) < npts:
            return

        print(f"Completed region: {name}")
        if idx < len(region_specs) - 1:
            nonlocal_current_idx[0] = idx + 1
            _prompt()
        else:
            print("\nAll required regions completed.")
            _save_to_excel()
            plt.close(fig)

    def _make_row_dict():
        row = {
            "experiment": str(Experiment),
            "Filenumber": str(Filenumber),
            "sample_type": str(sample_type)
        }
        for name, npts in all_region_specs:
            pts = regions[name]
            for i in range(1, npts + 1):
                xi = pts[i - 1][0] if len(pts) >= i else np.nan
                yi = pts[i - 1][1] if len(pts) >= i else np.nan
                row[f"{name}_p{i}_x"] = xi
                row[f"{name}_p{i}_y"] = yi
        return row

    def _save_to_excel():
        out = excel_master_path
        binary_out_dir = "/Volumes/LauraDrive/SAXS/Presentations and notes/Images/DefineRegions"

        os.makedirs(os.path.dirname(out), exist_ok=True)
        new_row = pd.DataFrame([_make_row_dict()])

        if os.path.exists(out):
            try:
                old = pd.read_excel(out, sheet_name="ROIs")
                old["experiment"] = old["experiment"].astype(str)
                old["Filenumber"] = old["Filenumber"].astype(str)

                mask = ~(
                    (old["experiment"] == str(Experiment)) &
                    (old["Filenumber"] == str(Filenumber))
                )
                old = old[mask]
                combined = pd.concat([old, new_row], ignore_index=True)
            except Exception:
                combined = new_row
        else:
            combined = new_row

        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            combined.to_excel(writer, sheet_name="ROIs", index=False)

        print(f"\nSaved ROI points to:\n{out}")

        save_images = {
            "curvearea_norm": (images["curvearea_norm"][0], images["curvearea_norm"][1]),
            "SAXS_norm": (images["SAXS_norm"][0], images["SAXS_norm"][1]),
            "D_period": (images["D_period"][0], images["D_period"][1]),
        }

        save_roi_overlay_2x2_png(
            Filenumber=str(Filenumber),
            Experiment=str(Experiment),
            images=save_images,
            regions=regions,
            out_dir=binary_out_dir,
            sample_type=sample_type
        )

    def on_click(event):
        if event.inaxes not in axes:
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = float(event.xdata), float(event.ydata)

        idx = nonlocal_current_idx[0]
        name, npts = region_specs[idx]

        if len(regions[name]) >= npts:
            return

        regions[name].append((x, y))
        _add_marker_all(x, y, name)
        _draw_region(name)
        print(f"{name} point [{len(regions[name])}/{npts}]: x={x:.2f}, y={y:.2f}")

        _advance_if_complete()

    def on_key(event):
        if event.key == 'r':
            idx = nonlocal_current_idx[0]
            name, _ = region_specs[idx]
            _reset_region(name)
            _prompt()
        elif event.key == 'R':
            for name, _ in region_specs:
                _reset_region(name)
            nonlocal_current_idx[0] = 0
            _prompt()
        elif event.key == 'q':
            print("\nFinishing early...")
            _save_to_excel()
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    print("\nROI picking started.")
    _prompt()
    plt.show()

    return regions

def draw_saved_regions_on_axes(axes, regions):
    colors = {
        "sample": "white",
        "dermis": "lime",
        "wound": "red"
    }

    for name, pts in regions.items():
        if len(pts) < 2:
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        if len(pts) >= 3:
            xs = xs + [xs[0]]
            ys = ys + [ys[0]]

        for ax in axes:
            ax.plot(xs, ys, color=colors.get(name, "white"), linewidth=1.4, zorder=19)

            marker = "x" if name == "sample" else "o"
            ax.scatter(
                [p[0] for p in pts],
                [p[1] for p in pts],
                marker=marker,
                s=50,
                color=colors.get(name, "white"),
                zorder=20
            )
            
def _poly_mask(points, shape_xy):
    """
    points: list of (x,y) in the same coordinate system as imshow extent (0..nx, 0..ny)
    shape_xy: (ny, nx)
    Returns boolean mask of pixels whose CENTRES lie inside the polygon.
    """
    ny, nx = shape_xy
    if points is None or len(points) < 3:
        return np.zeros((ny, nx), dtype=bool)

    poly = Path(np.asarray(points, dtype=float))

    yy, xx = np.mgrid[0:ny, 0:nx]
    pts = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])

    inside = poly.contains_points(pts)
    return inside.reshape(ny, nx)


def _overlay_mask(ax, mask, rgba, extent):
    if mask is None or not np.any(mask):
        return
    ny, nx = mask.shape
    img = np.zeros((ny, nx, 4), dtype=float)
    img[..., 0] = rgba[0]
    img[..., 1] = rgba[1]
    img[..., 2] = rgba[2]
    img[..., 3] = rgba[3] * mask.astype(float)
    ax.imshow(img, origin="lower", extent=extent, interpolation="nearest")


def save_roi_overlay_2x2_png(Filenumber, Experiment, images, regions, out_dir, sample_type):
    """
    images: dict key -> (img, (zmin,zmax)) for 3 maps
    regions: dict region_name -> list[(x,y)]
    """
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{Filenumber}_{Experiment}.png")

    img0 = images["curvearea_norm"][0]
    ny, nx = img0.shape
    extent = (0, nx, 0, ny)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    def _plot_map(ax, key, title):
        img, (zmin, zmax) = images[key]
        im = ax.imshow(img, origin="lower", cmap="jet", vmin=zmin, vmax=zmax, extent=extent)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _plot_map(ax1, "curvearea_norm", "curvearea_norm")
    _plot_map(ax2, "SAXS_norm", "SAXS_norm")
    _plot_map(ax3, "D_period", "D_period")

    ax4.set_title(f"ROI overlays ({sample_type})")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_xlim(0, nx)
    ax4.set_ylim(0, ny)

    ax4.imshow(
        np.ones((ny, nx)),
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    sample_mask = _poly_mask(regions.get("sample", []), (ny, nx))
    dermis_mask = _poly_mask(regions.get("dermis", []), (ny, nx))
    wound_mask = _poly_mask(regions.get("wound", []), (ny, nx))

    _overlay_mask(ax4, sample_mask, (0.5, 0.5, 0.5, 1.0), extent)
    _overlay_mask(ax4, dermis_mask, (0.0, 1.0, 0.0, 0.5), extent)

    if sample_type == "wound":
        _overlay_mask(ax4, wound_mask, (1.0, 0.0, 0.0, 0.5), extent)

    ax4.set_aspect("equal")

    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved ROI overlay PNG to:\n{out_png}")


# ----------------------------
# Example usage
# ----------------------------
Output_directory = r"/Users/lauraforster/Documents/Uni/3 - PhD/SAXS/DLS visits/"
Experiment = 'July23'

Output_directoryCSV = Output_directory + Experiment + '/CSVs/'
Filenumber = "692800"

thresholds = {
    'curvearea_norm': 0.1,
    'SAXS_norm': 0.05,
    'D_period': 60
}
zlims = {
    'curvearea_norm': (0.0, 1.0),
    'SAXS_norm': (0.0, 1.0),
    'D_period': None
}

heatmaps_with_rois(
    Filenumber=Filenumber,
    Output_directoryCSV=Output_directoryCSV,
    Experiment=Experiment,
    show_angular_overlay=True,
    thresholds=thresholds,
    zlims=zlims,
    cmap='jet',
)