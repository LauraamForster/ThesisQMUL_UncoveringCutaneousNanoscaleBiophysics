#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:54:02 2025

@author: lauraforster
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import re
from sklearn.mixture import GaussianMixture
from statistics import stdev
from scipy.stats import mannwhitneyu, stats, binned_statistic, kruskal
from scipy.stats import kstest, shapiro
from scipy.stats import lognorm, norm, normaltest
from statsmodels.stats.multitest import multipletests
# import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde
from matplotlib.ticker import FuncFormatter
from collections import defaultdict
from scipy.stats import (mannwhitneyu, stats, binned_statistic, kruskal,
                         kstest, shapiro, lognorm, norm, normaltest,
                         sem)                         #  ← add this
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu


VAR_MAP = {
    # from header/file (no screening needed upstream)
    "Eff_file":    "Eff modulus from file",
    "mod_file":    "modulus from file",

    # contact points / fits
    "CP_Hertz":    "Hertz - Contact Point",
    "mod_Hertz":   "Hertz - Modulus(Pa) fit",
    "Rsq_Hertz":   "Hertz - Rsq",

    "CP_RoV":      "RoV - Contact Point",

    "mod_OP":      "OP - Modulus",
    "Rsq_OP":      "OP - Rsq",

    # holding (no screening)
    "TimeHeld":    "Holding - Time Held (s)",
    "Hold_LoadStart":"Hold - Load Start",
    "Hold_LoadEnd": "Hold - Load End",
    "RelaxFrac":   "Hold - Relaxation Fraction",

    # analytic visco (screened already in ReadManifest via ViscoAna_r2)
    "tau_Visco":   "Visco (Analytic) - tau (s)",
    "G0_Visco":    "Visco (Analytic) - G0 (Pa)",
    "G1_Visco":    "Visco (Analytic) - G1 (Pa)",
    "E0_Visco":    "Visco (Analytic) - E0 (Pa)",
    "Einf_Visco":  "Visco (Analytic) - E_inf (Pa)",
    "Rsq_Visco":   "ViscoAna_r2",
}
def _axis_meta(var):
    """
    var = canonical variable name (e.g. 'mod_Hertz', 'Einf_Visco', etc.)
    Returns: (ylabel_string, formatter or None)
    """

    # ---------- Modulus-like (stored in Pa → show kPa) ----------
    if var in [
        "mod_Hertz", "mod_OP",
        "E0_Visco", "Einf_Visco",
        "G0_Visco", "G1_Visco",
        "Eff_file", "mod_file"
    ]:
        label = f"{var} (kPa)"
        fmt = FuncFormatter(lambda v, _: f"{v/1000:.1f}")
        return label, fmt

    # ---------- Contact Points (stored in m → show µm) ----------
    if var in ["CP_Hertz", "CP_RoV"]:
        label = f"{var} (µm)"
        fmt = FuncFormatter(lambda v, _: f"{v*1e6:.2f}")
        return label, fmt

    # ---------- Load (N) ----------
    if var in ["Hold_LoadStart", "Hold_LoadEnd"]:
        return f"{var} (N)", None

    # ---------- Time (s) ----------
    if var in ["TimeHeld", "tau_Visco"]:
        return f"{var} (s)", None

    # ---------- Relaxation fraction ----------
    if var in ["RelaxFrac"]:
        return f"{var} (–)", None

    # ---------- Rsq ----------
    if var in ["Rsq_Hertz", "Rsq_OP", "Rsq_Visco"]:
        return f"{var} (R²)", None

    # fallback
    return var, None
# ---------------------------------------------------------------------------
# UNIVERSAL ReadManifest – handles Bleo / AP1 / wounding formats
# ---------------------------------------------------------------------------

def ReadManifest(manifest_path, base_path, regions, ST):

    col_map = {
        "Bleo": {
            "lowerdermis":  "lower dermis",
            "upperdermis":  "upper dermis",
            "linescan":     "line scan",
        },
        "Bleomycin": {
            "lowerdermis":  "lower dermis",
            "upperdermis":  "upper dermis",
            "linescan":     "line scan",
        },
        "AP1": {
            "linescan":        "linescan",
            "horiz_linescan":  "horiz_linescan",
        },
        "wounding": {
            "linescan":        "linescan",
            "horiz_linescan":  "horiz_linescan",
        },
    }

    if ST not in col_map:
        raise ValueError(f"Unknown set_type {ST!r}")

    region_cols = col_map[ST]

    df = pd.read_csv(manifest_path, dtype=str).fillna("yes")
    data_dict = {}

    # ------------------------------------------------------------
    # 1) Build manifest dictionary
    # ------------------------------------------------------------
    for _, row in df.iterrows():
        folder = row["FOLDER NAME"]

        sample = {
            "SAMPLE NAME":   row["SAMPLE NAME"],
            "TYPE":          row["TYPE"],
            "FOLDER NAME":   folder,
            "Sample Number": row["Sample Number"],
        }

        for canon, col in region_cols.items():
            sample[canon] = row.get(col, "yes").strip().lower()

        for c in row.index:
            if c not in sample and not c.startswith("Unnamed:"):
                sample[c] = row[c]

        data_dict[folder] = sample

    # ------------------------------------------------------------
    # 2) Iterate regions and load CSV
    # ------------------------------------------------------------
    for folder, sample in data_dict.items():

        s_type = sample["TYPE"]

        for canon_reg in regions:

            empty_region = {}
            if canon_reg == "lowerdermis":
                sample["lower dermis"] = empty_region
            elif canon_reg == "upperdermis":
                sample["upper dermis"] = empty_region
            elif canon_reg == "linescan":
                sample["line scan"] = empty_region
            elif canon_reg == "horiz_linescan":
                sample["horiz_linescan"] = empty_region

            if str(sample.get(canon_reg, "yes")).lower() == "no":
                continue

            # ---------------- PATH LOGIC ----------------
            if ST.lower() in ("bleo", "bleomycin"):
                subfolder = {
                    "lowerdermis": "lowerdermis",
                    "upperdermis": "upperdermis",
                    "linescan":    "linescan",
                }[canon_reg]
                folder_path = os.path.join(base_path, s_type, subfolder)

            elif ST.lower() == "ap1":
                orient = "horiz" if canon_reg == "horiz_linescan" else "vert"
                group  = "AP1" if s_type in ["TS", "VH"] else "CL"
                folder_path = os.path.join(base_path, f"{group} {orient}", s_type)

            elif ST.lower() == "wounding":
                folder_path = os.path.join(base_path, ST, s_type)

            else:
                continue

            csv_path = os.path.join(folder_path,
                                    f"OutputExcel_{folder}_linescan.csv")

            if not os.path.exists(csv_path):
                continue

            try:
                df_raw = pd.read_csv(csv_path)
            except Exception:
                continue

            # ------------------------------------------------------------
            # 3) Build region_dict (NO SCREENING YET)
            # ------------------------------------------------------------
            region_dict = {}

            for _, r in df_raw.iterrows():

                try:
                    x = r["x"]
                    y = r["y"]
                    xy = f"{x}_{y}"

                    region_dict[xy] = {
                        "File Name": r.get("File Name", ""),
                        "x": x,
                        "y": y,

                        "Hertz - Modulus(Pa) fit": r.get("Hertz - Modulus(Pa) fit", np.nan),
                        "Hertz - Rsq": r.get("Hertz - Rsq", np.nan),

                        "OP - Modulus": r.get("OP - Modulus", np.nan),
                        "OP - Rsq": r.get("OP - Rsq", np.nan),

                        "Holding - Time Held (s)": r.get("Holding - Time Held (s)", np.nan),
                        "Hold - Load Start": r.get("Hold - Load Start", np.nan),
                        "Hold - Load End": r.get("Hold - Load End", np.nan),
                        "Hold - Relaxation Fraction": r.get("Hold - Relaxation Fraction", np.nan),

                        "Visco (Analytic) - tau (s)": r.get("Visco (Analytic) - tau (s)", np.nan),
                        "Visco (Analytic) - G0 (Pa)": r.get("Visco (Analytic) - G0 (Pa)", np.nan),
                        "Visco (Analytic) - G1 (Pa)": r.get("Visco (Analytic) - G1 (Pa)", np.nan),
                        "Visco (Analytic) - E0 (Pa)": r.get("Visco (Analytic) - E0 (Pa)", np.nan),
                        "Visco (Analytic) - E_inf (Pa)": r.get("Visco (Analytic) - E_inf (Pa)", np.nan),

                        "ViscoAna_r2": r.get("ViscoAna_r2", np.nan),
                    }

                except Exception:
                    continue

            # ------------------------------------------------------------
            # 4) SCREENING (vectorised, correct)
            # ------------------------------------------------------------
            if region_dict:

                keys = list(region_dict.keys())

                def _screen_field(field, rsq_field=None):
                    vals = np.array([region_dict[k].get(field, np.nan) for k in keys], dtype=float)
                    rsq  = None if rsq_field is None else np.array(
                        [region_dict[k].get(rsq_field, np.nan) for k in keys], dtype=float
                    )
                
                    vals_f = ScreenData(vals, rsq_values=rsq, field=field)
                
                    for kk, vv in zip(keys, vals_f):
                        region_dict[kk][field] = vv
                
                # Hertz
                _screen_field("Hertz - Modulus(Pa) fit", "Hertz - Rsq")
                
                # OP
                _screen_field("OP - Modulus", "OP - Rsq")
                
                # Holding / calculated
                _screen_field("Holding - Time Held (s)")
                _screen_field("Hold - Load Start")
                _screen_field("Hold - Load End")
                _screen_field("Hold - Relaxation Fraction")  # now auto-removes negatives + NO hi clamp
                
                # Visco (fit params gated by ViscoAna_r2)
                for f in [
                    "Visco (Analytic) - tau (s)",
                    "Visco (Analytic) - G0 (Pa)",
                    "Visco (Analytic) - G1 (Pa)",
                    "Visco (Analytic) - E0 (Pa)",
                    "Visco (Analytic) - E_inf (Pa)",
                ]:
                    _screen_field(f, "ViscoAna_r2")

            # ------------------------------------------------------------
            # 5) Store region
            # ------------------------------------------------------------
            if region_dict:
                if canon_reg == "lowerdermis":
                    sample["lower dermis"] = region_dict
                elif canon_reg == "upperdermis":
                    sample["upper dermis"] = region_dict
                elif canon_reg == "linescan":
                    sample["line scan"] = region_dict
                elif canon_reg == "horiz_linescan":
                    sample["horiz_linescan"] = region_dict

    return data_dict

def ScreenData(y, rsq_values=None, field=None, rsq_min=0.3, hi=1_000_000):
    """
    Field-aware screening.

    Rules:
      Hertz/OP modulus:
        - Rsq gate (rsq_values required)
        - y < 0 -> NaN
        - y > hi -> NaN
        - 3σ outlier removal

      Visco fit params (tau, G0, G1, E0, E_inf):
        - Rsq gate (ViscoAna_r2 passed in rsq_values)
        - y < 0 -> NaN   (tau/G/E should not be negative)
        - no upper clamp by default (unless you want one)
        - 3σ outlier removal

      RelaxFrac:
        - y < 0 -> NaN   (your request)
        - NO rsq gate
        - NO hi clamp
        - 3σ outlier removal (kept, since you said keep 3σ on everything)

      Holding time/load start/load end:
        - no rsq gate
        - no sign clamp by default
        - 3σ outlier removal
    """
    y = np.asarray(y, dtype=np.float64)

    field = "" if field is None else str(field)

    # ---- Identify field category ----
    is_hertz_mod = field == "Hertz - Modulus(Pa) fit"
    is_op_mod    = field == "OP - Modulus"
    is_visco_fit = field in {
        "Visco (Analytic) - tau (s)",
        "Visco (Analytic) - G0 (Pa)",
        "Visco (Analytic) - G1 (Pa)",
        "Visco (Analytic) - E0 (Pa)",
        "Visco (Analytic) - E_inf (Pa)",
    }
    is_relaxfrac = field == "Hold - Relaxation Fraction"

    # ---- Rsq gating only for fitted things ----
    if (is_hertz_mod or is_op_mod or is_visco_fit) and rsq_values is not None:
        rsq_values = np.asarray(rsq_values, dtype=np.float64)
        y = np.where(rsq_values < rsq_min, np.nan, y)

    # ---- Negatives ----
    if is_hertz_mod or is_op_mod or is_visco_fit or is_relaxfrac:
        y = np.where(y < 0, np.nan, y)

    # ---- Upper clamp only for modulus (NOT RelaxFrac) ----
    if is_hertz_mod or is_op_mod:
        y = np.where(y > hi, np.nan, y)

    # ---- 3σ outlier removal (on remaining finite values) ----
    finite = np.isfinite(y)
    if finite.sum() < 2:
        return y

    mu = float(np.nanmean(y))
    sd = float(np.nanstd(y, ddof=1))
    if not np.isfinite(sd) or sd == 0:
        return y

    lo, up = mu - 3*sd, mu + 3*sd
    y = np.where((y < lo) | (y > up), np.nan, y)
    return y

def explore_sample(data_dict, sample_key, show_points_per_region=3):
    def format_float(value):
        if pd.isna(value):
            return "N/A"
        elif abs(value) < 0.01 and value != 0:
            return f"{value:.2e}"
        elif abs(value) > 10000:
            return f"{value:.1f}"
        else:
            return f"{value:.4f}"

    if sample_key not in data_dict:
        print(f"❌ Sample '{sample_key}' not found in dictionary.")
        return

    sample = data_dict[sample_key]

    print(f"\n📁 Sample: {sample_key}")
    print("--------------------------------------------------")

    print("🧾 Metadata:")
    for key, value in sample.items():
        if key not in ["lower dermis", "upper dermis", "line scan", "horiz_linescan"]:
            print(f"  {key}: {value}")

    print("\n📍 Region Data (Vertical):")
    line_scan = sample.get("line scan", {})
    if isinstance(line_scan, dict) and line_scan:
        region_groups = defaultdict(list)
        for k, v in line_scan.items():
            region_groups[v.get("layer", "unassigned")].append((k, v))
        for region, points in region_groups.items():
            print(f"  ✅ {region}: {len(points)} points")
            for pt_key, pt in points[:show_points_per_region]:
                print(f"    ↳ ({pt_key}):")
                for k, v in pt.items():
                    print(f"       {k}: {format_float(v) if isinstance(v, float) else v}")
    else:
        print("  ❌ No vertical data found")

    print("\n📍 Region Data (Horizontal):")
    horiz_scan = sample.get("horiz_linescan", {})
    if isinstance(horiz_scan, dict) and horiz_scan:
        region_groups = defaultdict(list)
        for k, v in horiz_scan.items():
            region_groups[v.get("region", "unassigned")].append((k, v))
        for region, points in region_groups.items():
            print(f"  ✅ {region}: {len(points)} points")
            for pt_key, pt in points[:show_points_per_region]:
                print(f"    ↳ ({pt_key}):")
                for k, v in pt.items():
                    print(f"       {k}: {format_float(v) if isinstance(v, float) else v}")
    else:
        print("  ❌ No horizontal data found")

    print("--------------------------------------------------\n")

def CutSampleLengths(data_dict):
    for sample_key, sample in data_dict.items():
        # --- Vertical (line scan) ---
        line_scan = sample.get("line scan", {})
        if isinstance(line_scan, dict) and line_scan:
            try:
                total_pts = int(sample.get("no pts", np.nan))
                lengths = {
                    "left glass": int(sample.get("Measured left glass", 0)),
                    "subcut": int(sample.get("Measured Subcut layer", 0)),
                    "dermis": int(sample.get("Measured dermis", 0)),
                    "epidermis": int(sample.get("Measured epi", 0)),
                    "right glass": int(sample.get("Measured right glass", 0))
                }
            except (TypeError, ValueError):
                print(f"⚠️ Skipping {sample_key}: invalid vertical region lengths")
                lengths = {}
            
            if all(v == 0 for v in lengths.values()):
                print(f"Sample {sample_key}: All vertical layer lengths are 0 — assigning all points to 'dermis'.")
                for pt in line_scan.values():
                    pt["layer"] = "dermis"
            elif lengths and sum(lengths.values()) != total_pts:
                print(f"⚠️ Sample {sample_key}: Vertical lengths ≠ total points ({sum(lengths.values())} ≠ {total_pts})")
            elif lengths:
                sorted_keys = sorted(line_scan.keys(), key=lambda k: (line_scan[k]["y"], line_scan[k]["x"]))
                current_idx = 0
                for region, length in lengths.items():
                    for i in range(current_idx, current_idx + length):
                        if i < len(sorted_keys):
                            point_key = sorted_keys[i]
                            line_scan[point_key]["layer"] = region
                    current_idx += length

        # --- Horizontal (horiz_linescan) ---
        horiz_scan = sample.get("horiz_linescan", {})
        if isinstance(horiz_scan, dict) and horiz_scan:
            try:
                total_pts = int(sample.get("no pts horiz", np.nan))
                lengths = {
                    "left": int(sample.get("Measured left ", 0)),
                    "centre": int(sample.get("measured centre", 0)),
                    "right": int(sample.get("Measured right", 0))
                }
            except (TypeError, ValueError):
                print(f"⚠️ Skipping {sample_key}: invalid horizontal region lengths")
                lengths = {}

            if all(v == 0 for v in lengths.values()):
                print(f"ℹ️ Sample {sample_key}: All horizontal region lengths are 0 — skipping trimming.")
            elif lengths and sum(lengths.values()) != total_pts:
                print(f"⚠️ Sample {sample_key}: Horizontal lengths ≠ total points ({sum(lengths.values())} ≠ {total_pts})")
            elif lengths:
                sorted_keys = sorted(horiz_scan.keys(), key=lambda k: (horiz_scan[k]["x"], horiz_scan[k]["y"]))
                current_idx = 0
                for region, length in lengths.items():
                    for i in range(current_idx, current_idx + length):
                        if i < len(sorted_keys):
                            point_key = sorted_keys[i]
                            horiz_scan[point_key]["region"] = region
                    current_idx += length
    return data_dict
        

# ---------------------------------------------------------------------------
# BIN DATA
# ---------------------------------------------------------------------------
def PrepareBinnedData(data_dict, nbins, layer, debug=False):
    """
    Bins line-scan points within a given layer into nbins per *line* (handles parallel lines).
    Returns:
      binned[subtype]["points"] = list of dict rows:
        {
          "sample": sample_key,
          "subtype": stype,
          "line_id": <value of constant coordinate for that line>,
          "axis": "x" or "y",
          "idx": index along line,
          "bin": 0..nbins-1,
          "x": ..., "y": ...,
          <canon vars from VAR_MAP>...
        }
    """
    sample_types = sorted({s.get("TYPE") for s in data_dict.values() if s.get("TYPE")})
    binned = {t: {"points": []} for t in sample_types}

    def _to_float(a):
        a = np.asarray(a)
        # robust: handle strings like "1" cleanly
        return a.astype(float)

    def _split_into_lines(pts_xy):
        """
        pts_xy: list of dicts with numeric x,y
        Returns (axis, groups) where groups is list of arrays of indices into pts_xy.
        """
        xs = _to_float([p["x"] for p in pts_xy])
        ys = _to_float([p["y"] for p in pts_xy])

        ux, uy = np.unique(xs[~np.isnan(xs)]), np.unique(ys[~np.isnan(ys)])
        rx, ry = (np.nanmax(xs) - np.nanmin(xs)) if np.isfinite(xs).any() else 0.0, (np.nanmax(ys) - np.nanmin(ys)) if np.isfinite(ys).any() else 0.0

        # Decide scan axis: whichever has the larger spread (or more unique values)
        if (rx > ry) or (len(ux) > len(uy)):
            axis = "x"           # x changes along scan; y identifies parallel lines
            line_coord = ys
        else:
            axis = "y"           # y changes along scan; x identifies parallel lines
            line_coord = xs

        # Group into parallel lines by constant-ish coordinate (exact in your format)
        finite = np.isfinite(line_coord)
        if finite.sum() == 0:
            return axis, []

        # Use unique coordinate values as line IDs
        line_ids = np.unique(line_coord[finite])
        groups = []
        for lid in line_ids:
            idx = np.where(line_coord == lid)[0]
            if idx.size:
                groups.append((lid, idx))

        return axis, groups

    def _bin_indices(n, nbins):
        """
        Split n points into nbins as evenly as possible.
        Returns bin assignment array length n with values 0..nbins-1.
        """
        if n <= 0:
            return np.array([], dtype=int)
        # equal-count bins (better than equal-distance when spacing irregular)
        edges = np.linspace(0, n, nbins + 1)
        b = np.empty(n, dtype=int)
        for i in range(n):
            # bin by index position along line
            bi = np.searchsorted(edges, i, side="right") - 1
            b[i] = int(np.clip(bi, 0, nbins - 1))
        return b

    # ---------------- main loop ----------------
    for sample_key, sample in data_dict.items():
        stype = sample.get("TYPE")
        if stype not in binned:
            continue

        line_scan = sample.get("line scan", {})
        if not isinstance(line_scan, dict) or not line_scan:
            continue

        # collect dermis points with usable x,y
        pts = []
        for pt in line_scan.values():
            if pt.get("layer") != layer:
                continue
            x, y = pt.get("x", np.nan), pt.get("y", np.nan)
            if pd.isna(x) or pd.isna(y):
                continue
            pts.append(pt)
        
        # # --- debug counts for dermis-only points ---
        # n_total_layer = len(pts)
        # n_hertz = sum(np.isfinite(pt.get("Hertz - Modulus(Pa) fit", np.nan)) for pt in pts)
        # n_tau   = sum(np.isfinite(pt.get("Visco (Analytic) - tau (s)", np.nan)) for pt in pts)
        # n_both  = sum(
        #     np.isfinite(pt.get("Hertz - Modulus(Pa) fit", np.nan)) and
        #     np.isfinite(pt.get("Visco (Analytic) - tau (s)", np.nan))
        #     for pt in pts
        # )
        
        # print(
        #     f"[LAYER] {stype} | {sample_key} | layer={layer}: "
        #     f"total={n_total_layer}, mod_Hertz={n_hertz}, tau_Visco={n_tau}, both={n_both}"
        # )

        if len(pts) < 2:
            if debug:
                print(f"[BINDBG] {sample_key} ({stype}) <2 points in layer '{layer}'")
            continue

        axis, line_groups = _split_into_lines(pts)
        if not line_groups:
            if debug:
                print(f"[BINDBG] {sample_key} ({stype}) could not form line groups")
            continue

        # for each line, order points along scan axis and bin by index
        used_points = 0
        for line_id, idx in line_groups:
            # pull subset
            sub = [pts[i] for i in idx]

            # sort along scan axis
            if axis == "x":
                order = np.argsort(_to_float([p["x"] for p in sub]))
            else:
                order = np.argsort(_to_float([p["y"] for p in sub]))
            sub = [sub[i] for i in order]

            n = len(sub)
            if n < 2:
                continue

            bins = _bin_indices(n, nbins)

            for j, (p, b) in enumerate(zip(sub, bins)):
                row = {
                    "sample": sample_key,
                    "subtype": stype,
                    "line_id": float(line_id),
                    "axis": axis,
                    "idx": int(j),
                    "bin": int(b),
                    "x": float(p.get("x", np.nan)),
                    "y": float(p.get("y", np.nan)),
                }
                for canon, key in VAR_MAP.items():
                    row[canon] = p.get(key, np.nan)
                binned[stype]["points"].append(row)
                used_points += 1

        if debug:
            print(f"[BINDBG] {sample_key} ({stype}) axis={axis} lines={len(line_groups)} points_binned={used_points}")

    # ---------------- summary debug ----------------
    if debug:
        print(f"\n[BINDBG] Summary (layer='{layer}', nbins={nbins})")
        for stype, entry in binned.items():
            pts = entry["points"]
            print(f"  {stype}: points={len(pts)}")
            if not pts:
                continue
            for b in range(nbins):
                nb = sum(p["bin"] == b for p in pts)
                print(f"    bin {b}: n={nb}")

    # ------------------------------------------------------------------
    # NEW: add per-line normalised position (0–100) using idx within line
    # ------------------------------------------------------------------
    for stype, entry in binned.items():
        pts = entry.get("points", [])
        if not pts:
            continue

        # group indices by (sample, line_id)
        groups = defaultdict(list)
        for i, p in enumerate(pts):
            groups[(p.get("sample"), p.get("line_id"))].append(i)

        for (samp, lid), idxs in groups.items():
            # sort by idx so direction is consistent
            idxs = sorted(idxs, key=lambda ii: pts[ii].get("idx", 0))
            n = len(idxs)
            if n < 2:
                pts[idxs[0]]["norm_pos"] = np.nan
                continue

            for j, ii in enumerate(idxs):
                pts[ii]["norm_pos"] = 100.0 * j / (n - 1)

    return binned

def _get_points(binned_dict, subtype):
    return binned_dict.get(subtype, {}).get("points", [])

def GetBinnedValues(binned_dict, subtype, var):
    pts = _get_points(binned_dict, subtype)
    if not pts:
        return np.array([]), np.array([]), np.array([])
    bins = np.array([p.get("bin", -1) for p in pts], int)
    # prefer 'norm' (0–100), else fall back to 'idx' if that's what you stored
    pos  = np.array([p.get("norm", p.get("idx", np.nan)) for p in pts], float)
    vals = np.array([p.get(var, np.nan) for p in pts], float)
    return bins, pos, vals

def _points_for(binned_dict, subtype):
    return binned_dict.get(subtype, {}).get("points", [])

def _values_by_bin(binned_dict, subtype, var, nbins, pooled=True):
    """
    Returns:
      per_bin_vals : list length nbins, each is a 1D np.array of values to use for stats/violin
      per_bin_raw  : list length nbins, each is a 1D np.array of RAW points (always pooled points)
                    (used for scatter overlays so you can still show every point if you want)
    pooled=True  -> per_bin_vals are all points pooled in that bin
    pooled=False -> per_bin_vals are per-sample bin means (one value per sample per bin)
    """
    pts = _points_for(binned_dict, subtype)
    if not pts:
        return [np.array([]) for _ in range(nbins)], [np.array([]) for _ in range(nbins)]

    # raw pooled points (always)
    per_bin_raw = []
    for b in range(nbins):
        v = np.array([p.get(var, np.nan) for p in pts if int(p.get("bin", -1)) == b], float)
        v = v[np.isfinite(v)]
        per_bin_raw.append(v)

    if pooled:
        return per_bin_raw, per_bin_raw

    # pooled=False: compute per-sample means inside each bin
    per_bin_vals = []
    for b in range(nbins):
        # group values by sample within this bin
        samp_to_vals = {}
        for p in pts:
            if int(p.get("bin", -1)) != b:
                continue
            s = p.get("sample", None)
            val = p.get(var, np.nan)
            if s is None or not np.isfinite(val):
                continue
            samp_to_vals.setdefault(s, []).append(float(val))

        # one value per sample per bin
        means = np.array([np.mean(vs) for vs in samp_to_vals.values() if len(vs)], float)
        means = means[np.isfinite(means)]
        per_bin_vals.append(means)

    return per_bin_vals, per_bin_raw

def BinStats(binned_dict, subtype, var, nbins, pooled=True):
    per_bin_vals, _ = _values_by_bin(binned_dict, subtype, var, nbins, pooled=pooled)
    means = np.full(nbins, np.nan)
    stds  = np.full(nbins, np.nan)
    ns    = np.zeros(nbins, int)

    for b in range(nbins):
        v = per_bin_vals[b]
        ns[b] = int(v.size)
        if v.size:
            means[b] = float(np.mean(v))
            stds[b]  = float(np.std(v, ddof=1)) if v.size > 1 else 0.0

    return means, stds, ns

# ---------------------------------------------------------------------------
# PLOT DATA
# ---------------------------------------------------------------------------

def PlotBarByBin(binned_dict, order, nbins, var, colors=None, title=None, pooled=True, ylim=None, ylog=False, show_points=True):
    colors = colors or {}

    M = np.array([BinStats(binned_dict, st, var, nbins, pooled=pooled)[0] for st in order])
    S = np.array([BinStats(binned_dict, st, var, nbins, pooled=pooled)[1] for st in order])

    x = np.arange(nbins)
    w = 0.8 / max(1, len(order))

    fig, ax = plt.subplots()

    for i, st in enumerate(order):
        offset = (i - (len(order)-1)/2) * w
        col = colors.get(st, None)

        ax.bar(x + offset, M[i], width=w, yerr=S[i], capsize=3, label=st, color=col)

        if show_points:
            # points to scatter:
            per_bin_vals, per_bin_raw = _values_by_bin(binned_dict, st, var, nbins, pooled=pooled)
            for b in range(nbins):
                vals = per_bin_raw[b] if pooled else per_bin_vals[b]   # pooled=True -> every point; pooled=False -> per-sample means
                if vals.size:
                    ax.scatter(np.full(vals.size, x[b] + offset), vals, s=10, alpha=0.6, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Bin {i+1}" for i in range(nbins)])
    ax.set_xlabel("Depth bin")
    ax.set_ylabel(var)
    ax.set_title(title or f"{var} by bin")
    ax.set_ylim(ylim)
    if ylog == True:
        ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    return fig, ax

def export_plotbarbybin_to_excel(
    binned_dict,
    order,
    nbins,
    var,
    *,
    pooled=True,
    out_xlsx="PlotBarByBin_export.xlsx",
    sheet_summary="summary",
    sheet_points="points",
):
    """
    Exports exactly what PlotBarByBin plots.

    Sheet 1: bin, subtype, mean, std, n
      - mean/std/n come from BinStats(...) (same as bars/error bars)

    Sheet 2: bin, subtype, value, point_id
      - value are the exact y-values used for scatter in PlotBarByBin
        (pooled=True -> every raw point; pooled=False -> per-sample means)
    """
    rows_sum, rows_pts = [], []

    for st in order:
        # BinStats now returns (means, stds, ns)
        M, S, N = BinStats(binned_dict, st, var, nbins, pooled=pooled)
        M = np.asarray(M, float).ravel()
        S = np.asarray(S, float).ravel()
        N = np.asarray(N, int).ravel()

        per_bin_vals, per_bin_raw = _values_by_bin(binned_dict, st, var, nbins, pooled=pooled)

        for b in range(nbins):
            # These are the y-values you actually plotted as scatter
            vals = per_bin_raw[b] if pooled else per_bin_vals[b]
            vals = np.asarray(vals, float).ravel()
            vals = vals[np.isfinite(vals)]

            rows_sum.append({
                "bin": b + 1,                 # 1..nbins
                "subtype": str(st),
                "mean": float(M[b]) if b < M.size and np.isfinite(M[b]) else np.nan,
                "std":  float(S[b]) if b < S.size and np.isfinite(S[b]) else np.nan,
                "n":    int(N[b]) if b < N.size else int(vals.size),
            })

            for i, v in enumerate(vals, start=1):
                rows_pts.append({
                    "bin": b + 1,
                    "subtype": str(st),
                    "value": float(v),
                    "point_id": i,              # 1..n per (subtype, bin)
                })

    df_sum = pd.DataFrame(rows_sum)
    df_pts = pd.DataFrame(rows_pts)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        df_sum.to_excel(xw, sheet_name=sheet_summary, index=False)
        df_pts.to_excel(xw, sheet_name=sheet_points, index=False)

    return df_sum, df_pts

def PlotViolinByBin(binned_dict, order, nbins, var, colors=None, title=None,
                    pooled=True, ylim=None, ylog=False, show_points=True, alpha=0.5, s=10):
    colors = colors or {}
    fig, ax = plt.subplots()

    positions = []
    datasets  = []
    dataset_meta = []  # (st, b, pos) for ones we actually violin

    pos = 1
    gap = 1

    # First pass: collect data + decide which get violins
    all_pos_centres = []
    for b in range(nbins):
        bin_positions = []
        for st in order:
            per_bin_vals, per_bin_raw = _values_by_bin(binned_dict, st, var, nbins, pooled=pooled)
            vals = per_bin_vals[b]
            raw  = per_bin_raw[b]
            col = colors.get(st, None)

            # scatter overlay (requested)
            if show_points:
                scatter_vals = raw if pooled else vals
                if scatter_vals.size:
                    ax.scatter(np.full(scatter_vals.size, pos), scatter_vals, color=col, alpha=alpha, s=s)

            # violin only if enough points
            if vals.size >= 2:
                datasets.append(vals)
                positions.append(pos)
                dataset_meta.append((st, b, pos))

            bin_positions.append(pos)
            pos += 1

        all_pos_centres.append(np.mean(bin_positions))
        pos += gap

    # Draw violins for the datasets that qualify
    if datasets:
        parts = ax.violinplot(datasets, positions=positions, showmeans=False, showmedians=True, showextrema=False)
        # colour each body to match subtype colour
        for body, (st, _, _) in zip(parts["bodies"], dataset_meta):
            col = colors.get(st, None)
            if col is not None:
                body.set_facecolor(col)
                body.set_alpha(0.35)
                body.set_edgecolor(col)

        if "cmedians" in parts and parts["cmedians"] is not None:
            parts["cmedians"].set_linewidth(1.5)

    ax.set_xticks(all_pos_centres)
    ax.set_xticklabels([f"Bin {b+1}" for b in range(nbins)])
    ax.set_xlabel("Depth bin")
    if ylim is not None:
        ax.set_ylim(ylim)
    if ylog == True:
        ax.set_yscale('log')
    ax.set_ylabel(var)
    ax.set_title(title or f"{var} violin by bin")
    plt.tight_layout()
    return fig, ax

def _per_line_norm_positions(pts):
    """
    Returns norm_pos array aligned to pts order, normalised 0..100 per (sample,line_id).
    Uses idx ordering within each line.
    """
    n = len(pts)
    norm = np.full(n, np.nan, float)

    # group indices by (sample, line_id)
    groups = {}
    for i, p in enumerate(pts):
        sid = p.get("sample", None)
        lid = p.get("line_id", None)
        groups.setdefault((sid, lid), []).append(i)

    for (_, _), idxs in groups.items():
        # sort within the line by idx (fallback: by y then x)
        def _sort_key(i):
            pi = pts[i]
            if "idx" in pi and np.isfinite(pi.get("idx", np.nan)):
                return (0, float(pi["idx"]))
            return (1, float(pi.get("y", np.nan)), float(pi.get("x", np.nan)))

        idxs_sorted = sorted(idxs, key=_sort_key)
        m = len(idxs_sorted)
        if m == 1:
            norm[idxs_sorted[0]] = 0.0
        else:
            for j, i in enumerate(idxs_sorted):
                norm[i] = 100.0 * j / (m - 1)

    return norm

def PlotScatterByNormPos(binned_dict, order, nbins, var,
                         colors=None, title=None,
                         alpha=0.35, s=12,
                         bestfit=False, ylim=None,ylog=False):
    """
    Scatter of variable vs normalised position (0–100%),
    normalised per sample + line.
    
    bestfit=True adds linear regression per subtype.
    """

    colors = colors or {}
    fig, ax = plt.subplots()

    for st in order:
        pts = binned_dict.get(st, {}).get("points", [])
        if not pts:
            continue

        # Resolve correct key
        vkey = var if var in pts[0] else VAR_MAP.get(var, var)

        y = np.array([p.get(vkey, np.nan) for p in pts], float)
        x = _per_line_norm_positions(pts)

        m = np.isfinite(x) & np.isfinite(y)
        if not np.any(m):
            continue

        x_valid = x[m]
        y_valid = y[m]

        color = colors.get(st, None)

        # Scatter
        ax.scatter(x_valid, y_valid,
                   alpha=alpha,
                   s=s,
                   label=st,
                   color=color)

        # ---------- BEST FIT ----------
        if bestfit and len(x_valid) >= 2:
            coeff = np.polyfit(x_valid, y_valid, 1)
            xx = np.linspace(0, 100, 200)
            yy = coeff[0] * xx + coeff[1]

            ax.plot(xx, yy,
                    linewidth=2,
                    color=color)

    # ---------- Bin boundaries ----------
    for k in range(1, nbins):
        ax.axvline(100.0 * k / nbins,
                   linestyle="--",
                   linewidth=1,
                   color="grey",
                   alpha=0.5)

    ax.set_xlabel("Normalised position (0–100%)")

    ylabel, fmt = _axis_meta(var)
    ax.set_ylabel(ylabel)
    if fmt is not None:
        ax.yaxis.set_major_formatter(fmt)

    ax.set_title(title or f"{var} vs normalised position")
    ax.legend()
    if ylim is not None:
        ax.set_ylim(ylim)
    if ylog == True:
        ax.set_yscale('log')
    plt.tight_layout()
    return fig, ax

def PlotHistogram(binned_dict, order, var, colors=None, ylog=False,
                  title=None, bins=40, alpha=0.4, normalise=True,
                  SplitHistos=False):
    colors = colors or {}

    # collect values first so limits are shared if needed
    vals_by_group = {}
    global_vals = []

    for st in order:
        _, _, vals = GetBinnedValues(binned_dict, st, var)
        vals = vals[np.isfinite(vals)]
        vals_by_group[st] = vals
        if vals.size:
            global_vals.append(vals)

    if not global_vals:
        fig, ax = plt.subplots()
        ax.set_title(title or f"{var} distribution")
        return fig, ax

    all_vals = np.concatenate(global_vals)
    x_min, x_max = np.min(all_vals), np.max(all_vals)
    bin_edges = np.linspace(x_min, x_max, bins + 1) if np.isscalar(bins) else bins

    def _weights(vals):
        return np.ones_like(vals) / vals.size if (normalise and vals.size) else None

    # --------------------------------------------------
    # split into 2x2 panels
    # --------------------------------------------------
    if SplitHistos:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
        axes = axes.flatten()

        ymax = 0
        # first pass to find common y limit
        for st in order[:4]:
            vals = vals_by_group.get(st, np.array([]))
            if vals.size == 0:
                continue
            hist, _ = np.histogram(vals, bins=bin_edges, weights=_weights(vals))
            ymax = max(ymax, np.max(hist) if hist.size else 0)

        for i, st in enumerate(order[:4]):
            ax = axes[i]
            vals = vals_by_group.get(st, np.array([]))
            if vals.size:
                ax.hist(
                    vals,
                    bins=bin_edges,
                    alpha=alpha,
                    color=colors.get(st, None),
                    ec='k',
                    weights=_weights(vals)
                )

            ax.set_title(st)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1)

            if ylog:
                ax.set_yscale('log')

            ax.set_xlabel(var)
            ax.set_ylabel("Density" if normalise else "Count")
            ax.grid(True, alpha=0.2)

        # hide unused panels if fewer than 4 groups
        for j in range(len(order[:4]), 4):
            axes[j].axis("off")

        fig.suptitle(title or f"{var} distribution")
        plt.tight_layout()
        return fig, axes

    # --------------------------------------------------
    # single overlaid histogram
    # --------------------------------------------------
    fig, ax = plt.subplots()
    for st in order:
        vals = vals_by_group.get(st, np.array([]))
        if vals.size:
            ax.hist(
                vals,
                bins=bin_edges,
                alpha=alpha,
                label=st,
                color=colors.get(st, None),
                ec='k',
                weights=_weights(vals)
            )

    ax.set_xlabel(var)
    ax.set_ylabel("Density" if normalise else "Count")

    if ylog:
        ax.set_yscale('log')

    ax.set_title(title or f"{var} distribution")
    ax.legend()
    plt.tight_layout()
    return fig, ax

def _resolve_var_key(var, pts):
    """
    pts contain canonical keys (Eff_file, mod_Hertz, Einf_Visco, ...)
    NOT the CSV header strings. So prefer canonical keys first.
    """
    if not pts:
        return var

    keys = set(pts[0].keys())

    # 1) already canonical
    if var in keys:
        return var

    # 2) if user accidentally passed the CSV header string, allow it
    if var in VAR_MAP.values() and var in keys:
        return var

    # 3) if var is canonical but we mapped it to header earlier, don't do that
    #    instead only use VAR_MAP if its VALUE actually exists in the point dict
    mapped = VAR_MAP.get(var, None)
    if mapped is not None and mapped in keys:
        return mapped

    # 4) otherwise: just return var (will likely yield NaNs, but that's honest)
    return var

def PlotCorrelation(binned_dict, order, xvar, yvar, colors=None, title=None,
                    alpha=0.35, s=12, bestfit=False, xlim=None, ylim=None, xlog=False, ylog=False, dbg=False):
    """
    Correlation using ALL layer-filtered points (no binning): yvar vs xvar.
    Points source is binned_dict[subtype]["points"].
    """
    colors = colors or {}
    fig, ax = plt.subplots()

    any_plotted = False

    for st in order:
        pts = binned_dict.get(st, {}).get("points", [])
        if not pts:
            if dbg: print(f"[CORRDBG] {st}: no points")
            continue

        xkey = _resolve_var_key(xvar, pts)
        ykey = _resolve_var_key(yvar, pts)

        x = np.asarray([p.get(xkey, np.nan) for p in pts], float)
        y = np.asarray([p.get(ykey, np.nan) for p in pts], float)

        m = np.isfinite(x) & np.isfinite(y)
        n = int(m.sum())

        if dbg:
            print(f"[CORRDBG] {st}: pts={len(pts)} finite_pairs={n} "
                  f"x='{xvar}'→'{xkey}' y='{yvar}'→'{ykey}'")

        if n == 0:
            continue

        col = colors.get(st, None)
        ax.scatter(x[m], y[m], alpha=alpha, s=s, label=st, color=col)
        any_plotted = True

        if bestfit and n >= 2:
            a, b = np.polyfit(x[m], y[m], 1)
            xx = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), 200)
            ax.plot(xx, a*xx + b, linewidth=2, color=col)

    # axis labels + sensible units
    xlabel, xfmt = _axis_meta(xvar)
    ylabel, yfmt = _axis_meta(yvar)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)    
    if xlim is not None:
        ax.set_xlim(xlim)
    if xlog == True:
        ax.set_xscale('log')
    if ylim is not None:
        ax.set_ylim(ylim)
    if ylog == True:
        ax.set_yscale('log')
    if xfmt is not None:
        ax.xaxis.set_major_formatter(xfmt)
    if yfmt is not None:
        ax.yaxis.set_major_formatter(yfmt)

    ax.set_title(title or f"{yvar} vs {xvar}")
    ax.legend()
    plt.tight_layout()

    if dbg and not any_plotted:
        print("[CORRDBG] Nothing plotted: all finite_pairs=0. Likely var names don't match keys in points.")

    return fig, ax

def PlotScatterTrends(binned_dict, order, PlotVariable, colors=None, nbins=None,
                     title=None, ylim=None, ylog=False,
                     show_points=True, show_bin_means=True,
                     point_alpha=0.25, point_size=12, line_width=2):

    colors = colors or {}
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for st in order:
        pts = _points_for(binned_dict, st)
        if not pts:
            continue

        x = np.array([p.get("norm_pos", np.nan) for p in pts], float)
        y = np.array([p.get(PlotVariable, np.nan) for p in pts], float)

        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        if x.size < 2:
            continue

        col = colors.get(st, None)

        # ---------------- RAW SCATTER ----------------
        if show_points:
            ax.scatter(x, y, s=point_size, alpha=point_alpha,
                       color=col, label=f"{st} points")

        # ---------------- RAW FIT ----------------
        m_raw, c_raw = np.polyfit(x, y, 1)
        y_pred = m_raw * x + c_raw

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        print(f"\n[{st}] RAW FIT:")
        print(f"  n = {x.size}")
        print(f"  slope = {m_raw:.6f}")
        print(f"  intercept = {c_raw:.6f}")
        print(f"  R² = {r2:.4f}")

        xx = np.linspace(0, 100, 200)
        yy = m_raw * xx + c_raw
        ax.plot(xx, yy, color=col, linewidth=line_width, linestyle="-",
                label=f"{st} raw fit")

        # ---------------- BIN MEAN FIT ----------------
        if show_bin_means:
            if nbins is None:
                bins_present = [p.get("bin", np.nan) for p in pts]
                bins_present = [b for b in bins_present if pd.notna(b)]
                nb = int(max(bins_present)) + 1 if bins_present else 0
            else:
                nb = nbins

            bx, by = [], []
            for b in range(nb):
                pb = [p for p in pts if int(p.get("bin", -1)) == b]
                if not pb:
                    continue

                xb = np.array([p.get("norm_pos", np.nan) for p in pb], float)
                yb = np.array([p.get(PlotVariable, np.nan) for p in pb], float)

                keepb = np.isfinite(xb) & np.isfinite(yb)
                xb, yb = xb[keepb], yb[keepb]
                if xb.size == 0:
                    continue

                bx.append(np.mean(xb))
                by.append(np.mean(yb))

            bx = np.array(bx, float)
            by = np.array(by, float)

            if bx.size:
                ax.scatter(bx, by, s=45, color=col, alpha=0.95, marker="o")

            if bx.size >= 2:
                m_bin, c_bin = np.polyfit(bx, by, 1)
                y_pred = m_bin * bx + c_bin

                ss_res = np.sum((by - y_pred) ** 2)
                ss_tot = np.sum((by - np.mean(by)) ** 2)
                r2_bin = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

                print(f"[{st}] BIN FIT:")
                print(f"  bins used = {bx.size}")
                print(f"  slope = {m_bin:.6f}")
                print(f"  intercept = {c_bin:.6f}")
                print(f"  R² = {r2_bin:.4f}")

                xx = np.linspace(0, 100, 200)
                yy = m_bin * xx + c_bin
                ax.plot(xx, yy, color=col, linewidth=line_width, linestyle="--",
                        label=f"{st} bin fit")

    ax.set_xlabel("Normalised position (%)")
    ax.set_ylabel(PlotVariable)
    ax.set_title(title or f"{PlotVariable} vs normalised position")

    if ylim is not None:
        ax.set_ylim(ylim)
    if ylog:
        ax.set_yscale("log")

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig, ax

def PlotBestFitByBin(binned_dict, order, nbins, PlotVariable, colors=None,
                     title=None, ylim=None, ylog=False, scale=1.0,
                     pooled=False, linestyles=None):
    """
    Plot a straight line of best fit through the binned means, with 95% CI
    around the fitted mean line.

    Parameters
    ----------
    pooled=False
        Uses per-sample bin means to form each group's binned means.
        Usually better than pooled=True.
    scale : float
        Divide plotted y values by this (e.g. 1000 for Pa -> kPa).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    colors = colors or {}
    if linestyles is None:
        linestyles = {
            "control": "-",
            "d7": "--",
            "d14": ":"
        }

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # use bin centres in normalised position (%)
    x = (np.arange(nbins) + 0.5) * (100.0 / nbins)

    for st in order:
        means, _, ns = BinStats(binned_dict, st, PlotVariable, nbins, pooled=pooled)
        y = np.asarray(means, float) / scale

        keep = np.isfinite(x) & np.isfinite(y)
        x_fit = x[keep]
        y_fit = y[keep]

        if x_fit.size < 2:
            continue

        # linear regression
        m, c = np.polyfit(x_fit, y_fit, 1)
        y_hat = m * x_fit + c

        # 95% CI of fitted mean line
        n = x_fit.size
        xbar = np.mean(x_fit)
        sxx = np.sum((x_fit - xbar) ** 2)
        resid = y_fit - y_hat
        dof = n - 2

        if dof > 0 and sxx > 0:
            s_err = np.sqrt(np.sum(resid ** 2) / dof)
            tcrit = 1.96  # good approximation for 95% CI
            x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
            y_line = m * x_line + c
            se_line = s_err * np.sqrt((1 / n) + ((x_line - xbar) ** 2 / sxx))
            ci = tcrit * se_line
            lo = y_line - ci
            hi = y_line + ci
        else:
            x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
            y_line = m * x_line + c
            lo = y_line.copy()
            hi = y_line.copy()

        col = colors.get(st, None)
        ls = linestyles.get(st, "-")

        # binned means only
        # ax.plot(x_fit, y_fit, marker="o", linestyle="None", color=col, label=f"{st} means")
        ax.plot(x_line, y_line, color=col, linestyle=ls, linewidth=2.2, label=f"{st} fit")
        ax.fill_between(x_line, lo, hi, color=col, alpha=0.12)
        ax.plot(x_line, lo, color=col, linestyle=ls, linewidth=1.0, alpha=0.9)
        ax.plot(x_line, hi, color=col, linestyle=ls, linewidth=1.0, alpha=0.9)

        # console stats
        ss_res = np.sum((y_fit - y_hat) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        print(f"\n[{st}] best fit through binned means:")
        print(f"  n bins used = {n}")
        print(f"  slope = {m:.6f}")
        print(f"  intercept = {c:.6f}")
        print(f"  R² = {r2:.4f}")

    ax.set_xlabel("Normalised position (%)")
    # ax.set_ylabel(PlotVariable if scale == 1 else f"{PlotVariable} / {scale:g}")
    ax.set_ylabel(PlotVariable if scale == 1 else f"{PlotVariable} (kPa)")
    ax.set_title(title or f"Best fit through binned means: {PlotVariable}")

    if ylim is not None:
        ax.set_ylim(ylim)
    if ylog:
        ax.set_yscale("log")

    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()
    return fig, ax

def PlotSpatialCurves1x3(binned_dict, order, PlotVariable, colors=None,
                         title=None, scale=1.0,
                         window=18, grid_size=200, min_n=2,
                         linestyles=None, figsize=(14, 4.8),
                         fixed_ylim=(0, 10), print_stats=True):

    colors = colors or {}
    if linestyles is None:
        base_styles = ["-", "--", ":"]
        linestyles = {st: base_styles[i % len(base_styles)] for i, st in enumerate(order)}

    fig, axes = plt.subplots(1, len(order), figsize=figsize, squeeze=False)
    axes = axes[0]
    xgrid = np.linspace(0, 100, grid_size)

    def _sample_arrays(subtype):
        pts = _points_for(binned_dict, subtype)
        sample_ids = sorted({p.get("sample") for p in pts if p.get("sample") is not None})
        out = {}
        for sid in sample_ids:
            x = np.array([p.get("norm_pos", np.nan) for p in pts if p.get("sample") == sid], float)
            y = np.array([p.get(PlotVariable, np.nan) for p in pts if p.get("sample") == sid], float) / scale
            keep = np.isfinite(x) & np.isfinite(y)
            x, y = x[keep], y[keep]
            if x.size:
                out[sid] = (x, y)
        return out

    def _smooth_curve(sample_xy):
        means = np.full_like(xgrid, np.nan)
        lo = np.full_like(xgrid, np.nan)
        hi = np.full_like(xgrid, np.nan)

        for i, gx in enumerate(xgrid):
            vals = []
            for sx, sy in sample_xy.values():
                mask = np.abs(sx - gx) <= window
                v = sy[mask]
                v = v[np.isfinite(v)]
                if v.size:
                    vals.append(np.mean(v))

            vals = np.asarray(vals, float)
            vals = vals[np.isfinite(vals)]
            n = vals.size
            if n < min_n:
                continue

            mu = np.mean(vals)
            means[i] = mu

            if n == 1:
                lo[i] = mu
                hi[i] = mu
            else:
                sd = np.std(vals, ddof=1)
                se = sd / np.sqrt(n)
                ci = 1.96 * se
                lo[i] = mu - ci
                hi[i] = mu + ci

        return means, lo, hi

    for ax, st in zip(axes, order):
        col = colors.get(st, None)
        ls = linestyles.get(st, "-")
        sample_xy = _sample_arrays(st)
        mean_y, lo_y, hi_y = _smooth_curve(sample_xy)

        for sx, sy in sample_xy.values():
            order_idx = np.argsort(sx)
            ax.plot(sx[order_idx], sy[order_idx], color=col, alpha=0.15, linewidth=1)

        valid = np.isfinite(mean_y)
        ax.plot(xgrid[valid], mean_y[valid], color=col, linestyle=ls, linewidth=2.8)
        ax.fill_between(xgrid[valid], lo_y[valid], hi_y[valid], color=col, alpha=0.15)
        ax.plot(xgrid[valid], lo_y[valid], color=col, linestyle=ls, linewidth=0.9, alpha=0.85)
        ax.plot(xgrid[valid], hi_y[valid], color=col, linestyle=ls, linewidth=0.9, alpha=0.85)

        ax.set_title(st)
        ax.set_xlabel("Normalised position (%)")
        ax.set_ylabel(f"{PlotVariable} / {scale:g}")
        ax.grid(True, alpha=0.2)
        if fixed_ylim is not None:
            ax.set_ylim(*fixed_ylim)

        if print_stats:
            print(f"\n[{st}] {PlotVariable} smoothed curve:")
            valid_fit = np.isfinite(mean_y)
            if np.sum(valid_fit) >= 2:
                m, c = np.polyfit(xgrid[valid_fit], mean_y[valid_fit], 1)
                print(f"  overall smooth slope = {m:.6f}")
                print(f"  overall smooth intercept = {c:.6f}")
                print(f"  start mean ≈ {mean_y[valid_fit][0]:.6f}")
                print(f"  end mean ≈ {mean_y[valid_fit][-1]:.6f}")
                print(f"  Δ(0→100) ≈ {mean_y[valid_fit][-1] - mean_y[valid_fit][0]:.6f}")

    fig.suptitle(title or f"{PlotVariable} across normalised position", fontsize=14)
    plt.tight_layout()
    return fig, axes


    
def AddDerivedViscoVars(binned_dict):
    """
    Adds derived visco variables into each point dict:
      - visco_ratio = tau_Visco / mod_Hertz
      - visco_index = tau_Visco * mod_Hertz

    Notes
    -----
    tau_Visco : s
    mod_Hertz : Pa
    visco_ratio : s/Pa
    visco_index : Pa*s
    """
    for stype, entry in binned_dict.items():
        pts = entry.get("points", [])
        for p in pts:
            tau = p.get("tau_Visco", np.nan)
            mod = p.get("mod_Hertz", np.nan)

            tau = float(tau) if pd.notna(tau) else np.nan
            mod = float(mod) if pd.notna(mod) else np.nan

            if np.isfinite(tau) and np.isfinite(mod) and mod != 0:
                p["visco_ratio"] = tau / mod
            else:
                p["visco_ratio"] = np.nan

            if np.isfinite(tau) and np.isfinite(mod):
                p["visco_index"] = tau * mod
            else:
                p["visco_index"] = np.nan

    return binned_dict

def PlotDerivedVisco1x4(binned_dict, order, PlotVariable, colors=None,
                        title=None, window=18, grid_size=200, min_n=2,
                        linestyles=None, figsize=(14, 4.8),
                        fixed_ylim=None, print_stats=True):
    """
    Plot a derived visco variable across normalised position, one panel per subtype.

    PlotVariable:
        "visco_ratio" or "visco_index"
    """
    colors = colors or {}
    if linestyles is None:
        linestyles = {
            "control": "-",
            "d7": "--",
            "d14": ":",
            "d21": "-"
        }

    fig, axes = plt.subplots(1, len(order), figsize=figsize, squeeze=False)
    axes = axes[0]
    xgrid = np.linspace(0, 100, grid_size)

    def _sample_arrays(subtype):
        pts = _points_for(binned_dict, subtype)
        sample_ids = sorted({p.get("sample") for p in pts if p.get("sample") is not None})
        out = {}
        for sid in sample_ids:
            x = np.array([p.get("norm_pos", np.nan) for p in pts if p.get("sample") == sid], float)
            y = np.array([p.get(PlotVariable, np.nan) for p in pts if p.get("sample") == sid], float)
            keep = np.isfinite(x) & np.isfinite(y)
            x, y = x[keep], y[keep]
            if x.size:
                out[sid] = (x, y)
        return out

    def _smooth_curve(sample_xy):
        means = np.full_like(xgrid, np.nan)
        lo = np.full_like(xgrid, np.nan)
        hi = np.full_like(xgrid, np.nan)

        for i, gx in enumerate(xgrid):
            vals = []
            for sx, sy in sample_xy.values():
                mask = np.abs(sx - gx) <= window
                v = sy[mask]
                v = v[np.isfinite(v)]
                if v.size:
                    vals.append(np.mean(v))

            vals = np.asarray(vals, float)
            vals = vals[np.isfinite(vals)]
            n = vals.size
            if n < min_n:
                continue

            mu = np.mean(vals)
            means[i] = mu

            if n == 1:
                lo[i] = mu
                hi[i] = mu
            else:
                sd = np.std(vals, ddof=1)
                se = sd / np.sqrt(n)
                ci = 1.96 * se
                lo[i] = mu - ci
                hi[i] = mu + ci

        return means, lo, hi

    for ax, st in zip(axes, order):
        col = colors.get(st, None)
        ls = linestyles.get(st, "-")

        sample_xy = _sample_arrays(st)
        mean_y, lo_y, hi_y = _smooth_curve(sample_xy)

        for sx, sy in sample_xy.values():
            order_idx = np.argsort(sx)
            ax.plot(sx[order_idx], sy[order_idx], color=col, alpha=0.15, linewidth=1)

        valid = np.isfinite(mean_y)
        ax.plot(xgrid[valid], mean_y[valid], color=col, linestyle=ls, linewidth=2.8)
        ax.fill_between(xgrid[valid], lo_y[valid], hi_y[valid], color=col, alpha=0.15)
        ax.plot(xgrid[valid], lo_y[valid], color=col, linestyle=ls, linewidth=0.9, alpha=0.85)
        ax.plot(xgrid[valid], hi_y[valid], color=col, linestyle=ls, linewidth=0.9, alpha=0.85)

        ax.set_title(st)
        ax.set_xlabel("Normalised position (%)")
        ax.set_ylabel(PlotVariable)
        ax.grid(True, alpha=0.2)

        if fixed_ylim is not None:
            ax.set_ylim(*fixed_ylim)

        if print_stats:
            print(f"\n[{st}] {PlotVariable} smoothed curve:")
            valid_fit = np.isfinite(mean_y)
            if np.sum(valid_fit) >= 2:
                m, c = np.polyfit(xgrid[valid_fit], mean_y[valid_fit], 1)
                print(f"  overall smooth slope = {m:.6e}")
                print(f"  overall smooth intercept = {c:.6e}")
                print(f"  start mean ≈ {mean_y[valid_fit][0]:.6e}")
                print(f"  end mean ≈ {mean_y[valid_fit][-1]:.6e}")
                print(f"  Δ(0→100) ≈ {mean_y[valid_fit][-1] - mean_y[valid_fit][0]:.6e}")

    fig.suptitle(title or f"{PlotVariable} across normalised position", fontsize=14)
    plt.tight_layout()
    return fig, axes

# ---------------------------------------------------------------------------
# ANALYSE DATA
# ---------------------------------------------------------------------------

def PCAVisualSuite(binned_dict, order, pca_vars,
                   colors=None,
                   markers=None,
                   standardise=True,
                   cmap="jet",
                   alpha=0.35,
                   s=12,
                   ellipse_conf=0.95,
                   dbg=False,
                   PCA1log = False,
                   PCA2log = False,
                   PCA3log = False):
    """
    Builds PCA from ALL layer-filtered points in binned_dict[subtype]["points"].

    Produces:
      1) PCA score scatter (PC1 vs PC2) coloured by subtype
      2) Biplot (scores + loading arrows)
      3) Position vs PC1 plot (norm_pos vs PC1)
      4) Centroids + confidence ellipses (PC1 vs PC2)
      5) PC1 boxplot + PC2 boxplot (separate figs)
      6) PCA scatter coloured by spatial position (norm_pos), subtype by marker
      7) Density contours per subtype (KDE on PC1/PC2)
      8) Scree bar chart

    Notes:
      - expects each point dict to contain your canonical variables (e.g. "tau_Visco", "Einf_Visco", etc.)
      - for position, uses "norm_pos" if present else computes from line-wise normalisation via _per_line_norm_positions(pts)
    """

    colors = colors or {}
    markers = markers or {st: mk for st, mk in zip(order, ["o", "x", "^", "s", "D", "v", "P", "*"])}

    # ------------------------- helpers -------------------------
    # ------------------------- helpers -------------------------
    def _pts(st):
        return binned_dict.get(st, {}).get("points", [])
    
    def _resolve(v, pts):
        # allow user passing either canonical key or csv-header-like via VAR_MAP
        if pts and v in pts[0]:
            return v
        return VAR_MAP.get(v, v)
    
    def _var_array_for_pts(pts, v):
        # special case: spatial position
        if v in ("norm", "norm_pos", "position", "pos"):
            if pts and ("norm_pos" in pts[0]):
                return np.array([p.get("norm_pos", np.nan) for p in pts], float)
            # fallback: compute per-line normed positions (0-100) from the points list
            return _per_line_norm_positions(pts)
    
        k = _resolve(v, pts)
        return np.array([p.get(k, np.nan) for p in pts], float)
    
    def _linfit_r2(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]; y = y[m]
        n = int(x.size)
        if n < 2:
            return np.nan, np.nan, np.nan, n
        A = np.vstack([x, np.ones(n)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        yhat = slope*x + intercept
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
        return float(slope), float(intercept), float(r2), n
    
    def _linfit_with_se(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        n = int(x.size)
        if n < 3:
            return np.nan, np.nan, np.nan, np.nan, n
    
        A = np.vstack([x, np.ones(n)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        yhat = slope * x + intercept
    
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    
        dof = n - 2
        sxx = np.sum((x - np.mean(x)) ** 2)
        if dof <= 0 or sxx <= 0:
            se_slope = np.nan
        else:
            mse = ss_res / dof
            se_slope = np.sqrt(mse / sxx)
    
        return float(slope), float(intercept), float(r2), float(se_slope), n

    def _chi2_2d_for_conf(conf):
        # 2 dof chi2 quantiles (hard-coded common values to avoid extra deps)
        # 95% -> 5.991, 90% -> 4.605, 99% -> 9.210
        if conf >= 0.99: return 9.210
        if conf >= 0.95: return 5.991
        if conf >= 0.90: return 4.605
        # fallback approx
        return 5.991

    def _ellipse_from_cov(mean, cov, conf=0.95, **kwargs):
        # eigen-decomp to get orientation + axes lengths
        vals, vecs = np.linalg.eigh(cov)
        order_e = np.argsort(vals)[::-1]
        vals = vals[order_e]
        vecs = vecs[:, order_e]

        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        chi2 = _chi2_2d_for_conf(conf)
        width, height = 2.0 * np.sqrt(vals * chi2)
        return Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)

    # ------------------------- build matrix -------------------------
    X_list, G_list, P_list, S_list = [], [], [],[]  # P_list = norm_pos for colouring
    resolved_keys = None

    for st in order:
        pts = _pts(st)
        if not pts:
            continue

        # --- optional log transforms ---
        log_flags = [PCA1log, PCA2log, PCA3log]
        
        cols = []
        for i, v in enumerate(pca_vars):
            arr = _var_array_for_pts(pts, v)
        
            if i < len(log_flags) and log_flags[i]:
                arr = np.asarray(arr, float)
        
                # safe log: remove non-positive values
                mask = arr > 0
                arr = np.where(mask, np.log(arr), np.nan)
        
            cols.append(arr)
        mat = np.column_stack(cols)

        # also store a norm_pos vector for plots that need it
        norm_pos = _var_array_for_pts(pts, "norm_pos")
        sample_ids = np.array([p.get("sample", np.nan) for p in pts], dtype=object)


        m = np.all(np.isfinite(mat), axis=1) & np.isfinite(norm_pos)
        mat = mat[m]
        norm_pos = norm_pos[m]
        sample_ids = sample_ids[m]
        

        if mat.shape[0] == 0:
            if dbg: print(f"[PCA] {st}: no complete rows for {pca_vars}")
            continue

        X_list.append(mat)
        G_list.append(np.full(mat.shape[0], st, dtype=object))
        P_list.append(norm_pos)
        S_list.append(sample_ids)

        if resolved_keys is None:
            resolved_keys = [("norm_pos" if v in ("norm", "norm_pos", "position", "pos") else _resolve(v, pts)) for v in pca_vars]

    figs = {}
    results = {"pca_vars": tuple(pca_vars), "resolved_keys": resolved_keys, "standardised": bool(standardise)}

    if not X_list:
        print(f"[PCAVisualSuite] No complete rows for selected vars: {pca_vars}")
        return results, figs

    X = np.vstack(X_list)
    groups_vec = np.concatenate(G_list)
    norm_pos_all = np.concatenate(P_list)
    sample_ids_all = np.concatenate(S_list)
    
    # standardise
    if standardise:
        mu = np.mean(X, axis=0)
        sd = np.std(X, axis=0, ddof=1)
        sd = np.where(sd == 0, 1.0, sd)
        Xz = (X - mu) / sd
    else:
        mu = np.zeros(X.shape[1], float)
        sd = np.ones (X.shape[1], float)
        Xz = X.copy()

    # PCA via SVD
    Xc = Xz - np.mean(Xz, axis=0)
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    scores = U * S  # (n, p)
    loadings = VT.T  # (p, p)

    # explained variance
    var = (S**2) / (Xc.shape[0] - 1)
    evr = var / np.sum(var)

    results.update({
        "scores": scores,
        "groups": groups_vec,
        "norm_pos": norm_pos_all,
        "sample_ids": sample_ids_all,
        "loadings": loadings,
        "explained_variance_ratio": evr,
        "mu": mu, "sd": sd
    })

    # convenient
    pc1, pc2 = scores[:, 0], scores[:, 1]
    evr2 = evr[:2]

    print("\n=== PCA summary ===")
    print(f"  Vars: {pca_vars}")
    print(f"  Explained variance: PC1={evr2[0]*100:.2f}% | PC2={evr2[1]*100:.2f}%")
    for i, v in enumerate(pca_vars):
        print(f"  Loading {v}:  PC1={loadings[i,0]: .4f}  PC2={loadings[i,1]: .4f}")

    # =====================================================================================
    # 1) PCA score scatter (PC1 vs PC2) coloured by subtype
    # =====================================================================================
    fig1, ax1 = plt.subplots()
    for st in order:
        m = (groups_vec == st)
        if not np.any(m): 
            continue
        ax1.scatter(pc1[m], pc2[m], s=s, alpha=alpha, color=colors.get(st, None), label=st)
    ax1.set_xlabel(f"PC1 ({evr2[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({evr2[1]*100:.1f}%)")
    ax1.set_title(f"PCA scores: {pca_vars}")
    ax1.grid(True)
    ax1.legend()
    plt.tight_layout()
    figs["scores"] = fig1
    
    

    # # =====================================================================================
    # # 2) Biplot (scores + loadings)
    # # =====================================================================================
    # fig2, ax2 = plt.subplots()
    # # scores
    # for st in order:
    #     m = (groups_vec == st)
    #     if not np.any(m): 
    #         continue
    #     ax2.scatter(pc1[m], pc2[m], s=s, alpha=alpha, color=colors.get(st, None), label=st)

    # # scale loadings to look sensible in score space
    # # (use a fraction of score spread)
    # span1 = np.nanpercentile(np.abs(pc1), 95)
    # span2 = np.nanpercentile(np.abs(pc2), 95)
    # L = loadings[:, :2]
    # Lscale = 0.35 * np.array([span1, span2])

    # for i, v in enumerate(pca_vars):
    #     ax2.arrow(0, 0, L[i, 0]*Lscale[0], L[i, 1]*Lscale[1],
    #               head_width=0.04*min(span1, span2),
    #               length_includes_head=True,
    #               linewidth=2)
    #     ax2.text(L[i, 0]*Lscale[0]*1.08, L[i, 1]*Lscale[1]*1.08, str(v), fontsize=10)

    # ax2.axhline(0, linewidth=1)
    # ax2.axvline(0, linewidth=1)
    # ax2.set_xlabel(f"PC1 ({evr2[0]*100:.1f}%)")
    # ax2.set_ylabel(f"PC2 ({evr2[1]*100:.1f}%)")
    # ax2.set_title("PCA biplot (scores + loadings)")
    # ax2.grid(True)
    # ax2.legend()
    # plt.tight_layout()
    # figs["biplot"] = fig2

   # =====================================================================================
    # 3) Position vs PC1
    # =====================================================================================
    fig3, ax3 = plt.subplots()
    
    print("\n=== PC1 vs position (linear fit per group) ===")
    
    for st in order:
        m = (groups_vec == st)
        if not np.any(m):
            continue
    
        x = norm_pos_all[m]
        y = pc1[m]
    
        col = colors.get(st, None)
    
        ax3.scatter(x, y, s=s, alpha=alpha, color=col, label=st)
    
        # ---- regression ----
        slope, intercept, r2, n = _linfit_r2(x, y)
        print(f"[PC1~pos] {st}: n={n} slope={slope:.4g} intercept={intercept:.4g} R2={r2:.4g}")
        # ---- residuals (PC1) ----
        if np.isfinite(slope) and n >= 2:
            yhat = slope*x + intercept
            resid = y - yhat
        
            # store (optional)
            results.setdefault("pc1_fit", {})[st] = {
                "slope": slope, "intercept": intercept, "r2": r2, "n": n,
                "rmse": float(np.sqrt(np.mean(resid**2))),
                "mae":  float(np.mean(np.abs(resid))),
            }
        
            # # residual plot (vs position)
            # figr, axr = plt.subplots()
            # axr.scatter(x, resid, s=s, alpha=alpha, color=col, label=st)
            # axr.axhline(0, linewidth=1)
            # axr.set_xlabel("Normalised position (0–100%)")
            # axr.set_ylabel("Residual (PC1 score)")
            # axr.set_title(f"PC1 residuals vs position ({st})")
            # axr.grid(True)
            # axr.legend()
            # plt.tight_layout()
            # figs[f"pc1_resid_{st}"] = figr
    
        if np.isfinite(slope):
            xx = np.linspace(0, 100, 200)
            ax3.plot(xx, slope*xx + intercept, linewidth=2, color=col)
    
    ax3.set_xlabel("Normalised position (0–100%)")
    ax3.set_ylabel(f"PC1 score ({evr2[0]*100:.1f}%)")
    ax3.set_title("PC1 vs spatial position")
    ax3.grid(True)
    ax3.legend()
    plt.tight_layout()
    
    figs["pc1_vs_pos"] = fig3
    
    # =====================================================================================
    # 3b) Position vs PC2
    # =====================================================================================
    fig3b, ax3b = plt.subplots()
    
    print("\n=== PC2 vs position (linear fit per group) ===")
    
    for st in order:
        m = (groups_vec == st)
        if not np.any(m):
            continue
    
        x = norm_pos_all[m]
        y = pc2[m]
    
        col = colors.get(st, None)
    
        ax3b.scatter(x, y, s=s, alpha=alpha, color=col, label=st)
    
        # ---- regression ----
        slope, intercept, r2, n = _linfit_r2(x, y)
        print(f"[PC2~pos] {st}: n={n} slope={slope:.4g} intercept={intercept:.4g} R2={r2:.4g}")
        # ---- residuals (PC2) ----
        if np.isfinite(slope) and n >= 2:
            yhat = slope*x + intercept
            resid = y - yhat
        
            results.setdefault("pc2_fit", {})[st] = {
                "slope": slope, "intercept": intercept, "r2": r2, "n": n,
                "rmse": float(np.sqrt(np.mean(resid**2))),
                "mae":  float(np.mean(np.abs(resid))),
            }
        
            figr, axr = plt.subplots()
            axr.scatter(x, resid, s=s, alpha=alpha, color=col, label=st)
            axr.axhline(0, linewidth=1)
            axr.set_xlabel("Normalised position (0–100%)")
            axr.set_ylabel("Residual (PC2 score)")
            axr.set_title(f"PC2 residuals vs position ({st})")
            axr.grid(True)
            axr.legend()
            plt.tight_layout()
            figs[f"pc2_resid_{st}"] = figr
    
        if np.isfinite(slope):
            xx = np.linspace(0, 100, 200)
            ax3b.plot(xx, slope*xx + intercept, linewidth=2, color=col)
    
    ax3b.set_xlabel("Normalised position (0–100%)")
    ax3b.set_ylabel(f"PC2 score ({evr2[1]*100:.1f}%)")
    ax3b.set_title("PC2 vs spatial position")
    ax3b.grid(True)
    ax3b.legend()
    plt.tight_layout()
    
    figs["pc2_vs_pos"] = fig3b
    
    # =====================================================================================
    # 3c) Slope summary plots for PC1 and PC2 vs position
    # =====================================================================================
    pc1_summ = []
    pc2_summ = []
    
    for st in order:
        m = (groups_vec == st)
        if not np.any(m):
            continue
    
        x = norm_pos_all[m]
    
        # PC1
        y1 = pc1[m]
        slope1, intercept1, r2_1, se1, n1 = _linfit_with_se(x, y1)
        pc1_summ.append({
            "group": st,
            "slope": slope1,
            "intercept": intercept1,
            "r2": r2_1,
            "se": se1,
            "n": n1,
            "ci_lo": slope1 - 1.96 * se1 if np.isfinite(se1) else np.nan,
            "ci_hi": slope1 + 1.96 * se1 if np.isfinite(se1) else np.nan,
        })
    
        # PC2
        y2 = pc2[m]
        slope2, intercept2, r2_2, se2, n2 = _linfit_with_se(x, y2)
        pc2_summ.append({
            "group": st,
            "slope": slope2,
            "intercept": intercept2,
            "r2": r2_2,
            "se": se2,
            "n": n2,
            "ci_lo": slope2 - 1.96 * se2 if np.isfinite(se2) else np.nan,
            "ci_hi": slope2 + 1.96 * se2 if np.isfinite(se2) else np.nan,
        })
    
    results["pc1_slope_summary"] = pc1_summ
    results["pc2_slope_summary"] = pc2_summ
    
    # console print
    print("\n=== PC1 slope summary ===")
    for row in pc1_summ:
        print(f"{row['group']:>8} | slope={row['slope']:.5f} | SE={row['se']:.5f} | "
              f"95% CI=[{row['ci_lo']:.5f}, {row['ci_hi']:.5f}] | R²={row['r2']:.4f} | n={row['n']}")
    
    print("\n=== PC2 slope summary ===")
    for row in pc2_summ:
        print(f"{row['group']:>8} | slope={row['slope']:.5f} | SE={row['se']:.5f} | "
              f"95% CI=[{row['ci_lo']:.5f}, {row['ci_hi']:.5f}] | R²={row['r2']:.4f} | n={row['n']}")
    
    # PC1 slope summary plot
    fig_pc1s, ax_pc1s = plt.subplots()
    xidx = np.arange(len(pc1_summ))
    slopes = np.array([r["slope"] for r in pc1_summ], float)
    ci_lo = np.array([r["ci_lo"] for r in pc1_summ], float)
    ci_hi = np.array([r["ci_hi"] for r in pc1_summ], float)
    labels = [r["group"] for r in pc1_summ]
    
    cols = [colors.get(g, None) for g in labels]
    ax_pc1s.scatter(xidx, slopes, s=70, c=cols, zorder=3)
    ax_pc1s.errorbar(
        xidx, slopes,
        yerr=[slopes - ci_lo, ci_hi - slopes],
        fmt="none", ecolor="black", capsize=4, linewidth=1.5, zorder=2
    )
    ax_pc1s.axhline(0, linewidth=1)
    ax_pc1s.set_xticks(xidx)
    ax_pc1s.set_xticklabels(labels)
    ax_pc1s.set_ylabel("Slope of PC1 vs position")
    ax_pc1s.set_title("PC1 slope summary (±95% CI)")
    ax_pc1s.grid(True, axis="y")
    plt.tight_layout()
    figs["pc1_slope_summary"] = fig_pc1s
    
    # PC2 slope summary plot
    fig_pc2s, ax_pc2s = plt.subplots()
    xidx = np.arange(len(pc2_summ))
    slopes = np.array([r["slope"] for r in pc2_summ], float)
    ci_lo = np.array([r["ci_lo"] for r in pc2_summ], float)
    ci_hi = np.array([r["ci_hi"] for r in pc2_summ], float)
    labels = [r["group"] for r in pc2_summ]
    
    cols = [colors.get(g, None) for g in labels]
    ax_pc2s.scatter(xidx, slopes, s=70, c=cols, zorder=3)
    ax_pc2s.errorbar(
        xidx, slopes,
        yerr=[slopes - ci_lo, ci_hi - slopes],
        fmt="none", ecolor="black", capsize=4, linewidth=1.5, zorder=2
    )
    ax_pc2s.axhline(0, linewidth=1)
    ax_pc2s.set_xticks(xidx)
    ax_pc2s.set_xticklabels(labels)
    ax_pc2s.set_ylabel("Slope of PC2 vs position")
    ax_pc2s.set_title("PC2 slope summary (±95% CI)")
    ax_pc2s.grid(True, axis="y")
    plt.tight_layout()
    figs["pc2_slope_summary"] = fig_pc2s

    # # =====================================================================================
    # # 4) Group centroids + ellipses (PC1 vs PC2)
    # # =====================================================================================
    # fig4, ax4 = plt.subplots()
    # for st in order:
    #     m = (groups_vec == st)
    #     if not np.any(m):
    #         continue
    #     col = colors.get(st, None)

    #     xs = pc1[m]; ys = pc2[m]
    #     ax4.scatter(xs, ys, s=s, alpha=0.18, color=col)

    #     mean = np.array([np.mean(xs), np.mean(ys)])
    #     cov = np.cov(np.vstack([xs, ys]))
    #     ax4.scatter(mean[0], mean[1], s=120, color=col, marker=markers.get(st, "o"), edgecolor="k", linewidth=1.0, label=st)

    #     if np.all(np.isfinite(cov)) and cov.shape == (2, 2) and np.linalg.det(cov) > 0:
    #         ell = _ellipse_from_cov(mean, cov, conf=ellipse_conf, fill=False, color=col, linewidth=2.0)
    #         ax4.add_patch(ell)

    # ax4.set_xlabel(f"PC1 ({evr2[0]*100:.1f}%)")
    # ax4.set_ylabel(f"PC2 ({evr2[1]*100:.1f}%)")
    # ax4.set_title(f"Centroids + {int(ellipse_conf*100)}% ellipses")
    # ax4.grid(True)
    # ax4.legend()
    # plt.tight_layout()
    # figs["centroids_ellipses"] = fig4

    # # =====================================================================================
    # # 5) PC1 + PC2 boxplots per group (separate figs)
    # # =====================================================================================
    # def _boxplot(figkey, vec, ylabel):
    #     fig, ax = plt.subplots()
    #     data = []
    #     labs = []
    #     for st in order:
    #         m = (groups_vec == st)
    #         v = vec[m]
    #         v = v[np.isfinite(v)]
    #         data.append(v)
    #         labs.append(st)
    
    #     bp = ax.boxplot(data, labels=labs, showfliers=True, patch_artist=True)
    
    #     # colour boxes by group colour (fill)
    #     for patch, st in zip(bp["boxes"], labs):
    #         col = colors.get(st, None)
    #         if col is not None:
    #             patch.set_facecolor(col)
    #             patch.set_alpha(0.35)
    #             patch.set_edgecolor(col)
    
    #     ax.set_ylabel(ylabel)
    #     ax.set_title(ylabel + " by group")
    #     ax.grid(True, axis="y")
    #     plt.tight_layout()
    #     figs[figkey] = fig
    
    # _boxplot("pc1_box", pc1, f"PC1 score ({evr2[0]*100:.1f}%)")
    # _boxplot("pc2_box", pc2, f"PC2 score ({evr2[1]*100:.1f}%)")

    # =====================================================================================
    # 6) Scatter coloured by spatial position; subtype by marker
    # =====================================================================================
    fig6, ax6 = plt.subplots()
    
    norm = plt.Normalize(0, 100)
    for st in order:
        m = (groups_vec == st)
        if not np.any(m):
            continue
        ax6.scatter(pc1[m], pc2[m],
                    c=norm_pos_all[m],
                    cmap=cmap,
                    norm=norm,
                    s=s,
                    alpha=0.55,
                    marker=markers.get(st, "o"),
                    label=st)
    
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax6)
    cb.set_label("Normalised position (0–100%)")
    
    ax6.set_xlabel(f"PC1 ({evr2[0]*100:.1f}%)")
    ax6.set_ylabel(f"PC2 ({evr2[1]*100:.1f}%)")
    ax6.set_title("PCA coloured by spatial position (subtype by marker)")
    ax6.grid(True)
    ax6.legend()
    plt.tight_layout()
    figs["pos_coloured_scores"] = fig6
    
    # =====================================================================================
    # 6b) Faceted PCA coloured by spatial position (one subtype per panel)
    # =====================================================================================
    n_panels = len(order)
    nrows, ncols = 2, 2
    fig6b, axes6b = plt.subplots(nrows, ncols, figsize=(10, 8), squeeze=False)
    
    # shared limits so panels are directly comparable
    xpad = 0.05 * (np.nanmax(pc1) - np.nanmin(pc1)) if np.nanmax(pc1) > np.nanmin(pc1) else 1.0
    ypad = 0.05 * (np.nanmax(pc2) - np.nanmin(pc2)) if np.nanmax(pc2) > np.nanmin(pc2) else 1.0
    xlim = (np.nanmin(pc1) - xpad, np.nanmax(pc1) + xpad)
    ylim = (np.nanmin(pc2) - ypad, np.nanmax(pc2) + ypad)
    
    norm = plt.Normalize(0, 100)
    last_sc = None
    
    for i, st in enumerate(order[: nrows * ncols]):
        ax = axes6b[i // ncols][i % ncols]
        m = (groups_vec == st)
    
        if np.any(m):
            last_sc = ax.scatter(
                pc1[m], pc2[m],
                c=norm_pos_all[m],
                cmap=cmap,
                norm=norm,
                s=s,
                alpha=0.65,
                marker=markers.get(st, "o")
            )
    
            # optional centroid trajectory through depth bands
            # gives a clearer "path" through PCA space
            bins_pos = np.linspace(0, 100, 11)
            cx, cy = [], []
            for b0, b1 in zip(bins_pos[:-1], bins_pos[1:]):
                mb = m & (norm_pos_all >= b0) & (norm_pos_all < b1 if b1 < 100 else norm_pos_all <= b1)
                if np.any(mb):
                    cx.append(np.mean(pc1[mb]))
                    cy.append(np.mean(pc2[mb]))
                else:
                    cx.append(np.nan)
                    cy.append(np.nan)
    
            cx = np.asarray(cx, float)
            cy = np.asarray(cy, float)
            good = np.isfinite(cx) & np.isfinite(cy)
            # if np.sum(good) >= 2:
                # ax.plot(cx[good], cy[good], color=colors.get(st, None), linewidth=1.5, alpha=0.9)
    
        ax.set_title(st)
        ax.set_xlabel(f"PC1 ({evr2[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({evr2[1]*100:.1f}%)")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True)
    
    # hide unused panels if fewer than 4 groups
    for j in range(n_panels, nrows * ncols):
        axes6b[j // ncols][j % ncols].axis("off")
    
    if last_sc is not None:
        # make space on the right for colourbar
        fig6b.subplots_adjust(right=0.80)
    
        # add a dedicated colourbar axis
        cax = fig6b.add_axes([1, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig6b.colorbar(last_sc, cax=cax)
        cbar.set_label("Normalised position (0–100%)")
    
    fig6b.suptitle("PCA coloured by spatial position (faceted by subtype)")
    plt.tight_layout()
    figs["pos_coloured_scores_faceted"] = fig6b

    # # =====================================================================================
    # # 7) Density contours per group (KDE on PC1/PC2)
    # # =====================================================================================
    # fig7, ax7 = plt.subplots()
    # # grid bounds
    # xmin, xmax = np.nanmin(pc1), np.nanmax(pc1)
    # ymin, ymax = np.nanmin(pc2), np.nanmax(pc2)
    # pad_x = 0.08 * (xmax - xmin) if xmax > xmin else 1.0
    # pad_y = 0.08 * (ymax - ymin) if ymax > ymin else 1.0
    # xmin -= pad_x; xmax += pad_x
    # ymin -= pad_y; ymax += pad_y
    
    # xx, yy = np.meshgrid(np.linspace(xmin, xmax, 160), np.linspace(ymin, ymax, 160))
    # grid = np.vstack([xx.ravel(), yy.ravel()])
    
    # for st in order:
    #     m = (groups_vec == st)
    #     if not np.any(m):
    #         continue
    #     xs = pc1[m]; ys = pc2[m]
    #     xs = xs[np.isfinite(xs)]; ys = ys[np.isfinite(ys)]
    #     if xs.size < 10:
    #         continue
    
    #     col = colors.get(st, None)
    #     kde = gaussian_kde(np.vstack([xs, ys]))
    #     zz = kde(grid).reshape(xx.shape)
    
    #     zmin, zmax = float(np.nanmin(zz)), float(np.nanmax(zz))
    #     if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
    #         continue
    
    #     # increasing contour levels (fixes "levels must be increasing")
    #     levels = np.linspace(zmin + 1e-12, zmax, 6)[1:]
    #     ax7.contour(xx, yy, zz, levels=levels,
    #                 colors=[col] if col is not None else None,
    #                 linewidths=1.8)
    #     ax7.scatter(xs, ys, s=6, alpha=0.12, color=col, label=st)
    
    # ax7.set_xlabel(f"PC1 ({evr2[0]*100:.1f}%)")
    # ax7.set_ylabel(f"PC2 ({evr2[1]*100:.1f}%)")
    # ax7.set_title("PC density contours per group")
    # ax7.grid(True)
    # ax7.legend()
    # plt.tight_layout()
    # figs["density_contours"] = fig7

    # # =====================================================================================
    # # 8) Scree bar chart
    # # =====================================================================================
    # fig8, ax8 = plt.subplots()
    # k = min(len(evr), 10)
    # ax8.bar(np.arange(1, k+1), evr[:k] * 100.0)
    # ax8.set_xlabel("Principal component")
    # ax8.set_ylabel("Explained variance (%)")
    # ax8.set_title("Scree plot")
    # ax8.set_xticks(np.arange(1, k+1))
    # ax8.grid(True, axis="y")
    # plt.tight_layout()
    # figs["scree"] = fig8
    
    # =====================================================================================
    # 9) PC1 and PC2 slope per sample, summarised across samples
    # =====================================================================================
    def _fit_sample_slope(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size < 3:
            return np.nan
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
    
    def _mean_ci(vals):
        vals = np.asarray(vals, float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return np.nan, np.nan, np.nan, 0
        mu = np.mean(vals)
        if vals.size == 1:
            return mu, mu, mu, 1
        sd = np.std(vals, ddof=1)
        se = sd / np.sqrt(vals.size)
        ci = 1.96 * se
        return float(mu), float(mu - ci), float(mu + ci), int(vals.size)
    
    pc1_sample_slopes = {}
    pc2_sample_slopes = {}
    
    print("\n=== PC1 per-sample slope summary ===")
    for st in order:
        m_group = (groups_vec == st)
        if not np.any(m_group):
            continue
    
        sids = np.unique(sample_ids_all[m_group])
        slopes = []
    
        print(f"\n[{st}] rows retained per sample for PCA:")
        for sid in sids:
            m = m_group & (sample_ids_all == sid)
    
            # debug: how many PCA-valid rows survived for this sample?
            n_rows = int(np.sum(m))
            print(f"  {sid}: {n_rows} PCA-valid rows")
    
            slope = _fit_sample_slope(norm_pos_all[m], pc1[m])
    
            if np.isfinite(slope):
                print(f"    PC1 slope = {slope:.6f}")
                slopes.append(slope)
            else:
                print(f"    PC1 slope skipped (too few valid rows)")
    
        slopes = np.asarray(slopes, float)
        pc1_sample_slopes[st] = slopes
    
        mu, lo, hi, n = _mean_ci(slopes)
        print(f"{st:>8} | n={n} | mean slope={mu:.5f} | 95% CI=[{lo:.5f}, {hi:.5f}]")
    
    print("\n=== PC2 per-sample slope summary ===")
    for st in order:
        m_group = (groups_vec == st)
        if not np.any(m_group):
            continue
    
        sids = np.unique(sample_ids_all[m_group])
        slopes = []
    
        print(f"\n[{st}] rows retained per sample for PCA:")
        for sid in sids:
            m = m_group & (sample_ids_all == sid)
    
            # debug: how many PCA-valid rows survived for this sample?
            n_rows = int(np.sum(m))
            print(f"  {sid}: {n_rows} PCA-valid rows")
    
            slope = _fit_sample_slope(norm_pos_all[m], pc2[m])
    
            if np.isfinite(slope):
                print(f"    PC2 slope = {slope:.6f}")
                slopes.append(slope)
            else:
                print(f"    PC2 slope skipped (too few valid rows)")
    
        slopes = np.asarray(slopes, float)
        pc2_sample_slopes[st] = slopes
    
        mu, lo, hi, n = _mean_ci(slopes)
        print(f"{st:>8} | n={n} | mean slope={mu:.5f} | 95% CI=[{lo:.5f}, {hi:.5f}]")
    
    results["pc1_sample_slopes"] = pc1_sample_slopes
    results["pc2_sample_slopes"] = pc2_sample_slopes
    
    # ---- plot PC1 per-sample slopes ----
    fig9a, ax9a = plt.subplots()
    xidx = np.arange(len(order))
    
    for i, st in enumerate(order):
        vals = pc1_sample_slopes.get(st, np.array([]))
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
    
        jitter = np.random.normal(0, 0.05, size=vals.size)
        ax9a.scatter(np.full(vals.size, xidx[i]) + jitter, vals,
                     color=colors.get(st, None), s=s*3 if s is not None else 35, alpha=0.75)
    
        mu, lo, hi, n = _mean_ci(vals)
        ax9a.errorbar(xidx[i], mu,
                      yerr=[[mu - lo], [hi - mu]],
                      fmt="o", color="black", capsize=4, linewidth=1.8)
    
    ax9a.axhline(0, linewidth=1)
    ax9a.set_xticks(xidx)
    ax9a.set_xticklabels(order)
    ax9a.set_ylabel("Slope of PC1 vs position")
    ax9a.set_title("PC1 slope per sample (±95% CI)")
    ax9a.grid(True, axis="y")
    plt.tight_layout()
    figs["pc1_sample_slope_summary"] = fig9a
    
    # ---- plot PC2 per-sample slopes ----
    fig9b, ax9b = plt.subplots()
    xidx = np.arange(len(order))
    
    for i, st in enumerate(order):
        vals = pc2_sample_slopes.get(st, np.array([]))
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
    
        jitter = np.random.normal(0, 0.05, size=vals.size)
        ax9b.scatter(np.full(vals.size, xidx[i]) + jitter, vals,
                     color=colors.get(st, None), s=s*3 if s is not None else 35, alpha=0.75)
    
        mu, lo, hi, n = _mean_ci(vals)
        ax9b.errorbar(xidx[i], mu,
                      yerr=[[mu - lo], [hi - mu]],
                      fmt="o", color="black", capsize=4, linewidth=1.8)
    
    ax9b.axhline(0, linewidth=1)
    ax9b.set_xticks(xidx)
    ax9b.set_xticklabels(order)
    ax9b.set_ylabel("Slope of PC2 vs position")
    ax9b.set_title("PC2 slope per sample (±95% CI)")
    ax9b.grid(True, axis="y")
    plt.tight_layout()
    figs["pc2_sample_slope_summary"] = fig9b
        
    
    

    return results, figs


def PlotPCASlopeBars(results, order, colors=None, title=None):
    """
    Grouped bar chart:
      x = subtype
      bars = PC1 slope and PC2 slope side by side
      error bars = 95% CI from PCAVisualSuite summaries
    """
    colors = colors or {}

    pc1 = {row["group"]: row for row in results.get("pc1_slope_summary", [])}
    pc2 = {row["group"]: row for row in results.get("pc2_slope_summary", [])}

    x = np.arange(len(order))
    w = 0.34

    fig, ax = plt.subplots(figsize=(8, 5.5))

    pc1_means = []
    pc1_err_lo = []
    pc1_err_hi = []

    pc2_means = []
    pc2_err_lo = []
    pc2_err_hi = []

    for st in order:
        r1 = pc1.get(st, {})
        m1 = r1.get("slope", np.nan)
        lo1 = r1.get("ci_lo", np.nan)
        hi1 = r1.get("ci_hi", np.nan)

        r2 = pc2.get(st, {})
        m2 = r2.get("slope", np.nan)
        lo2 = r2.get("ci_lo", np.nan)
        hi2 = r2.get("ci_hi", np.nan)

        pc1_means.append(m1)
        pc1_err_lo.append(m1 - lo1 if np.isfinite(m1) and np.isfinite(lo1) else np.nan)
        pc1_err_hi.append(hi1 - m1 if np.isfinite(m1) and np.isfinite(hi1) else np.nan)

        pc2_means.append(m2)
        pc2_err_lo.append(m2 - lo2 if np.isfinite(m2) and np.isfinite(lo2) else np.nan)
        pc2_err_hi.append(hi2 - m2 if np.isfinite(m2) and np.isfinite(hi2) else np.nan)

    pc1_means = np.asarray(pc1_means, float)
    pc1_err_lo = np.asarray(pc1_err_lo, float)
    pc1_err_hi = np.asarray(pc1_err_hi, float)

    pc2_means = np.asarray(pc2_means, float)
    pc2_err_lo = np.asarray(pc2_err_lo, float)
    pc2_err_hi = np.asarray(pc2_err_hi, float)

    # PC1 bars: coloured by subtype
    pc1_cols = [colors.get(st, "grey") for st in order]
    ax.bar(x - w/2, pc1_means, width=w, color=pc1_cols, alpha=0.75, label="PC1")
    ax.errorbar(x - w/2, pc1_means, yerr=[pc1_err_lo, pc1_err_hi],
                fmt="none", ecolor="black", capsize=4, linewidth=1.3)

    # PC2 bars: same colours but hatched so they are visually distinct
    ax.bar(x + w/2, pc2_means, width=w, color=pc1_cols, alpha=0.45,
           hatch="//", label="PC2")
    ax.errorbar(x + w/2, pc2_means, yerr=[pc2_err_lo, pc2_err_hi],
                fmt="none", ecolor="black", capsize=4, linewidth=1.3)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Slope vs position")
    ax.set_title(title or "PCA slope summary")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    return fig, ax



def _linfit_r2(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = x.size
    if n < 2:
        return np.nan, np.nan, np.nan, 0, np.full(0, np.nan)
    A = np.vstack([x, np.ones(n)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = slope*x + intercept
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    resid = y - yhat
    return float(slope), float(intercept), float(r2), int(n), resid

def _holm_adjust(pvals):
    pvals = np.asarray(pvals, float)
    m = pvals.size
    if m == 0:
        return pvals
    order = np.argsort(pvals)
    adj = np.empty(m, float)
    for i, idx in enumerate(order):
        adj[idx] = min(1.0, pvals[idx] * (m - i))
    # enforce monotonicity
    for i in range(m-2, -1, -1):
        idx_i = order[i]
        idx_j = order[i+1]
        adj[idx_i] = min(adj[idx_i], adj[idx_j])
    return adj

def _bootstrap_ci(x, n_boot=2000, ci=95, seed=0, func=np.mean):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.array([func(rng.choice(x, size=x.size, replace=True)) for _ in range(n_boot)], float)
    lo = np.percentile(boots, (100-ci)/2)
    hi = np.percentile(boots, 100 - (100-ci)/2)
    return float(func(x)), float(lo), float(hi)







def PCAByBin(binned_dict, order, nbins, pca_vars,
             colors=None, standardise=True, pooled=True,
             min_rows=10, alpha=0.35, s=12, dbg=False):
    """
    PCA per bin using binned_dict[subtype]["points"].

    Parameters
    ----------
    binned_dict : dict
        { subtype : {"points":[{... point dict ...}]} }
        points must contain: "bin" (0..nbins-1), "sample", "line_id"
    order : list[str]
        subtypes in plotting order
    nbins : int
        number of depth bins
    pca_vars : tuple/list[str]
        variables to include in PCA (e.g. ("tau_Visco","Einf_Visco","RelaxFrac"))
        Uses VAR_MAP if user passes display/header names.
    colors : dict[str,str]
        subtype -> color
    standardise : bool
        z-score each variable before PCA
    pooled : bool
        True  -> use all points in bin
        False -> average within (sample,line_id,bin) before PCA
    min_rows : int
        minimum rows required to run PCA in a bin
    Returns
    -------
    results_by_bin : dict
        per-bin PCA outputs (evr, loadings, n per group, etc.)
    figs_by_bin : dict
        per-bin matplotlib figures (scores + loadings overlay)
    """

    colors = colors or {}

    def _pts(st):
        return binned_dict.get(st, {}).get("points", [])

    def _resolve(v, pts0):
        # accept canonical key or mapped header name
        if pts0 and v in pts0:
            return v
        return VAR_MAP.get(v, v)

    def _rows_for_bin(st, b):
        pts = _pts(st)
        if not pts:
            return np.empty((0, len(pca_vars))), []

        # filter to bin
        pts_b = [p for p in pts if int(p.get("bin", -999)) == b]
        if not pts_b:
            return np.empty((0, len(pca_vars))), []

        # resolve variable keys using first point in this bin
        keys = [_resolve(v, pts_b[0]) for v in pca_vars]

        if pooled:
            mat = np.column_stack([np.array([p.get(k, np.nan) for p in pts_b], float) for k in keys])
            # keep rows with all finite
            m = np.all(np.isfinite(mat), axis=1)
            return mat[m], keys

        # pooled=False: average within (sample,line_id,bin)
        # (gives each line equal weight inside the bin)
        by = {}
        for p in pts_b:
            sid = p.get("sample", None)
            lid = p.get("line_id", None)
            if sid is None or lid is None:
                continue
            by.setdefault((sid, lid), []).append(p)

        rows = []
        for (sid, lid), plist in by.items():
            row = []
            ok = True
            for k in keys:
                vals = np.array([pp.get(k, np.nan) for pp in plist], float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    ok = False
                    break
                row.append(float(np.mean(vals)))
            if ok:
                rows.append(row)

        mat = np.array(rows, float) if rows else np.empty((0, len(pca_vars)))
        return mat, keys

    def _pca(mat):
        # mat shape (n, p)
        if standardise:
            mu = np.mean(mat, axis=0)
            sd = np.std(mat, axis=0, ddof=1)
            sd = np.where(sd == 0, 1.0, sd)
            Xz = (mat - mu) / sd
        else:
            mu = np.zeros(mat.shape[1], float)
            sd = np.ones(mat.shape[1], float)
            Xz = mat.copy()

        Xc = Xz - np.mean(Xz, axis=0)
        U, S, VT = np.linalg.svd(Xc, full_matrices=False)
        scores = U * S
        loadings = VT.T

        var = (S**2) / (Xc.shape[0] - 1)
        evr = var / np.sum(var)
        return scores, loadings, evr, mu, sd

    results_by_bin = {}
    figs_by_bin = {}

    print("\n=== PCA by bin ===")
    print(f"Vars: {tuple(pca_vars)} | pooled={pooled} | standardise={standardise}")

    for b in range(nbins):
        # build combined matrix across groups
        mats = []
        groups = []
        keys_used = None
        n_by_group = {}

        for st in order:
            mat_st, keys = _rows_for_bin(st, b)
            n_by_group[st] = int(mat_st.shape[0])
            if mat_st.shape[0] == 0:
                continue
            mats.append(mat_st)
            groups.append(np.full(mat_st.shape[0], st, dtype=object))
            if keys_used is None:
                keys_used = keys

        if not mats:
            if dbg:
                print(f"[PCAByBin] bin {b}: no data")
            results_by_bin[b] = {"ok": False, "reason": "no data", "n_by_group": n_by_group}
            continue

        X = np.vstack(mats)
        G = np.concatenate(groups)

        if X.shape[0] < max(min_rows, X.shape[1] + 1):
            if dbg:
                print(f"[PCAByBin] bin {b}: too few rows (n={X.shape[0]})")
            results_by_bin[b] = {"ok": False, "reason": "too few rows", "n": int(X.shape[0]),
                                 "n_by_group": n_by_group, "resolved_keys": keys_used}
            continue

        scores, loadings, evr, mu, sd = _pca(X)
        pc1, pc2 = scores[:, 0], scores[:, 1]

        # console summary
        print(f"\n--- Bin {b+1}/{nbins} ---  n={X.shape[0]}  (per group: {n_by_group})")
        print(f"Explained variance: PC1={evr[0]*100:.2f}% | PC2={evr[1]*100:.2f}%")
        for i, v in enumerate(pca_vars):
            print(f"Loading {v}:  PC1={loadings[i,0]: .4f}  PC2={loadings[i,1]: .4f}")

        # plot: scores + loadings overlay
        fig, ax = plt.subplots()
        for st in order:
            m = (G == st)
            if not np.any(m):
                continue
            ax.scatter(pc1[m], pc2[m], s=s, alpha=alpha, color=colors.get(st, None), label=st)

        # # loading arrows scaled to score space
        # span1 = np.nanpercentile(np.abs(pc1), 95)
        # span2 = np.nanpercentile(np.abs(pc2), 95)
        # L = loadings[:, :2]
        # Lscale = 0.35 * np.array([span1 if span1 > 0 else 1.0, span2 if span2 > 0 else 1.0])

        # for i, v in enumerate(pca_vars):
        #     ax.arrow(0, 0, L[i, 0]*Lscale[0], L[i, 1]*Lscale[1],
        #              head_width=0.04*min(Lscale[0], Lscale[1]),
        #              length_includes_head=True, linewidth=2)
        #     ax.text(L[i, 0]*Lscale[0]*1.08, L[i, 1]*Lscale[1]*1.08, str(v), fontsize=10)

        # ax.axhline(0, linewidth=1)
        # ax.axvline(0, linewidth=1)
        ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
        ax.set_title(f"PCA (bin {b+1}/{nbins})")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        figs_by_bin[b] = fig
        results_by_bin[b] = {
            "ok": True,
            "bin": b,
            "n": int(X.shape[0]),
            "n_by_group": n_by_group,
            "resolved_keys": keys_used,
            "standardise": bool(standardise),
            "pooled": bool(pooled),
            "mu": mu, "sd": sd,
            "scores": scores,
            "groups": G,
            "loadings": loadings,
            "explained_variance_ratio": evr,
        }

    return results_by_bin, figs_by_bin

def PrintPCABinCloudSummary(results_by_bin, order, conf=0.95):
    """
    Prints centroid and uncertainty per group for each bin PCA.

    Uses:
        - mean PC1, mean PC2
        - SD in PC1 and PC2
        - 95% ellipse radii (principal directions)
    """

    def chi2_2d(conf):
        # fixed common values
        if conf >= 0.99: return 9.210
        if conf >= 0.95: return 5.991
        if conf >= 0.90: return 4.605
        return 5.991

    chi2_val = chi2_2d(conf)

    print("\n=== PCA Cloud Summary by Bin ===")

    for b, res in results_by_bin.items():

        if not res.get("ok", False):
            continue

        scores = res["scores"]
        groups = res["groups"]

        pc1 = scores[:, 0]
        pc2 = scores[:, 1]

        print(f"\n--- Bin {b+1} ---")

        for st in order:
            m = (groups == st)
            if not np.any(m):
                continue

            xs = pc1[m]
            ys = pc2[m]

            mean_x = np.mean(xs)
            mean_y = np.mean(ys)

            sd_x = np.std(xs, ddof=1)
            sd_y = np.std(ys, ddof=1)

            # covariance ellipse principal axes
            cov = np.cov(np.vstack([xs, ys]))
            if cov.shape == (2,2) and np.linalg.det(cov) > 0:
                vals, vecs = np.linalg.eigh(cov)
                vals = np.sort(vals)[::-1]  # descending
                # 95% ellipse radii
                r_major = np.sqrt(vals[0] * chi2_val)
                r_minor = np.sqrt(vals[1] * chi2_val)
            else:
                r_major = np.nan
                r_minor = np.nan

            print(f"{st:>8} | n={len(xs):3d} "
                  f"| centre = ({mean_x: .3f}, {mean_y: .3f}) "
                  f"| SD = (±{sd_x:.3f}, ±{sd_y:.3f}) "
                  f"| {int(conf*100)}% ellipse radii = ({r_major:.3f}, {r_minor:.3f})")

def MixedModelSuite(
    binned_dict,
    order,
    vars_to_model=("tau_Visco", "Einf_Visco", "RelaxFrac"),
    colors=None,
    depth_key="norm_pos",          # uses stored per-point normalised position (0–100) if present
    depth_scale="0-1",             # "0-1" or "0-100" (model is nicer on 0–1)
    re_group="sample",             # random intercept grouping
    add_line_vc=True,              # add variance component for line_id (recommended)
    min_n_per_group=30,
    dropna=True,
    make_plots=True,
    dbg=False
):
    """
    Mixed model per variable:
        y ~ depth * group
    with random intercept by `re_group` (default: sample) and optional variance component for line_id.

    Inputs:
      - binned_dict[subtype]["points"] must contain:
          subtype, sample, line_id, and your vars; plus either norm_pos or enough to compute earlier.
      - order: e.g. ["control","d7","d14"]

    Outputs:
      - results: dict[var] = {"model": fit, "fixed": df_fixed, "slopes": df_slopes, ...}
      - figs: dict keys per var: "fit_<var>", "resid_<var>", "slopes_<var>" (if make_plots True)

    Requires:
      pip install statsmodels pandas numpy matplotlib
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    try:
        import statsmodels.formula.api as smf
    except Exception as e:
        raise ImportError("statsmodels is required. Install with: pip install statsmodels") from e

    colors = colors or {}
    figs = {}
    out = {}

    # ----------------------------
    # 1) Build a long dataframe
    # ----------------------------
    rows = []
    for st in order:
        pts = binned_dict.get(st, {}).get("points", [])
        if not pts:
            continue
        for p in pts:
            r = {"group": st}
            # common IDs
            r["sample"]  = p.get("sample", np.nan)
            r["line_id"] = p.get("line_id", np.nan)

            # depth
            if depth_key in p:
                r["depth"] = p.get(depth_key, np.nan)
            else:
                # if you ever don't store norm_pos, you could compute it here — but you said you now have it
                r["depth"] = np.nan

            # variables
            for v in vars_to_model:
                r[v] = p.get(v, np.nan)
            rows.append(r)

    df = pd.DataFrame(rows)
    if df.empty:
        print("[MixedModelSuite] No points found.")
        return out, figs

    # tidy
    df["group"] = pd.Categorical(df["group"], categories=order, ordered=True)

    # depth scaling
    if depth_scale == "0-1":
        df["depth_s"] = df["depth"] / 100.0
        depth_term = "depth_s"
        depth_label = "Normalised position (0–1)"
    else:
        df["depth_s"] = df["depth"]
        depth_term = "depth_s"
        depth_label = "Normalised position (0–100)"

    # enforce required columns
    req = ["group", "sample", "line_id", "depth_s"]
    for c in req:
        if c not in df.columns:
            raise KeyError(f"Dataframe missing required column: {c}")

    # optional dropna
    if dropna:
        df = df[np.isfinite(df["depth_s"])].copy()

    # group size info
    print("\n=== Mixed model input summary ===")
    print(df.groupby("group").size().to_string())
    if dbg:
        print(df.head())

    # variance component for line_id (nested/crossed-ish)
    vc = {"line": "0 + C(line_id)"} if add_line_vc else None

    # helper: simple fixed-effects R² proxy (marginal) + conditional proxy
    # (Nakagawa-style exact needs more work; this gives you something consistent for comparison)
    def _r2_proxies(y, yhat, resid, group_ids=None):
        y = np.asarray(y, float)
        yhat = np.asarray(yhat, float)
        resid = np.asarray(resid, float)
        m = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(resid)
        if m.sum() < 3:
            return np.nan, np.nan
        y = y[m]; yhat = yhat[m]; resid = resid[m]
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum(resid**2)
        r2_marg = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
        # "conditional-ish": include random effects already inside yhat (statsmodels predict includes RE by default)
        # here we treat it the same; for separation we'd need fixed-only predictions (see below).
        return r2_marg, r2_marg

    # fixed-only predictions
    def _predict_fixed_only(fit, dfx):
        # statsmodels MixedLMResults has fe_params; build design matrix via model.exog
        exog = fit.model.exog
        # but exog matches fit.model.data.frame ordering; easiest: refit predict with exog_re zeros:
        # statsmodels doesn't give a clean fixed-only predict API across versions.
        # We'll approximate fixed-only by:
        #   y_fixed = X * beta
        X = fit.model.exog
        beta = np.asarray(fit.fe_params)
        return X @ beta

    # helper: slopes per group from fixed effects
    def _group_slopes(fit, groups):
        """
        For model: y ~ depth * group  (treatment coding, baseline=first category)
        slope(group=baseline) = b_depth
        slope(group=g) = b_depth + b_depth:group[T.g]
        """
        fe = fit.fe_params
        cov = fit.cov_params()  # approx
        base = groups[0]

        rows = []
        # baseline
        b = float(fe.get(depth_term, np.nan))
        se = float(np.sqrt(cov.loc[depth_term, depth_term])) if (depth_term in cov.index) else np.nan
        rows.append({"group": base, "slope": b, "se": se})

        for g in groups[1:]:
            term = f"{depth_term}:group[T.{g}]"
            b_int = float(fe.get(term, 0.0))
            slope = b + b_int

            # SE(slope) = Var(b) + Var(b_int) + 2Cov(b,b_int)
            if (depth_term in cov.index) and (term in cov.index):
                var = cov.loc[depth_term, depth_term] + cov.loc[term, term] + 2*cov.loc[depth_term, term]
                se = float(np.sqrt(max(var, 0.0)))
            else:
                se = np.nan

            rows.append({"group": g, "slope": float(slope), "se": se})

        df_s = pd.DataFrame(rows)
        df_s["ci95_lo"] = df_s["slope"] - 1.96*df_s["se"]
        df_s["ci95_hi"] = df_s["slope"] + 1.96*df_s["se"]
        return df_s

    # ----------------------------
    # 2) Fit models per variable
    # ----------------------------
    for v in vars_to_model:
        d = df[["group", "sample", "line_id", "depth_s", v]].copy()
        d = d.rename(columns={v: "y"})

        if dropna:
            d = d[np.isfinite(d["y"])].copy()

        # ensure enough data per group
        counts = d.groupby("group").size()
        ok_groups = [g for g in order if counts.get(g, 0) >= min_n_per_group]
        if len(ok_groups) < 2:
            print(f"\n[{v}] skipped: not enough data per group (min {min_n_per_group}).")
            continue
        d = d[d["group"].isin(ok_groups)].copy()
        d["group"] = pd.Categorical(d["group"], categories=ok_groups, ordered=True)

        print(f"\n=== Mixed model: {v} ===")
        print("Formula: y ~ depth * group  (random intercept: sample" + (" + line_id VC" if add_line_vc else "") + ")")

        # fit
        md = smf.mixedlm(
            f"y ~ {depth_term} * group",
            d,
            groups=d[re_group],
            vc_formula=vc,
            re_formula="1"
        )
        
        fit = md.fit(method="lbfgs", maxiter=200, disp=False)
        
        # Residual diagnostics
        try:
            CheckMixedModelResiduals(fit, title=f"{v} residual diagnostics")
        except Exception as e:
            print(f"[{v}] residual diagnostics failed: {e}")
        except Exception as e:
            print(f"[{v}] fit failed: {e}")
            continue

        # console: compact fixed effects
        fe = fit.fe_params
        se = fit.bse_fe
        p = fit.pvalues.loc[fe.index] if hasattr(fit, "pvalues") else None
        fixed = pd.DataFrame({
            "coef": fe,
            "se": se,
            "z": fe / se,
            "p": p
        })
        print("\nFixed effects:")
        print(fixed.to_string(float_format=lambda x: f"{x: .4g}"))

        # slopes per group
        slopes = _group_slopes(fit, ok_groups)
        print("\nSlopes vs depth (per group):")
        # interpret slope in chosen units
        # if depth_s is 0–1, slope is "per full thickness"; if 0–100, slope per percent point.
        print(slopes.to_string(index=False, float_format=lambda x: f"{x: .4g}"))

        # predictions + residuals
        # statsmodels predict() includes RE by default; we'll compute fixed-only ourselves for comparisons.
        yhat_fixed = _predict_fixed_only(fit, d)
        resid = d["y"].values - yhat_fixed

        r2_marg, r2_cond = _r2_proxies(d["y"].values, yhat_fixed, resid)
        print(f"\nR² proxy (fixed-only): {r2_marg: .4g}")

        # store
        out[v] = {
            "model": fit,
            "fixed": fixed,
            "slopes": slopes,
            "n": int(len(d)),
            "groups_used": ok_groups,
            "r2_fixed_proxy": r2_marg
        }

        if not make_plots:
            continue

        # ----------------------------
        # 3) Plot: scatter + fitted lines (fixed-only)
        # ----------------------------
        fig_fit, ax = plt.subplots()
        for g in ok_groups:
            m = (d["group"] == g).to_numpy()
            if not np.any(m):
                continue
            col = colors.get(g, None)

            ax.scatter(
                d.loc[m, "depth_s"].values,
                d.loc[m, "y"].values,
                s=10, alpha=0.25, color=col, label=g
            )

            # fixed line for group: use slope/intercept from fixed effects
            # build line using model matrix would be overkill; use slope table:
            sl = float(slopes.loc[slopes["group"] == g, "slope"].values[0])
            # intercept differs by group: intercept + group[T.g] (treatment coding)
            b0 = float(fe.get("Intercept", 0.0))
            if g != ok_groups[0]:
                b0 += float(fe.get(f"group[T.{g}]", 0.0))
            # plus interaction in slope already included in sl

            xx = np.linspace(d["depth_s"].min(), d["depth_s"].max(), 200)
            yy = b0 + sl * xx
            ax.plot(xx, yy, linewidth=2, color=col)

        ax.set_xlabel(depth_label)
        ax.set_ylabel(v)
        ax.set_title(f"{v}: mixed model fit (fixed effects)")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        figs[f"fit_{v}"] = fig_fit

        # ----------------------------
        # 4) Plot: residuals vs depth (fixed-only)
        # ----------------------------
        fig_res, axr = plt.subplots()
        for g in ok_groups:
            m = (d["group"] == g).to_numpy()
            if not np.any(m):
                continue
            col = colors.get(g, None)
            axr.scatter(
                d.loc[m, "depth_s"].values,
                resid[m],
                s=10, alpha=0.35, color=col, label=g
            )
        axr.axhline(0, linewidth=1)
        axr.set_xlabel(depth_label)
        axr.set_ylabel("Residual (y - fixed fit)")
        axr.set_title(f"{v}: residuals vs depth")
        axr.grid(True)
        axr.legend()
        plt.tight_layout()
        figs[f"resid_{v}"] = fig_res

        # ----------------------------
        # 5) Plot: slope comparison with 95% CI
        # ----------------------------
        fig_sl, axsl = plt.subplots()
        xidx = np.arange(len(ok_groups))
        axsl.errorbar(
            xidx,
            slopes["slope"].values,
            yerr=1.96 * slopes["se"].values,
            fmt="o",
            capsize=4
        )
        axsl.set_xticks(xidx)
        axsl.set_xticklabels(ok_groups)
        axsl.set_xlabel("Group")
        axsl.set_ylabel(f"Slope d({v})/d(depth)")
        axsl.set_title(f"{v}: depth slope by group (±95% CI)")
        axsl.grid(True, axis="y")
        plt.tight_layout()
        figs[f"slopes_{v}"] = fig_sl

    return out, figs

def CheckMixedModelResiduals(model_result, title="Mixed Model Residual Diagnostics"):
    """
    Diagnostic plots and statistics for residual normality in statsmodels MixedLM.
    
    Parameters
    ----------
    model_result : fitted statsmodels MixedLM result
    title : str
    """

    resid = model_result.resid
    fitted = model_result.fittedvalues

    resid = np.asarray(resid)
    fitted = np.asarray(fitted)

    resid = resid[np.isfinite(resid)]
    fitted = fitted[np.isfinite(fitted)]

    print("\n=== Residual diagnostics ===")
    print(f"N residuals: {len(resid)}")

    # Basic stats
    skew = stats.skew(resid)
    kurt = stats.kurtosis(resid, fisher=True)

    print(f"Skewness: {skew:.4f}")
    print(f"Kurtosis (Fisher): {kurt:.4f}")

    # Shapiro test (limited to n<=5000 for safety)
    if len(resid) > 5000:
        sample = np.random.choice(resid, 5000, replace=False)
    else:
        sample = resid

    shapiro_stat, shapiro_p = stats.shapiro(sample)

    print(f"Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4g}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram
    axes[0].hist(resid, bins=30, density=True, alpha=0.7)
    x = np.linspace(np.min(resid), np.max(resid), 200)
    axes[0].plot(x, stats.norm.pdf(x, np.mean(resid), np.std(resid)), linewidth=2)
    axes[0].set_title("Residual Histogram")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Density")

    # QQ plot
    stats.probplot(resid, dist="norm", plot=axes[1])
    axes[1].set_title("QQ Plot")

    # Residual vs fitted
    axes[2].scatter(fitted, resid, alpha=0.4)
    axes[2].axhline(0, linestyle="--")
    axes[2].set_xlabel("Fitted values")
    axes[2].set_ylabel("Residual")
    axes[2].set_title("Residuals vs Fitted")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    return {
        "skew": skew,
        "kurtosis": kurt,
        "shapiro_p": shapiro_p
    }


