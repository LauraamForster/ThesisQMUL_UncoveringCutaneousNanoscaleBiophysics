import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 15,
})
# ============================================================
# USER SETTINGS
# ============================================================

manifest_path = "/Users/lauraforster/Documents/Uni/3 - PhD/SAXS/Presentations and notes/For Experiments/current/Bovine Manifest.xlsx"

csv_base_dir = Path("/Users/lauraforster/Documents/Uni/3 - PhD/SAXS/DLS visits")

DataMode = "IQ"      # "IQ", "IChi", or "WAXS"
WhatPlot = "Dperiod"  #|SAXS: Dperiod, totalSAXS, SAXSnorm, curvearea, curvenorm, skewness, 
                        #fibrilradius, q0, peak_width, peak_amplitude 
                        #| WAXS: total_area, g1_center, g1_amp, g1_height, g1_present, r2 , I_period
                        #| IChi: peak_position, peak_width, peak_amplitude, peak_height, 
                        #SM, AP, area_fit, area_peaks
PeakNo   = 1         # only used for IChi peak_* metrics

# ============================================================
# PARAMETER RULES (AS YOU PROVIDED)
# ============================================================

def get_plot_rules(WhatPlot):
    """
    Returns (min_value, max_value, zmin, zmax) for the chosen WhatPlot.
    Edit/tune thresholds here as you learn your data ranges.
    """
    rules = {
        # ---------------- IQ (SAXS) ----------------
        "Dperiod":       (60, None, 64, 67),
        "totalSAXS":     (0,  None, 0,  None),
        "SAXSnorm":      (0,  1,    0,  1),
        "curvearea":     (0,  None, 0,  None),
        "curvenorm":     (0,  0.5,  0,  0.5),
        "skewness":      (0,  None, 0,  None),
        "fibrilradius":  (5,  600,  0,  600),
        "q0":            (0,  0.35, 0.25, 0.35),
        "peak_width":    (0.0005, 0.0075, 0.001, 0.004),
        "peak_position": (0,  360,  0,  360),
        "peak_amplitude":(0,  15,   0,  15),
        "peak_height":   (0,  0.5,  0,  0.5),
        "SM":            (0,  2,    0,  2),
        "AP":            (0,  1,    0,  0.35),
        "area_fit":      (0,  100,  0,  100),
        "area_peaks":    (0,  0.5,  0,  0.5),

        # ---------------- WAXS ----------------
        "total_area":    (None, None, None, None),
        "g1_center":     (None, None, None, None),
        "g1_amp":        (None, None, None, None),
        "g1_height":     (None, None, None, None),
        "g1_present":    (0, 1, 0, 1),
        "r2":            (0, 1, 0, 1),
        "I_period":      (0, 3, 0, 3),
    }

    return rules.get(WhatPlot, (None, None, None, None))

SUBTYPE_ORDER = [
    "dehydrated_no_ultralene",
    "hydrated_no_ultralene",
    "dehydrated_ultralene",
    "hydrated_ultralene",
]

min_value, max_value, zmin, zmax = get_plot_rules(WhatPlot)
# ============================================================
# LOAD MANIFEST
# ============================================================

manifest = pd.read_excel(manifest_path)

experiment_col = "Experiment"
analysis_col   = "Analysis Number"

# Build a simple subtype from TYPE + FOLDER NAME
# TYPE: "hy" / "de"
# FOLDER NAME: contains "ul" / "ultralene" -> ultralene, otherwise no_ultralene
def make_subtype(row):
    t = str(row["TYPE"]).strip().lower()
    folder = str(row["FOLDER NAME"]).strip().lower()

    # hydration from TYPE (works for "hy", "hy noul", "de", "de noul", etc.)
    hydration = "hydrated" if t.startswith("hy") else "dehydrated"

    # ultralene flag: "noul" / "no ul" means no ultralene; otherwise assume ultralene
    no_ul = ("noul" in t) or ("noul" in folder) or ("no ul" in t) or ("no_ul" in t)
    ultralene = "no_ultralene" if no_ul else "ultralene"

    return f"{hydration}_{ultralene}"
manifest["Subtype"] = manifest.apply(make_subtype, axis=1)
subtype_col = "Subtype"


# ============================================================
# LOOP OVER SAMPLES
# ============================================================

rows = []

for _, row in manifest.iterrows():
    experiment = str(row[experiment_col]).strip()
    analysis   = str(row[analysis_col]).strip()
    subtype    = str(row[subtype_col]).strip()

    csv_dir = csv_base_dir / experiment / "CSVs"

    # Choose folder + filename pattern
    if DataMode == "WAXS":
        csv_dir = csv_base_dir / experiment / "WAXS_CSVs"
        csv_file = csv_dir / f"{analysis} IQ_fitting.csv"  # change if your actual suffix differs
    else:
        csv_dir = csv_base_dir / experiment / "CSVs"
        if DataMode == "IQ":
            csv_file = csv_dir / f"{analysis} IQ_fitting.csv"
        else:
            csv_file = csv_dir / f"{analysis} IChi_fitting.csv"
        
    if not csv_file.exists():
        print(f"[skip] Missing {DataMode} CSV for {analysis}")
        continue

    df = pd.read_csv(csv_file)
    # DEBUG: print columns once so we can set WhatPlot correctly
    if "printed_cols" not in globals():
        printed_cols = True
        print("\nExample CSV:", csv_file.name)
        print("Columns:", list(df.columns), "\n")
    

    # --------------------------------------------------------
    # Resolve column name
    # -------------------------
    if DataMode == "IQ":
        iq_map = {
            "totalSAXS": "total SAXS intensity",
            "SAXSnorm": "total_SAXS_norm_0_1",
            "curvearea": "area under third order curve",
            "curvenorm": "collagen_third_norm_0_1",
            "Dperiod": "D_period",
            "peak_width": "peak_width",
            "peak_amplitude": "peak_amplitude",
            "skewness": "skewness",
            "q0": "q0",
            "fibrilradius": "fibril_radius",
        }
        colname = iq_map.get(WhatPlot, WhatPlot)
    
    elif DataMode == "IChi":
        if WhatPlot.startswith("peak_") and PeakNo > 1:
            colname = f"{WhatPlot}{PeakNo}"
        else:
            colname = WhatPlot
    
    else:  # DataMode == "WAXS"
        waxs_map = {
            "total_area": "total_area",
            "g1_center": "g1_center",
            "g1_amp": "g1_amp",
            "g1_height": "g1_height",
            "g1_present": "g1_present",
            "r2": "r2",
            "I_period": "I_period",
        }
        colname = waxs_map.get(WhatPlot, WhatPlot)


    if colname not in df.columns:
        print(f"[skip] {colname} not in {csv_file.name}")
        continue

    values = pd.to_numeric(df[colname], errors="coerce")

    if min_value is not None:
        values = values[values >= min_value]
    if max_value is not None:
        values = values[values <= max_value]

    values = values.dropna()

    if values.empty:
        print(f"[skip] No valid values for {analysis}")
        continue

    rows.append({
        "analysis": analysis,
        "experiment": experiment,
        "subtype": subtype,
        "csv_file": str(csv_file),   # <-- add this
        "mean": values.mean(),
        "std": values.std(),
        "n": len(values)
    })
    
    df_samples = pd.DataFrame(rows)

# ============================================================
# BUILD DATAFRAME
# ============================================================

df_samples = pd.DataFrame(rows)

df_samples["subtype"] = pd.Categorical(
    df_samples["subtype"],
    categories=SUBTYPE_ORDER,
    ordered=True
)

print("\nPer-sample means:")
print(df_samples)

# ============================================================
# AVERAGE PER SUBTYPE
# ============================================================

df_subtype = (
    df_samples
    .groupby("subtype")["mean"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

print("\nPer-subtype summary:")
print(df_subtype)

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
colors = []
for s in df_subtype["subtype"]:
    if "dehydrated" in s:
        colors.append("tab:green")
    elif "hydrated" in s:
        colors.append("tab:blue")
    else:
        colors.append("grey")
        
ax.bar(
    df_subtype["subtype"],
    df_subtype["mean"],
    yerr=df_subtype["std"],
    capsize=5,
    alpha=0.8,
    color=colors

)

ax.set_ylabel(WhatPlot)
# ax.set_title(f"{WhatPlot} across bovine tendon subtypes")
ax.set_xticklabels(df_subtype["subtype"], rotation=30, ha="right")

if zmin is not None or zmax is not None:
    ax.set_ylim(zmin, zmax)
ax.axhline(65.5, color="k", linestyle="--", linewidth=1)

print(f"\n{WhatPlot} summary (mean ± std):")
print(df_subtype[["subtype", "mean", "std"]].to_string(index=False))
# ax.set_ylim(0.8,1.5)
plt.show()

# ============================================================
# PER-SAMPLE BAR PLOT
# ============================================================

# sort nicely: subtype first, then analysis number
df_plot = df_samples.sort_values(["subtype", "analysis"])

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
# colour by hydration state
# colour by hydration state (careful: dehydrated contains "hydrated")
colors = []
for s in df_plot["subtype"]:
    if "dehydrated" in s:
        colors.append("tab:orange")
    elif "hydrated" in s:
        colors.append("tab:blue")
    else:
        colors.append("grey")
ax.bar(
    df_plot["analysis"],
    df_plot["mean"],
    yerr=df_plot["std"],
    capsize=4,
    alpha=0.8,
    color=colors
)

ax.set_ylabel(WhatPlot)
ax.set_title(f"{WhatPlot} per sample (bovine tendon)")

# ax.axhline(67, color="k", linestyle="--", linewidth=1)

ax.set_xticklabels(df_plot["analysis"], rotation=45, ha="right")

if zmin is not None or zmax is not None:
    ax.set_ylim(zmin, zmax)

print(f"\n{WhatPlot} per-sample values (mean ± std):")
print(df_plot[["analysis", "subtype", "mean", "std"]].to_string(index=False))

plt.show()


# ============================================================
# MULTIPANEL HEATMAPS (per sample)
# ============================================================

def _xy_to_grid(df, xcol="x", ycol="y", vcol="value"):
    xs = np.sort(df[xcol].unique())
    ys = np.sort(df[ycol].unique())
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    Z = np.full((len(ys), len(xs)), np.nan, dtype=float)
    Z[df[ycol].map(yi).to_numpy(), df[xcol].map(xi).to_numpy()] = df[vcol].to_numpy(dtype=float)
    return Z

# keep same ordering as your per-sample plot
df_hm = df_samples.sort_values(["subtype", "analysis"]).reset_index(drop=True)

n = len(df_hm)
if n == 0:
    print("[skip] No samples for heatmaps.")
else:
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6*ncols, 3.2*nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    im = None
    for i, r in enumerate(df_hm.itertuples(index=False)):
        ax = axes[i]
        df = pd.read_csv(r.csv_file)

        if not {"x", "y", colname}.issubset(df.columns):
            ax.set_title(f"{r.analysis}\n[missing x/y/{colname}]")
            ax.axis("off")
            continue

        d = df[["x", "y", colname]].copy()
        d[colname] = pd.to_numeric(d[colname], errors="coerce")

        # apply same filtering rules as your stats
        if min_value is not None:
            d.loc[d[colname] < min_value, colname] = np.nan
        if max_value is not None:
            d.loc[d[colname] > max_value, colname] = np.nan

        d = d.dropna(subset=[colname])
        if d.empty:
            ax.set_title(f"{r.analysis}\n[no valid values]")
            ax.axis("off")
            continue

        d = d.rename(columns={colname: "value"})
        Z = _xy_to_grid(d, vcol="value")

        im = ax.imshow(Z, origin="lower", cmap="jet", vmin=zmin, vmax=zmax, aspect="equal")
        ax.set_title(f"{r.analysis}\n{r.subtype}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # hide unused panels
    for j in range(n, len(axes)):
        axes[j].axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes[:n], shrink=0.9, pad=0.02)
        cbar.set_label(WhatPlot)

    plt.show()
    
    
    
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_overlay_heatmaps_per_sample(
    df_samples,
    var1="Dperiod",
    var2="totalSAXS",
    get_plot_rules=None,     # your existing get_plot_rules(WhatPlot) -> (min,max,zmin,zmax)
    iq_map=None,             # dict mapping your keys to CSV column headers
    alpha2=0.35,             # transparency for var2 greyscale overlay
    ncols=4
):
    """
    Multipanel overlay heatmaps per sample.
    var1 shown with JET; var2 shown with gray + transparency over var1.
    Uses x,y columns to rebuild a 2D grid.
    """

    if iq_map is None:
        iq_map = {
            "totalSAXS": "total SAXS intensity",
            "SAXSnorm": "total_SAXS_norm_0_1",
            "curvearea": "area under third order curve",
            "curvenorm": "collagen_third_norm_0_1",
            "Dperiod": "D_period",
            "peak_width": "peak_width",
            "peak_amplitude": "peak_amplitude",
            "skewness": "skewness",
            "q0": "q0",
            "fibrilradius": "fibril_radius",
        }

    def resolve_col(v):
        return iq_map.get(v, v)

    def xy_to_grid(d, xcol="x", ycol="y", vcol="value"):
        xs = np.sort(d[xcol].unique())
        ys = np.sort(d[ycol].unique())
        xi = {v: i for i, v in enumerate(xs)}
        yi = {v: i for i, v in enumerate(ys)}
        Z = np.full((len(ys), len(xs)), np.nan, dtype=float)
        Z[d[ycol].map(yi).to_numpy(), d[xcol].map(xi).to_numpy()] = d[vcol].to_numpy(dtype=float)
        return Z

    c1 = resolve_col(var1)
    c2 = resolve_col(var2)

    # rules
    if get_plot_rules is None:
        # fallback: no filtering/scaling
        min1 = max1 = zmin1 = zmax1 = None
        min2 = max2 = zmin2 = zmax2 = None
    else:
        min1, max1, zmin1, zmax1 = get_plot_rules(var1)
        min2, max2, zmin2, zmax2 = get_plot_rules(var2)

    df_hm = df_samples.sort_values(["subtype", "analysis"]).reset_index(drop=True)
    n = len(df_hm)
    if n == 0:
        print("[skip] No samples for overlay heatmaps.")
        return

    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8*ncols, 3.3*nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    im1 = None
    im2 = None

    for i, r in enumerate(df_hm.itertuples(index=False)):
        ax = axes[i]
        df = pd.read_csv(r.csv_file)

        if not {"x", "y", c1, c2}.issubset(df.columns):
            ax.set_title(f"{r.analysis}\n[missing x/y/{c1}/{c2}]")
            ax.axis("off")
            continue

        d = df[["x", "y", c1, c2]].copy()
        d[c1] = pd.to_numeric(d[c1], errors="coerce")
        d[c2] = pd.to_numeric(d[c2], errors="coerce")

        # apply point-level filters by turning out-of-range into NaN
        if min1 is not None: d.loc[d[c1] < min1, c1] = np.nan
        if max1 is not None: d.loc[d[c1] > max1, c1] = np.nan
        if min2 is not None: d.loc[d[c2] < min2, c2] = np.nan
        if max2 is not None: d.loc[d[c2] > max2, c2] = np.nan

        # build grids
        d1 = d.dropna(subset=[c1]).rename(columns={c1: "value"})[["x", "y", "value"]]
        d2 = d.dropna(subset=[c2]).rename(columns={c2: "value"})[["x", "y", "value"]]

        if d1.empty and d2.empty:
            ax.set_title(f"{r.analysis}\n[no valid values]")
            ax.axis("off")
            continue

        # var1 base
        if not d1.empty:
            Z1 = xy_to_grid(d1)
            im1 = ax.imshow(Z1, origin="lower", cmap="terrain", vmin=zmin1, vmax=zmax1, aspect="equal")

        # var2 overlay
        if not d2.empty:
            Z2 = xy_to_grid(d2)
            im2 = ax.imshow(Z2, origin="lower", cmap="binary", vmin=zmin2, vmax=zmax2,
                            aspect="equal", alpha=alpha2)

        ax.set_title(f"{r.analysis}\n{r.subtype}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # hide unused panels
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # colourbars (one for each layer)
    if im1 is not None:
        cb1 = fig.colorbar(im1, ax=axes[:n], shrink=0.9, pad=0.02)
        cb1.set_label(var1)

    if im2 is not None:
        cb2 = fig.colorbar(im2, ax=axes[:n], shrink=0.9, pad=0.08)
        cb2.set_label(var2)

    plt.show()
    
    
    
# plot_overlay_heatmaps_per_sample(
#     df_samples,
#     var1="totalSAXS",
#     var2="Dperiod",
#     get_plot_rules=get_plot_rules,
#     alpha2=0.5,
#     ncols=4
#     )