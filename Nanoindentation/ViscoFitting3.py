#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 15:16:30 2026

@author: lauraforster
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model
from scipy.special import erfi


NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
TABLE_HDR = "Time (s)\tLoad (uN)\tIndentation (nm)\tCantilever (nm)\tPiezo (nm)\tAuxiliary"


# ---------------------------- IO + parsing ----------------------------

def _tab_kv(line: str):
    p = [x.strip() for x in line.split("\t") if x.strip()]
    return {p[i]: p[i + 1] for i in range(0, len(p) - 1, 2)}

def _as_float(x):
    m = re.search(NUM, str(x))
    return float(m.group()) if m else None

def parse_header(lines):
    kv = {}
    for ln in lines:
        if "\t" in ln:
            kv.update(_tab_kv(ln))

    meta = {}
    def put(out_key, in_key, scale=1.0):
        v = _as_float(kv.get(in_key))
        if v is not None:
            meta[out_key] = v * scale

    put("tip_radius_m", "Tip radius (um)", 1e-6)
    put("k_N_per_m", "k (N/m)", 1.0)
    put("Eeff_file_Pa", "E[eff] (Pa)", 1.0)
    put("E_v05_file_Pa", "E[v=0.500] (Pa)", 1.0)

    for i in (1, 2, 3):
        put(f"DZ{i}_m", f"D[Z{i}] (nm)", 1e-9)
        put(f"t{i}_s",  f"t[{i}] (s)", 1.0)

    for ln in lines:
        if "Step absolute start times (s)" in ln:
            meta["step_start_s"] = [float(x) for x in re.findall(NUM, ln)]
        elif "Step absolute end times (s)" in ln:
            meta["step_end_s"] = [float(x) for x in re.findall(NUM, ln)]

    return meta

def read_chiaro_txt(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    try:
        i0 = next(i for i, ln in enumerate(lines) if ln.strip() == TABLE_HDR)
    except StopIteration:
        raise ValueError("Could not find the 6-column table header line.")

    meta = parse_header(lines[:i0])

    data_lines = [ln for ln in lines[i0 + 1:] if re.match(rf"^\s*{NUM}\s*\t", ln)]
    df = pd.read_csv(pd.io.common.StringIO("\n".join([TABLE_HDR] + data_lines)), sep="\t", engine="python")

    # SI conversion
    df = df.rename(columns={
        "Time (s)": "Time_s",
        "Load (uN)": "Load_uN",
        "Indentation (nm)": "Indentation_nm",
        "Cantilever (nm)": "Cantilever_nm",
        "Piezo (nm)": "Piezo_nm",
        "Auxiliary": "Auxiliary",
    })
    df["Time_s"] = df["Time_s"].astype(float)
    df["Load_N"] = df["Load_uN"].astype(float) * 1e-6
    df["Indentation_m"] = df["Indentation_nm"].astype(float) * 1e-9
    df["Cantilever_m"] = df["Cantilever_nm"].astype(float) * 1e-9
    df["Piezo_m"] = df["Piezo_nm"].astype(float) * 1e-9
    df = df[["Time_s", "Load_N", "Indentation_m", "Cantilever_m", "Piezo_m", "Auxiliary"]]

    # segmentation
    starts = meta.get("step_start_s", [])
    ends = meta.get("step_end_s", [])
    if not starts or len(starts) != len(ends):
        return meta, df, {"all": df}

    t = df["Time_s"].to_numpy(float)
    seg = {"step0_pre": df[(t >= 0.0) & (t < starts[0])]}
    for i, (s, e) in enumerate(zip(starts, ends), start=1):
        seg[f"step{i}"] = df[(t >= s) & (t < e)]
    post = df[t >= ends[-1]]
    if len(post):
        seg["post_last"] = post
    return meta, df, seg

# ---------------------------- phase helpers ----------------------------

def concat_segments(segments, keys):
    dfs = [segments[k] for k in keys if k in segments and len(segments[k])]
    if not dfs:
        raise ValueError(f"No data for segments: {keys}")
    return pd.concat(dfs, ignore_index=True)

def baseline_from_approach_tail(segments, tail_frac=0.3, approach_key="step0_pre"):
    ap = segments.get(approach_key)
    if ap is None or len(ap) < 20:
        raise ValueError("Approach segment missing/too short for baseline.")
    y = ap["Load_N"].to_numpy(float)
    y = y[np.isfinite(y)]
    tail = y[int((1 - tail_frac) * len(y)):]
    return float(np.median(tail))

def apply_baseline(df, offset_N):
    out = df.copy()
    out["Load_N"] = out["Load_N"].to_numpy(float) - float(offset_N)
    return out

# ---------------------------- extra helpers ----------------------------
def relaxation_fraction_from_segments(segments, baseline_offset_N=0.0, hold_key="step2"):
    """
    Computes:
        RelaxFrac = (Load_start - Load_end) / Load_start
    using the holding segment.
    """

    hold = segments.get(hold_key)
    if hold is None or len(hold) < 2:
        raise ValueError("Holding segment not found or too short.")

    load = hold["Load_N"].to_numpy(float) - float(baseline_offset_N)

    # remove NaNs
    load = load[np.isfinite(load)]
    if len(load) < 2:
        raise ValueError("Not enough valid load points in holding phase.")

    load_start = load[0]
    load_end = load[-1]

    relax_frac = (load_start - load_end) / load_start if load_start != 0 else np.nan

    return load_start, load_end, relax_frac

# ---------------------------- visco model ----------------------------

def visco_P_analytic(t, G0, tau, G1, R, v0, t_ramp):
    t = np.asarray(t, float)
    tau = max(float(tau), 1e-6)

    # a = min(t, t_ramp)
    a = np.minimum(t, float(t_ramp))
    a = np.maximum(a, 0.0)

    # avoid sqrt(0) issues
    a_safe = np.maximum(a, 1e-15)

    x = np.sqrt(a_safe / tau)

    bracket = (
        np.sqrt(a_safe / tau) * np.exp(a_safe / tau)
        - (np.sqrt(np.pi) / 2.0) * erfi(x)
    )

    pref = (16.0 * np.sqrt(R) / 3.0) * (3.0 / 2.0) * (v0 ** 1.5)

    term_visc = G0 * (tau ** 1.5) * np.exp(-t / tau) * bracket
    term_el = (2.0 / 3.0) * G1 * (a_safe ** 1.5)

    P = pref * (term_visc + term_el)

    P = np.asarray(P, float)
    P[~np.isfinite(P)] = 1e30
    return P

def fit_analytic_model(meta, df_fit_bl, depth_col="Indentation_m"):
    t = df_fit_bl["Time_s"].to_numpy(float)
    t = t - t[0]
    P = df_fit_bl["Load_N"].to_numpy(float)

    R = float(meta["tip_radius_m"])

    t_ramp = float(meta["t1_s"])                      # loading time
    v0 = float(meta["DZ1_m"]) / float(meta["t1_s"])   # loading speed

    model = Model(visco_P_analytic, independent_vars=["t"])
    params = model.make_params(G0=500.0, tau=0.6, G1=200.0)

    # smarter bounds
    params["tau"].set(value=0.7, min=0.1, max=5.0)
    params["G0"].set(value=300.0, min=0.0, max=5000.0)
    params["G1"].set(value=200.0, min=0.0, max=5000.0)

    params.add("R", value=R, vary=False)
    params.add("v0", value=v0, vary=False)
    params.add("t_ramp", value=t_ramp, vary=False)

    # seed G1 from end plateau
    A = 16*np.sqrt(R)/3
    P_end = float(np.median(P[int(0.9*len(P)):] ))
    G1_guess = P_end / (A * (v0**1.5) * (t_ramp**1.5))
    params["G1"].set(value=float(np.clip(G1_guess, 0.0, 1e7)))

    # trim first 0.15 s for analytic fit
    mask = t >= 0.15
    t2, P2 = t[mask], P[mask]

    # robust then refine
    out = model.fit(P2, params, t=t2, nan_policy="omit", method="powell")
    # out = model.fit(P2, out.params, t=t2, nan_policy="omit", method="leastsq")
    return out

# ---------------------------- plotting ----------------------------

def plot_fit_overlay(df_plot_bl, df_fit_bl, result, depth_col="Indentation_m"):
    t_plot = df_plot_bl["Time_s"].to_numpy(float)
    t_plot0 = t_plot - t_plot[0]
    P_plot = df_plot_bl["Load_N"].to_numpy(float)

    t_fit_abs = df_fit_bl["Time_s"].to_numpy(float)
    t_fit_rel = t_fit_abs - t_fit_abs[0]
    P_fit = result.eval(t=t_fit_rel, depth_m=df_fit_bl[depth_col].to_numpy(float))

    fit_shift = t_fit_abs[0] - t_plot[0]
    t_fit_on_plot = t_fit_rel + fit_shift

    fig, ax = plt.subplots()
    ax.plot(t_plot0, P_plot, label="Data (Approach+Loading+Holding)")
    ax.plot(t_fit_on_plot, P_fit, label="Fit (Loading+Holding; measured depth)", linewidth=2.0)
    ax.set_xlabel("Time since ViscoPhase start (s)")
    ax.set_ylabel("Load (N)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------- run ----------------------------

path = "/Volumes/LauraDrive/NI_Data/Data/Wounding/Wounding/Day7 1/matrix_scan01/wd7 horiz 233 S-1 X-01 Y-90 I-01.txt"

meta, df, segments = read_chiaro_txt(path)

# plot uses approach+loading+holding; fit uses loading+holding only
df_plot = concat_segments(segments, ["step0_pre", "step1", "step2"])
df_fit  = concat_segments(segments, ["step1", "step2"])

offset_N = baseline_from_approach_tail(segments, tail_frac=0.3)
df_plot_bl = apply_baseline(df_plot, offset_N)
df_fit_bl  = apply_baseline(df_fit, offset_N)

# ---- Fit analytical ramp model ----
result_analytic = fit_analytic_model(meta, df_fit_bl)
# print(result.fit_report())

t_plot = df_plot_bl["Time_s"].to_numpy(float)
t_plot0 = t_plot - t_plot[0]
P_plot = df_plot_bl["Load_N"].to_numpy(float)

# ----- depth model prediction -----
t_fit_abs = df_fit_bl["Time_s"].to_numpy(float)
t_fit_rel = t_fit_abs - t_fit_abs[0]


# ----- analytic model prediction -----
P_analytic = result_analytic.eval(t=t_fit_rel)

fit_shift = t_fit_abs[0] - t_plot[0]
t_fit_on_plot = t_fit_rel + fit_shift

fig, ax = plt.subplots()
ax.plot(t_plot0, P_plot, label="Data", linewidth=1.5)


ax.plot(t_fit_on_plot, P_analytic,
        label="Fit – Analytical Ramp (erfi)",
        linestyle="--",
        linewidth=2.0)

ax.set_xlabel("Time since ViscoPhase start (s)")
ax.set_ylabel("Load (N)")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()


load_start, load_end, relax_frac = relaxation_fraction_from_segments(segments, baseline_offset_N=offset_N, hold_key="step2")

# quick outputs
nu = 0.5
G0 = result_analytic.params["G0"].value
G1 = result_analytic.params["G1"].value
E0 = 2*(G0+G1)*(1+nu)
Einf = 2*G1*(1+nu)
print("Analytical Fit Results (nu=0.5)")
print(f"G0={G0:.1} Pa, tau={result_analytic.params['tau'].value:.3f} s, G1={G1:.1f} Pa")
print(f"E0={E0:.1f} Pa, E_inf={Einf:.1f} Pa")
print('\n')

print("Calculated from nanoindentation software")
print(f"Eeff_file={meta['Eeff_file_Pa']:.1f} Pa, E_v0.5_file={meta.get('E_v05_file_Pa', float('nan')):.1f} Pa")
print(f"Load start={load_start:.1e} N, Load end={load_end:.1e} N Relaxation Fraction={relax_frac:.3f}")
    