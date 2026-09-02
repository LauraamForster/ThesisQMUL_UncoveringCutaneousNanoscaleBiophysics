import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on your setup

import matplotlib.image as mpimg
import tkinter as tk
from tkinter import simpledialog
import pandas as pd
import numpy as np
import math
from matplotlib.widgets import Button

# === USER INPUTS ===
image_sample_number = 305  # e.g. '1' for '1.jpg'
typetype = 'PBS'
image_path = f"/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Raman Data/Data/{typetype}/images/{image_sample_number}.jpg"
csv_path = "/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Papers and Presentations/Samples/RamanManifest.csv"  # Replace with your CSV file
txt_path = f"/Users/lauraforster/Documents/Uni/3 - PhD/Raman/Raman Data/Data/{typetype}/{image_sample_number}A_linescan1.txt"

# === Load Image and CSV ===
img = mpimg.imread(image_path)
df = pd.read_csv(csv_path)  # uses comma by default
df.columns = df.columns.str.strip()

# Load Raman .txt to determine scan points
raman_df = pd.read_csv(txt_path, sep='\t', comment='#', names=['X', 'Y', 'Wavelength', 'Intensity'])
unique_coords = raman_df[['X', 'Y']].drop_duplicates().reset_index(drop=True)

# Estimate step size (in microns or same units as coords)
if len(unique_coords) >= 2:
    diffs = np.sqrt(np.diff(unique_coords['X'])**2 + np.diff(unique_coords['Y'])**2)
    step_size = np.round(np.median(diffs), 3)
    total_points = len(unique_coords)
    print(f"🟢 Total scan points: {total_points}")
    print(f"🟢 Estimated step size: {step_size} units")
else:
    step_size = None
    print("⚠️ Could not determine step size.")

# === Get line scan start/end from CSV
image_sample_number = int(image_sample_number)
row = df[df["Sample Number"] == image_sample_number]
if row.empty:
    raise ValueError(f"No matching row in CSV for Sample Number = {image_sample_number}")
    
start_x_real = row["start x"].values[0]
start_y_real = row["start y"].values[0]
end_x_real = row["end x"].values[0]
end_y_real = row["end y"].values[0]

# ============================================================
# GLOBAL STATE
# ============================================================
clicks = []
region_clicks = []
step = 0
map_x = None
map_y = None
inv_x = None
inv_y = None
fig = None
ax = None


region_prompts = [
    "Click START of left glass",
    "Click END of left glass / START of subcut",
    "Click END of subcut / START of dermis",
    "Click END of dermis / START of epidermis",
    "Click END of epidermis / START of right glass",
    "Click END of right glass",
]

region_length_cols = [
    "leng left glass",
    "leng subcut",
    "leng dermis",
    "leng epi",
    "leng right glass",
]

region_point_cols = [
    ("start x image", "start y image"),
    ("left glass x", "left glass y"),
    ("dermis x", "dermis y"),
    ("epi x", "epi y"),
    (" right glass x", " right glass y"),
    ("end x image", "end y image"),
]


def dist(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def reset_region_clicks(event=None):
    global region_clicks

    if step != 3:
        print("⚠️ Region clicking has not started yet.")
        return

    region_clicks.clear()

    # Clear the axes and redraw image + original Raman line
    ax.clear()
    ax.imshow(img)
    draw_line_scan()

    ax.set_title(region_prompts[0])
    fig.canvas.draw()

    print("\n🔄 Region clicks cleared.")
    print(f"🖱️ {region_prompts[0]}")

def update_manifest_with_regions():
    global df

    sample_mask = df["Sample Number"] == image_sample_number

    if not sample_mask.any():
        raise ValueError(f"No matching row for Sample Number = {image_sample_number}")

    # Save clicked real-world coordinates into manifest
    for point, (x_col, y_col) in zip(region_clicks, region_point_cols):
        df.loc[sample_mask, x_col] = point[0]
        df.loc[sample_mask, y_col] = point[1]

    # Save region lengths
    lengths = []
    for i, col in enumerate(region_length_cols):
        length = dist(region_clicks[i], region_clicks[i + 1])
        lengths.append(length)
        df.loc[sample_mask, col] = length

    # Save scan info
    df.loc[sample_mask, "no pts"] = total_points
    df.loc[sample_mask, "Step size"] = step_size
    df.loc[sample_mask, "verify"] = "yes"

    # Save back to CSV
    df.to_csv(csv_path, index=False)

    print("\n✅ Manifest updated for sample:", image_sample_number)
    print(f"Saved to: {csv_path}")

    for col, length in zip(region_length_cols, lengths):
        n_points = round(length / step_size) if step_size else "unknown"
        print(f"{col}: {length:.2f} µm ≈ {n_points} points")

def ask_for_values(axis):
    root = tk.Tk()
    root.withdraw()
    val1 = simpledialog.askfloat(f"{axis}-axis Calibration", f"Enter real-world value at Point 1 on {axis}-axis:")
    val2 = simpledialog.askfloat(f"{axis}-axis Calibration", f"Enter real-world value at Point 2 on {axis}-axis:")
    root.destroy()
    return val1, val2

def calibrate(pixel1, pixel2, real1, real2):
    px1, py1 = pixel1
    px2, py2 = pixel2
    dx = px2 - px1
    dy = py2 - py1
    real_dx = real2 - real1
    real_dy = real2 - real1

    def map_x(x_pix):
        return real1 + ((x_pix - px1) / dx) * real_dx

    def map_y(y_pix):
        return real1 + ((y_pix - py1) / dy) * real_dy

    def inv_x(x_real):
        return px1 + ((x_real - real1) / real_dx) * dx

    def inv_y(y_real):
        return py1 + ((y_real - real1) / real_dy) * dy

    return map_x, map_y, inv_x, inv_y

# === Draw Original Line from CSV ===
def draw_line_scan():
    start_x_pix = inv_x(start_x_real)
    start_y_pix = inv_y(start_y_real)
    end_x_pix = inv_x(end_x_real)
    end_y_pix = inv_y(end_y_real)

    # Draw the red line
    ax.plot([start_x_pix, end_x_pix], [start_y_pix, end_y_pix], 'r-', linewidth=2, label="Original Scan")

    # Add markers for start ('x') and end ('o')
    ax.plot(start_x_pix, start_y_pix, marker='x', color='black', markersize=10, label="Start")
    ax.plot(end_x_pix, end_y_pix, marker='o', color='black', markersize=10, label="End")

    ax.legend()
    fig.canvas.draw()

    

# === Draw Section Lines After Calibration ===
def draw_section_line(p1, p2, color):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linestyle='-', linewidth=2)
    fig.canvas.draw()
    
# === Interactive Click Handler ===
def onclick(event):
    global step, clicks, map_x, map_y, inv_x, inv_y, section_points

    if event.xdata is None or event.ydata is None:
        return

    if step < 2:
        print(f"Clicked {step + 1} for X-axis calibration: ({event.xdata:.2f}, {event.ydata:.2f})")
        clicks.append((event.xdata, event.ydata))
        step += 1
        if step == 2:
            real1, real2 = ask_for_values("X")
            map_x, _, inv_x, _ = calibrate(clicks[0], clicks[1], real1, real2)
            clicks.clear()
            step = 2.5
            print("Now click 2 points along the Y-axis scale.")
    elif step == 2.5:
        print(f"Clicked {len(clicks) + 1} for Y-axis calibration: ({event.xdata:.2f}, {event.ydata:.2f})")
        clicks.append((event.xdata, event.ydata))
        if len(clicks) == 2:
            real1, real2 = ask_for_values("Y")
            _, map_y, _, inv_y = calibrate(clicks[0], clicks[1], real1, real2)
            clicks.clear()
            step = 3
            print("Calibration complete. Line scan will be drawn.")
            draw_line_scan()
            print("🖱️ Now click along the scan to divide it into sections.")
    elif step == 3:
        if len(region_clicks) >= len(region_prompts):
            print("✅ All region points already clicked. Manifest has been updated.")
            return

        x_real = map_x(event.xdata)
        y_real = map_y(event.ydata)
        region_clicks.append((x_real, y_real))

        click_no = len(region_clicks)
        print(f"\n📍 {region_prompts[click_no - 1]}")
        print(f"Pixel: ({event.xdata:.2f}, {event.ydata:.2f})")
        print(f"Real:  ({x_real:.2f}, {y_real:.2f})")

        # Draw clicked point
        ax.plot(event.xdata, event.ydata, marker='o', color='black', markersize=6)

        # Draw segment if this is not the first point
        if click_no >= 2:
            prev_real = region_clicks[-2]
            curr_real = region_clicks[-1]

            prev_pix = (inv_x(prev_real[0]), inv_y(prev_real[1]))
            curr_pix = (inv_x(curr_real[0]), inv_y(curr_real[1]))

            cmap = plt.get_cmap("tab10")
            color = cmap((click_no - 2) % 10)

            ax.plot(
                [prev_pix[0], curr_pix[0]],
                [prev_pix[1], curr_pix[1]],
                color=color,
                linewidth=3,
            )

            seg_length = dist(prev_real, curr_real)
            n_points = round(seg_length / step_size) if step_size else "unknown"

            print(f"🟦 Segment {click_no - 1}: {seg_length:.2f} µm ≈ {n_points} points")

        fig.canvas.draw()

        # Print next instruction or save when complete
        if click_no < len(region_prompts):
            print(f"➡️ Next: {region_prompts[click_no]}")
        else:
            update_manifest_with_regions()

# ============================================================
# SHOW IMAGE
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Leave space at bottom for button
plt.subplots_adjust(bottom=0.18)

ax.imshow(img)
ax.set_title("Step 1: Click two points on X-axis scale")

fig.canvas.mpl_connect("button_press_event", onclick)

# Reset button
reset_ax = plt.axes([0.38, 0.04, 0.24, 0.06])
reset_button = Button(reset_ax, "Reset region clicks")
reset_button.on_clicked(reset_region_clicks)

plt.show()








