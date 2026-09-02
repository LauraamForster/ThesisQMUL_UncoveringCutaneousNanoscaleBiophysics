# ThesisQMUL_UncoveringCutaneousNanoscaleBiophysics
Thesis by Laura Forster and Himadri Gupta

Analysis code for Laura Forster's thesis work with Prof. Himadri Gupta and Dr. Emanuel Rognoni at Queen Mary University of London (QMUL).

The repository contains separate processing workflows for three experimental techniques:

- **Nanoindentation** — mechanical and viscoelastic properties
- **Raman spectroscopy** — biochemical composition and spectral features
- **Small-angle X-ray scattering (SAXS)** — collagen structure and orientation

The outputs from these technique-specific workflows are brought together in **`Final analysis/`** for experiment-level, multi-technique analysis.

> **Important:** this repository contains analysis code and manifests, but not the full raw datasets. Several scripts currently contain absolute paths - update the paths and check the settings at the top of a driver script before running it.

## Repository structure

```text
.
├── Nanoindentation/       Raw curve fitting and technique-level analysis
├── Raman/                 Raman preprocessing, peak analysis and PCA
├── SAXS/                  SAXS reduction, fitting and visualisation
├── Final analysis/        Export and combined multi-technique analysis
├── OtherTechniques/       Supporting and laboratory SAXS scripts/data
└── README.md
```

Each main technique uses the same broad pattern:

```text
raw exported data → driver script → technique-specific functions → processed outputs
                                                                    ↓
                                                          Final analysis export
                                                                    ↓
                                                     combined figures/statistics
```

## Before running the code

### 1. Create a Python environment

The scripts use the following main packages:

numpy
pandas
matplotlib
scipy
scikit-learn
statsmodels
lmfit
openpyxl
h5py
pyFAI
ramanspy
tkinter

### 2. Provide the data and update paths

1. Obtain the relevant raw data, processed data and generate (or edit existing) manifest files. 
2. Open the relevant driver script.
3. Replace the absolute `DataDir`, manifest, calibration, input and output paths with current paths.
4. Check the experiment, subtype, region, scan number and plotting settings near the top of the script.
5. Run the script from its own directory so that its companion function module can be imported.


## Workflow 1: Nanoindentation

### A. Fit raw indentation curves

**Driver:** `Nanoindentation/Call_NI_Model_Functions2.py`  
**Functions:** `Nanoindentation/NI_Model_Functions2.py`

The driver reads raw text exports from a matrix or line scan, then:

1. filters the raw load–piezo signal;
2. splits the curve into experimental phases;
3. finds the contact point;
4. applies Hertz and Oliver–Pharr fits;
5. fits viscoelastic behaviour during the hold phase;
6. calculates loading stress–strain values; and
7. writes fitted parameters to CSV and diagnostic plots to PDF.

Edit the settings near the top of the driver:
`group`
`settype`
`toscan`
`foldergroup`
`region`
input/output paths
fitting parameters

### B. Compare samples and conditions

**Driver:** `Nanoindentation/Call_NI_Analysis_Functions3.py`  
**Functions:** `Nanoindentation/NI_Analysis_Functions4.py`  
**Manifests:** `Nanoindentation/AP1Manifest.csv`, `BleomycinManifest.csv`, `JEBManifest.csv`, and `woundingManifest6.csv`

This stage reads the fitted CSV files through a manifest, trims and normalises line-scan positions, divides the dermis into spatial bins, and produces comparisons such as bar/violin plots, spatial trends, correlations, PCA and statistical models.

Check:
`set_type`
`manifest_path`
`base_path`
`regions`
`groups`
`nbins`
`layer`
selected plot variables

## Workflow 2: Raman spectroscopy

Raman scripts are organised as driver/function pairs for different datasets:

| Dataset or purpose | Driver | Function module |
|---|---|---|
| Main workflow | `RamanDriver.py` | `RamanFunctions2.py` |
| Updated workflow | `RamanDriver_NEW2.py` | `RamanFunctions_NEW2.py` |
| Wounding | `RamanDriver_wounds.py` | `RamanFunctions_wounds.py` |
| Bovine comparison | `RamanDriver_Bovine2.py` | `RamanFunctions_Bovine2.py` |

The selected driver reads spectral files, a sample manifest and a peak manifest. Depending on its settings, it then:

1. builds a sample dictionary;
2. trims spectra to the tissue region of interest;
3. despikes, smooths, baseline-corrects and normalises spectra;
4. separates fingerprint and extended spectral regions;
5. bins spectra along the line scan;
6. calculates peak-region or fitted-component measurements; and
7. produces average spectra, PCA/loadings and other diagnostic plots.

Before running, set:
`DataDir`
`PeakDir`
`ManifestDir`
`Save_folder`
`Type`
spectral ranges`
`NBins`
preprocessing choices
plot modes

`Raman/RamanCutting3.py` is an interactive helper for inspecting/cutting Raman maps into the different skin layers (run via terminal if needed)

## Workflow 3: SAXS

**Main driver:** `SAXS/SAXS_DriverFile.py`  
**Reduction:** `SAXS/ReductionScript.py`  
**Fitting:** `SAXS/FittingScript_add4.py`  
**Visualisation:** `SAXS/VisualisingScript2.py`

The SAXS workflow processes Diamond Light Source I22 NeXus (`.nxs`) data:

1. load the detector mask and calibration;
2. reduce detector frames to radial intensity, `I(q)`, and/or azimuthal intensity, `I(chi)`;
3. save reduced data as NeXus files;
4. fit structural and orientational features;
5. write fitted measurements to CSV; and
6. create maps and diagnostic plots.

Before running, check:
the experiment and scan definitions
input/output paths
mask and calibrant paths (these are specific to each experiment date)
reduction flags
q/chi ranges
fitting thresholds
optional frame selections near the top of `SAXS_DriverFile.py`.

The SAXS scripts import `Utils.py`, which is currently stored in `SAXS/HelpfulExtras/`

Scripts in `SAXS/HelpfulExtras/` and `SAXS/RadiationDamage/` are supporting scripts for things like colourmap visualisation without having to run the entire pipeline. 

## Workflow 4: Final multi-technique analysis

The final analysis has two stages: **export**, then **analysis**.

### A. Build standardised multi-technique workbooks

Choose the export script that matches the experiment:

| Experiment | Export script | Companion functions | Output folder configured in the script |
|---|---|---|---|
| Wounding | `AllAnalysisExport.py` | `AllAnalysisExport_functions.py` | `Multitech_Export` |
| AP1 | `AllAnalysisExport_ap.py` | `AllAnalysisExport_ap_functions.py` | `Multitech_Export_ap1` |
| Bleomycin | `AllAnalysisExport_bleo.py` | `AllAnalysisExport_bleo_functions.py` | `Multitech_Export_bleo` |
| JEB | `AllAnalysisExport_jeb.py` | `AllAnalysisExport_jeb_functions.py` | `Multitech_Export_jeb` |

These scripts read the Raman raw data/manifests, nanoindentation fitted CSVs/manifests, SAXS fitted CSVs/ROI manifests and cell-density spreadsheet. They harmonise subtype and region labels and write per-subtype Excel workbooks. Depending on the experiment, workbook sheets include `Raman`, `RawRaman`, `Cell`, `SAXS`, `Nanoindentation` and `RawNanoindentation`.

Update every path in the script's initial **paths** section before running it.

### B. Generate combined figures and statistics

Run the matching analysis script after the export is complete:

| Experiment | Final analysis script |
|---|---|
| Wounding | `Final analysis/AllAnalysis_Wounding.py` |
| AP1 | `Final analysis/AllAnalysis_AP1.py` |
| Bleomycin | `Final analysis/AllAnalysis_Bleo.py` |
| JEB | `Final analysis/AllAnalysis_JEB.py` |

These scripts read the standardised workbooks, apply experiment-specific labels and ordering, and generate the cross-technique plots and statistical comparisons used in the thesis analysis.


