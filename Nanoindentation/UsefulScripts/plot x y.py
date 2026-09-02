import os
import csv
import sys
import numpy as np
import math 
from lmfit import Model, Parameters
from lmfit.models import PolynomialModel
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy import signal
from scipy.signal import find_peaks
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
import time

# ----------------------------------------------------------------------------------------------------------------------------------------
# DEFINING INPUT AND OUTPUT FOLDERS
# ----------------------------------------------------------------------------------------------------------------------------------------

input_folder = ('/Users/lauraforster/Documents/Uni/3 - PhD/Nanoindentation/NI_Data/Data/Paper/test/test/real') #find the data

# ----------------------------------------------------------------------------------------------------------------------------------------
# DATA MANIPULATING FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------------------------

def NEWSplit(indentation, time, load, piezo, radius, date, timeT, file_name): 
    loading_indent, loading_load, unloading_indent, unloading_load, holding_indent, holding_load = [], [], [], [], [], []
    time_load, time_unload, time_hold, piezo_load, piezo_unload, piezo_hold = [], [], [], [], [], [] 
    rampup_indent, rampdown_indent, rampup_load, rampdown_load, rampup_time, rampdown_time, rampup_piezo, rampdown_piezo = [], [], [], [], [], [], [], []
    piezo_window = 200
    
    indentation = np.array(indentation)
    time = np.array(time)
    piezo = np.array(piezo)
    load = np.array(load)
    
    # plt.plot(time, indentation, label='Approach to surface', color='orange')
    
    # # plt.gca().xaxis.set_major_formatter(FuncFormatter(format_kilo_pascals))
    # plt.legend()
    # plt.xlabel('Time (s)')
    # plt.ylabel('indentation (nm)')
    # plt.title(f'{radius, date, timeT} Time vs Indentation')
    # # pdf.savefig()
    # # plt.close()
    # plt.show()
    if file_name == 'Sample.txt':
        piezo = piezo - 5000
    plt.plot(piezo, load, label=f'{file_name}')
    plt.xlim(10000,30000)
    
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(format_kilo_pascals))
    plt.legend()
    plt.xlabel('piezo (nm)')
    plt.ylabel('Load (N)')
    plt.title(f'Piezo vs Load')
    # pdf.savefig()
    # plt.close()
    # plt.show()

    return holding_indent, holding_load, loading_indent, loading_load, unloading_indent, unloading_load, time_hold, time_load, time_unload, piezo_load, piezo_unload, piezo_hold, rampup_indent, rampdown_indent, rampup_load, rampdown_load, rampup_time, rampdown_time, rampup_piezo, rampdown_piezo
 
def subbaseline(data, load_data, slope, y_intercept):
     load_baseline = slope*data + y_intercept
     return load_data - load_baseline
 
def extract_xy_numbers_from_header(header):#Extract the x and y numbers from the header.
    lines = header.split('\n')
    scan_line = lines[2]  
    _, _, _, x_number, _, y_number, _, _ = scan_line.split('\t')
    date = lines[0].split()[1]
    timeT = lines[0].split()[3]
    # print(timeT)
    radius_line = lines[11]
    radius = radius_line.split('\t')[1]
    lines2 = header.split("\n")
    for line in lines2:
        if "E[eff] (Pa)" in line:
            value = line.split(":")[-1].strip()
            Eff = float(value.split("\t")[1])
        if "E[v=0.500]" in line:
            value = line.split(":")[-1].strip()
            E = float(value.split("\t")[1])
    return int(x_number), int(y_number), float(radius), float(lines[22].split('\t')[1]), Eff, E, date, timeT

def extract_xy_key(file_name):
    parts = file_name.split(' ')
    xy_parts = [part for part in parts if part.startswith('X-') or part.startswith('Y-')]
    x, y = (0, 0)  # Default values if X or Y is not found
    for part in xy_parts:
        if part.startswith('X-'):
            x = int(part.split('-')[1])
        elif part.startswith('Y-'):
            y = int(part.split('-')[1])
    return x, y

# ----------------------------------------------------------------------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------------------------

def format_kilo_pascals(value, _):
    return f"{value / 1000:.1f}" 


# ----------------------------------------------------------------------------------------------------------------------------------------
# DATA FROM FILES
# ----------------------------------------------------------------------------------------------------------------------------------------

data = {}
dataOP = {}

file_list = os.listdir(input_folder)
sorted_file_list = sorted(file_list, key=extract_xy_key)

for file_name in sorted_file_list:
    if file_name.endswith(".txt") and "position" not in file_name:
        file_path = os.path.join(input_folder, file_name)

        with open(file_path, "r") as file:
            lines = file.readlines()
        header = ""
        data_rows = []

        in_data_section = False
        for line in lines:
            if line.strip() == "Time (s)\tLoad (uN)\tIndentation (nm)\tCantilever (nm)\tPiezo (nm)\tAuxiliary":
                in_data_section = True
                continue
            if in_data_section:
                data_rows.append(line.strip())
            else:
                header += line
                
        x_number, y_number, radius, h_max, eff_modulus, HertzYM, date, timeT = extract_xy_numbers_from_header(header)
        # print(timeT)# Extract X and Y numbers from the header
        load_data, indentation_data, time_data, piezo = [], [],[],[]

        for row in data_rows:# Extract load and indentation data from rows
            row_values = row.split("\t")
            load_data.append(float(row_values[1]))  # Load (uN) column
            indentation_data.append(float(row_values[2]))  # Indentation (nm) column
            time_data.append(float(row_values[0])) # Time (s) column
            piezo.append(float(row_values[4]))

        file_name=os.path.basename(file_path)
        holding_indent, holding_load, loading_indent, loading_load, unloading_indent, unloading_load, time_hold, time_load, time_unload, piezo_load, piezo_unload, piezo_hold, rampup_indent, rampdown_indent, rampup_load, rampdown_load, rampup_time, rampdown_time, rampup_piezo, rampdown_piezo = NEWSplit(indentation_data, time_data, load_data, piezo, radius, date, timeT, file_name)


            

    
    
    
    
    
    
    
    
    
    

