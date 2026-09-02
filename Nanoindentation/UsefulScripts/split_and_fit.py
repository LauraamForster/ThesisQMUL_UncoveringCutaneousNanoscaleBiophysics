# takes single load vs indentation graph, splits into loading and unloading and does 
# exponential fit

import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from lmfit import Model, Parameter, report_fit, Parameters
import os
import sys


# #-------File Extraction--------

directory = '/Users/lauraforster/Documents/Uni/3 - PhD/Nanoindentation/NI_Data/Data/Bleomycin/2w/48P/48P_1'
os.chdir(directory)

# List all files in the directory
files = sorted(os.listdir(directory))

# Filter the list to only include files ending with ".txt"
txt_files = [file for file in files if file.endswith('.txt')]

# Check if there are any .txt files in the directory
if not txt_files:
    print("No .txt files found in the directory.")
else:
    print("List of .txt files in the directory:")
    for i, txt_file in enumerate(txt_files):
        print(f"{i + 1}: {txt_file}")

    # Ask the user to select a file by entering the index
    while True:
        try:
            selected_index = int(input("Enter the index of the file you want to open: ")) - 1
            if 0 <= selected_index < len(txt_files):
                selected_file = txt_files[selected_index]
                break
            else:
                print("Invalid index. Please enter a valid index.")
        except ValueError:
            print("Invalid input. Please enter a valid index as a number.")

    # Now, you can open the selected file
    selected_file_path = os.path.join(directory, selected_file)
    with open(selected_file_path, 'r') as file:
        # Perform operations on the selected file here
        # ...
        #read file (ignoring header containing info on experiment)
        linesall = file.readlines()
        lines = linesall[37:]
        linesheader = linesall
        #set some empty lists
        time, load, indentation, cant, piezo, aux = [], [], [], [], [], []

#-------Data Storing--------

#put each column of data into labelled list
length = len(lines)
for x in range(length):
	a = lines[x].split()
	time.append(float(a[0]))
	load.append(float(a[1]))
	indentation.append(float(a[2]))
	cant.append(float(a[3]))
	piezo.append(float(a[4]))
	aux.append(float(a[5]))

# plot load vs indentation graphs
plt.plot(indentation, load)
plt.ylabel('Load  (uN)')  
plt.xlabel('Indentation  (nm)')  
plt.title('Load vs Indentation curve for data')
plt.show()

#-------Data Splitting--------
#find max point on indentation axes and split data into loading, holding and unloading
loading_indent, loading_load, unloading_indent, unloading_load, holding_indent, holding_load, holding_indent_value = [], [], [], [], [], [], []
length_indent = len(indentation)

indentation = np.array(indentation)
maxi = max(indentation)
maxi2 = math.floor(maxi-10)
maxi3 = math.ceil(maxi+10)

for x in range(length_indent):
	if int(indentation[x]) in range(maxi2, maxi3):
		holding_indent.append(indentation[x])
		holding_load.append(load[x])
		holding_indent_value.append(x)
for x in range(length_indent):
	if x < holding_indent_value[0]:
		loading_indent.append(indentation[x])
		loading_load.append(load[x])
for x in range(length_indent):
	if x > holding_indent_value[-1]:
		unloading_indent.append(indentation[x])
		unloading_load.append(load[x])

plt.scatter(loading_indent, loading_load, label='loading curve', s=1)
plt.scatter(unloading_indent, unloading_load, label='unloading curve', s=1)
plt.scatter(holding_indent,holding_load, label='holding curve', s=1)
plt.legend()
plt.ylabel('Load  (uN)')  
plt.xlabel('Indentation  (nm)')  
plt.title('Load vs Indentation curve')
plt.show()

#-------Extract Values--------

header = ""
data_rows = []
# eff_modulus, x_number, y_number = [], [], []
in_data_section = False
for line in linesheader:
    if line.strip() == "Time (s)\tLoad (uN)\tIndentation (nm)\tCantilever (nm)\tPiezo (nm)\tAuxiliary":
        in_data_section = True
        continue
    if in_data_section:
        data_rows.append(line.strip())
    else:
        header += line
        
lines = header.split('\n')
scan_line = lines[2]  # Assuming the line containing the X and Y numbers is at index 2
_, _, _, x_number, _, y_number, _, _ = scan_line.split('\t')
radius_line = lines[11]
radius = radius_line.split('\t')[1]
lines = header.split("\n")
for line in lines:
    if "E[eff] (Pa)" in line:
        value = line.split(":")[-1].strip()
        Eff = float(value.split("\t")[1])
    if "E[v=0.500]" in line:
        value = line.split(":")[-1].strip()
        E = float(value.split("\t")[1])


#-------Loading Data Fitting--------

# #define the x and y axes as only the loading data
x_load = np.array(loading_indent)
y_load = np.array(loading_load)

def eqnHertz(x, E, x0):   
    # P = (4/3)*(E/(1-(0.5**2)))*(np.sqrt(25e-6))*(x**(3/2))
    P = (4/3)*(E/(1-(0.5**2)))*(np.sqrt(25e-6))*((x-x0)**(3/2))
    return P

hertzmodel = Model(eqnHertz)
params = Parameters()

YM = (((3/4)*y_load[-1])*(1-(0.5**2)))/(np.sqrt(25e-6) * (x_load[-1]**(3/2)) ) 

params.add('E', value = YM)
params.add('x0', value = -3e-6, max=0)

xx = x_load[1000:]
yy = y_load[1000:]

hertzfit = hertzmodel.fit(y_load, x=x_load, params=params)
hertzfit2 = hertzfit.eval(params=hertzfit.params, x=x_load)

longline = np.linspace(-2e3, 6e3, 100)
hertzfit3 = hertzfit.eval(params=hertzfit.params, x=longline)

plt.plot(x_load, y_load, label='data')
plt.plot(longline, hertzfit3, label='long fit')
# plt.show()


# #-------Unloading Fitting--------

x_unload = np.array(unloading_indent)
y_unload = np.array(unloading_load)

v = 0.5  # Assuming Poisson's ratio for the material is 0.5 (for isotropic materials)
radius_R = float(radius) * 1e-6
epsilon = 0.75
Ei = 51e9
v_i = 0.2
P_max = np.max(y_unload)
h_max = float(linesheader[22].split('\t')[1])

# Plot the oliver pharr model
S, plus_C = np.polyfit(x_unload[0:500], y_unload[0:500], 1)
y = S*x_unload + plus_C

plt.plot(x_unload, y_unload, label='data')
plt.plot(unloading_indent[:1500], y[:1500], label='OP')
plt.legend()
plt.show()









