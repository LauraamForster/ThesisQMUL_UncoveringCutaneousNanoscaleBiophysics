# creates a colour heat map of Eff YM values from the 
# "MatrixScan7 S-1 E-eff vs XY position.txt" file 
# can also display in log values

import numpy as np
import os
import matplotlib.pyplot as plt


# # directory = os.getcwd()
# directory = '/Users/lauraforster/Desktop/Matrix'
# # directory = '/Users/lauraforster/Documents/Uni/3 - PhD/Nanoindentation/NI_Data/Data/Protocol Testing/testglass long scan/Long Test  Glass Cross'
# os.chdir(directory)

# YM2 = []

# for filename in os.listdir(directory):
#     if "Grid 1 E-eff.txt" in filename:
#         file = open(filename)
#         lines = file.readlines()
#         for line in lines:
#             length = len(lines)
#             line2 = line.split()
#             if 'Stepsize' not in line2 and 'YM' not in line2:
#                 YM2.append(line2)

# Set the directory where the file is located
# directory = '/Users/lauraforster/Desktop/Matrix'
directory = '/Users/lauraforster/Documents/Uni/3 - PhD/Nanoindentation/NI_Data/Data/Wounding/Compare/Day14 3'
os.chdir(directory)

# Provide the filename
for file_name in os.listdir(directory):
    if "position" in file_name:
        filename = file_name
        print(filename)
# filename = 'ap1 f32 S-1 E-eff vs XY position.txt'

YM2 = []

# Check if the file exists in the directory
if filename in os.listdir(directory):
    with open(filename) as file:
        lines = file.readlines()
        for line in lines:
            line2 = line.split()
            if 'Stepsize' not in line2 and 'YM' not in line2:
                YM2.append(line2)
else:
    print(f"{filename} not found in the directory")
    
YM4 = np.array(YM2, dtype=float)
# YM4 = YM4[::-1] 
YM4=YM4.T
# YM4 = YM4[::-1]
# YM4=YM4.T

fig, ax=plt.subplots()
# fig, ax=plt.subplots(figsize=(50,50))
im = ax.imshow(YM4, vmin =0, vmax = 10000)
# im = ax.imshow(YM4)
ax.tick_params(axis=u'both', which=u'both',length=0)
plt.axis('off')
# fig.colorbar(im, orientation='vertical')
# cbar = fig.colorbar(im, orientation='vertical')
plt.subplots_adjust(left=1, right=10, top=10, bottom=1)

# ticks = np.linspace(0, 10000, num=5)  # Create ticks from 0 to 20000
# cbar.set_ticks(ticks)
# cbar.set_ticklabels([f'{int(tick/1000)}' for tick in ticks])  # Convert ticks to kPa

plt.show()

# # print(YM4) b
# logYM = np.log(YM4)

# fig, ax=plt.subplots()
# im = ax.imshow(logYM)
# ax.tick_params(axis=u'both', which=u'both',length=0)
# cbar = fig.colorbar(im, orientation='vertical')
# cbar.set_label('Eff YM (Pa)',size=10)
# plt.show()





