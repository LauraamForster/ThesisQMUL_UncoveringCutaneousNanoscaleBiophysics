# save cropped image of section indented, along with 
# "MatrixScan7 S-1 E-eff vs XY position.txt" file into same folder as pyhthon
# file. Code takes Eff YM and creates a heatmap of YM and overlays this 
# with image

import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import data, io, filters
import matplotlib.image as mpimg
from PIL import Image
import cv2

directory = '/Users/lauraforster/Documents/Uni/3 - PhD/Nanoindentation/NI_Data/Data/Bleomycin/2w/48P/48P_1'
os.chdir(directory)

text = "E[eff] (Pa)" #find YM value in each .txt file
img_name = "1_48P.png" #microscope image

YM = []

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file = open(filename)
        lines = file.readlines()
        for line in lines:
            if text in line:
                YM.append(float(line[12:]))

YM.reverse()
YM2 = []

for filename in os.listdir(directory):
    if "XY" in filename:
        file = open(filename)
        lines = file.readlines()
        for line in lines:
            length = len(lines)
            line2 = line.split()
            if 'Stepsize' not in line2:
                YM2.append(line2)

YM3 = np.array(YM2, dtype=float)
YM4 = (np.fliplr(YM3))[::-1]

print(YM4)

# fig, ax=plt.subplots()
# im = ax.imshow(YM4)
# ax.set_xticks([])
# ax.set_yticks([])
# fig.colorbar(im, orientation='vertical')
# plt.show()

directory = '/Users/lauraforster/Documents/Uni/3 - PhD/Nanoindentation/NI_Data/Data/Bleomycin/2w/48P'
os.chdir(directory)

fig, ax = plt.subplots()
img0 = ax.imshow(mpimg.imread(img_name))
img1 = ax.imshow(YM4, alpha=0.7, extent=img0.get_extent())
plt.show()



