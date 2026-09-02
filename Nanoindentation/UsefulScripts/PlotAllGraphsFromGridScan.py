# plots all load v indentation graphs from matrix scan in one plot

import csv
import numpy as np
import matplotlib.pyplot as plt
import os

directory = os.getcwd()
fig, ax = plt.subplots()
for filename in os.listdir(directory):
	if filename.endswith(".txt"):
		file = open(filename)
		lines = file.readlines()[37:]
		length = len(lines)
		time, load, indentation, cant, piezo, aux = [], [], [], [], [], []
		for x in range(length):
			a = lines[x].split()
			load.append(float(a[1]))
			indentation.append(float(a[2]))
		# ax.plot(load, indentation, label=filename)
		ax.plot(indentation, load)
		leng = len(load)
		sums = sum(load)
		if sums > 0 and leng > 0 :
			av = sums/leng
			print("average load", av, "uN")
		leng2 = len(indentation)
		sums2 = sum(indentation)
		if sums2 > 0 and leng2 > 0 :
			av2 = sums2/leng2
			print("average indentation depth", av2, "nm")

# ax.legend()
plt.ylabel('load  uN')  
plt.xlabel('indentation  nm')  
plt.show() 



