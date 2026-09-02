# takes single load vs indentation graph, splits into loading and unloading and does 
# exponential fit

import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import sys
from lmfit import Model, Parameters, report_fit

#-------File Extraction--------

#open a specfic file
file = open('MatrixScan7 S-1 X-1 Y-1 I-1.txt')
#read file (ignoring header containing info on experiment)
lines = file.readlines()[37:]
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

# plt.plot(loading_indent, loading_load, label='loading curve')
# plt.plot(holding_indent, holding_load, label='holding curve')
# plt.plot(unloading_indent, unloading_load, label='unloading curve')
# plt.legend()
# plt.show()

#-------Loading Data Fitting--------

# #define the x and y axes as only the loading data

unloading_indent_rev = unloading_indent[::-1]+loading_indent[-1]


indentationData_concat = loading_indent, unloading_indent_rev
indentationData_concat = np.concatenate(indentationData_concat).ravel()

loadingData_concat = loading_load, unloading_load[::-1]
loadingData_concat = np.concatenate(loadingData_concat).ravel()
loadingData_concat = np.float128(loadingData_concat)

plt.plot(indentationData_concat, loadingData_concat, label='concat curve')
plt.legend()
plt.show()

#-------Shared Fitting--------


join_point = loading_indent[-1]
dx2 = 0.02+join_point

def funccat(x,A1,b,c,D2, e, f, x1start, x1end, x2start, x2end,dx2, join_point):
    xunscal = np.where(x<=join_point,x1start + x*(x1end-x1start),x2start + (x-join_point-dx2)*(x2end-x2start))
    rval = np.where(x<=join_point,exponential_load(xunscal, A1, b, c), exponential_unload(xunscal,D2,e,f))
    return rval

def exponential_load(x, A1, b, c):
	return(A1 * np.exp(0.0001*x*b) + c)

def exponential_unload(x, D2, e, f):
	return(D2 * np.exp(0.0001*x*e) + f)

funcJoin = Model(funccat)
parsJoin = Parameters()
parsJoin.add('A1',value=0.07)
parsJoin.add('b',value=0.00015)
parsJoin.add('c',value=-0.1)
parsJoin.add('D2',value=0.02)
parsJoin.add('e',value=0.0001)
parsJoin.add('f',value=-0.3)
parsJoin.add('x1start',value=loading_indent[0],vary=False)
parsJoin.add('x1end',value=loading_indent[-1],vary=False)
parsJoin.add('x2start',value=unloading_indent[0],vary=False)
parsJoin.add('x2end',value=unloading_indent[-1],vary=False)
parsJoin.add('dx2',value=dx2,vary=False)
parsJoin.add('join_point',value=join_point,vary=False)

# print(funcJoin.params)

initJoin = funcJoin.eval(parsJoin,x=indentationData_concat)
# plt.plot(indentationData_concat,initJoin,'b--', label='int guess')
fitJoin = funcJoin.fit(loadingData_concat,parsJoin,x=indentationData_concat)
finalJoin = funcJoin.eval(fitJoin.params,x=indentationData_concat)
plt.plot(indentationData_concat,finalJoin,'r-', label='finalfit')
plt.plot(indentationData_concat, loadingData_concat, label='concat curve')
plt.ylabel('Load  (uN)')  
plt.xlabel('Indentation  (nm)')  
plt.title('concatenated Loading and Unloading curves with shared exponential fit')
plt.legend()
plt.show()

print(fitJoin.fit_report())
