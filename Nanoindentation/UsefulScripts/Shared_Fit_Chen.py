# takes single load vs indentation graph, splits into loading and unloading and does 
# exponential fit

import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import sys
from lmfit import Model, Parameters, report_fit

#--------------File Extraction---------------------------------------------

file = open('M2 S-1 X-1 Y-1 I-1.txt') #open a specfic file
linesopen = file.readlines() #read file (ignoring header containing info on experiment)
lines = linesopen[37:]
time, load, indentation, cant, piezo, aux = [], [], [], [], [], []#set some empty lists

#define some variables for later
radius = float(((linesopen[11]).split())[3]) / 1e6
Eeff = float(((linesopen[32]).split())[2])
Eeffv = float(((linesopen[33]).split())[2])

#--------------Data Storing---------------------------------------------

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

time, load, indentation, cant, piezo, aux = np.array(time), np.array(load), np.array(indentation), np.array(cant), np.array(piezo), np.array(aux)

#--------------Data Splitting---------------------------------------------
#find max point on indentation axes and split data into loading, holding and unloading
loading_indent, loading_load, unloading_indent, unloading_load, holding_indent, holding_load, holding_indent_value = [], [], [], [], [], [], []
time_load, time_unload, time_hold = [], [], [] 
length_indent = len(indentation)

maxi = max(indentation) #find highest point on indenatation curve
maxi2, maxi3 = math.floor(maxi-10),  math.ceil(maxi+10) #find intergers 10 either side of max

for x in range(length_indent):
	if int(indentation[x]) in range(maxi2, maxi3):
		holding_indent.append(indentation[x])
		holding_load.append(load[x])
		holding_indent_value.append(x)
		time_hold.append(time[x])
for x in range(length_indent):
	if x < holding_indent_value[0]:
		loading_indent.append(indentation[x])
		loading_load.append(load[x])
		time_load.append(time[x])
for x in range(length_indent):
	if x > holding_indent_value[-1]:
		unloading_indent.append(indentation[x])
		unloading_load.append(load[x])
		time_unload.append(time[x])

holding_indent, holding_load = np.array(holding_indent), np.array(holding_load) #holding curve indentation vs load
holding_indent = holding_indent/1e9
holding_load = holding_load/1e6
loading_indent, loading_load = np.array(loading_indent), np.array(loading_load) # loading curve indentation vs load
loading_indent = loading_indent/1e9
loading_load = loading_load/1e6
unloading_indent, unloading_load = np.array(unloading_indent), np.array(unloading_load) #unloading curve indentation vs load
unloading_indent = unloading_indent/1e9
unloading_load = unloading_load/1e6
time_hold, time_load, time_unload = np.array(time_hold), np.array(time_load), np.array(time_unload) #time curves for load hold unload

#--------------Concat data ---------------------------------------------
timediff = time_hold[-1] - time_hold[0]
unloading_time_rev = time_unload - timediff

timeData_concat = time_load, unloading_time_rev
timeData_concat = np.concatenate(timeData_concat).ravel()

loadingData_concat = loading_load, unloading_load
loadingData_concat = np.concatenate(loadingData_concat).ravel()
loadingData_concat = np.float128(loadingData_concat)

# plt.plot(timeData_concat, loadingData_concat)
# plt.show()
# sys.exit()
#--------------Defining variables ---------------------------------------------
join_point = time_load[-1]
# joint_point2 = time_hold[-1]
dx2 = 0.02+join_point
# dx3 = 0.02+joint_point2

# ---------finding tau
loadstart = loading_load[0] #find the first value of the holding load curve 
loadend = loading_load[-1] #find the first value of the holding load curve 
load_diff = loadstart - loadend #difference in load values where load has decayed by half
exps = 1/(np.exp(1))
load_div = load_diff * exps
load_amp = loadend + load_div
valuetau = min(enumerate(loading_load), key=lambda x: abs(x[1]-load_amp))
tau_calc = time_load[valuetau[0]] 
time_start = time_unload[0]
tau = tau_calc - time_start

#---------estimate g values for fit
ginf = 0.6
g1 = 0.4

#---------defining remaining variables / values
v2 = 0.49**2 #poisson ratio squared (taken from literature) kg/m3
rootR = np.sqrt(radius) #radius in m
delta_Sq = np.power(loading_indent, 1.5) #indentation depth in m (depends on t)
delta_HC_Sq = np.average(delta_Sq)
t1 = time_unload[0] #initial time of holding curve
time = time_unload # time in s 
Pt = holding_load[0] #force in N (depends on t)
A = Pt/ginf #first part of equation (depends on t due to force and indentation depth varying)

# #--------------Shared Fitting---------------------------------------------
def funccat(x, A, ginf, tau, g1, t1, A2, ginf2, tau2, g12, t12, x1start, x1end, x2start, x2end,dx2, join_point):
    xunscal = np.where(x<=join_point,x1start + x*(x1end-x1start),x2start + (x-join_point-dx2)*(x2end-x2start))
    rval = np.where(x<=join_point,exponential_load(xunscal, A2, ginf2, tau2, g12, t12), exponential_unload(xunscal, A, ginf, tau, g1, t1))
    return rval

def exponential_load(x, A2, ginf2, tau2, g12, t12):
	full_term = A2*(ginf2+g12*(np.exp(-x/tau2))*(np.exp(t12/tau2)-1))
	return(full_term)

def exponential_unload(x, A, ginf, tau, g1, t1):
	full_term = A*(ginf+g1*(np.exp(-x/tau))*(np.exp(t1/tau)-1))
	return(full_term)
	
funcJoin = Model(funccat)
parsJoin = Parameters()
parsJoin.add('A',value=A)
parsJoin.add('ginf',value=ginf)
parsJoin.add('tau',value=tau, min=0.1)
parsJoin.add('g1',value=g1, expr='1-ginf')
parsJoin.add('t1',value=t1, vary=False)
parsJoin.add('A2',value=A)
parsJoin.add('ginf2',value=ginf)
parsJoin.add('tau2',value=tau, min=0.1)
parsJoin.add('g12',value=g1, expr='1-ginf')
parsJoin.add('t12',value=t1, vary=False)
parsJoin.add('x1start',value=time_load[0],vary=False)
parsJoin.add('x1end',value=time_load[-1],vary=False)
parsJoin.add('x2start',value=time_unload[0],vary=False)
parsJoin.add('x2end',value=time_unload[-1],vary=False)
parsJoin.add('dx2',value=dx2,vary=False)
parsJoin.add('join_point',value=join_point,vary=False)

# timediff = time_hold[-1] - time_hold[0]
# unloading_time_rev = time_unload - timediff

# timeData_concat = time_load, unloading_time_rev
# timeData_concat = np.concatenate(timeData_concat).ravel()

# loadingData_concat = loading_load, unloading_load
# loadingData_concat = np.concatenate(loadingData_concat).ravel()
# loadingData_concat = np.float128(loadingData_concat)
# timeData_concat, loadingData_concat

initJoin = funcJoin.eval(parsJoin,x=timeData_concat)
fitJoin = funcJoin.fit(loadingData_concat, parsJoin,x=timeData_concat)
finalJoin = funcJoin.eval(fitJoin.params,x=timeData_concat)

#---------Plot the Fit
plt.plot(timeData_concat,finalJoin,'r-', label='finalfit')
plt.plot(timeData_concat, loadingData_concat, label='concat curve')
plt.ylabel('Load  (uN)')  
plt.xlabel('time  (e)')  
plt.title('concatenated Loading and Unloading curves with shared exponential fit')
plt.legend()
plt.show()

#---------Fit Report
print("fit report")
print(fitJoin.fit_report())
print(' ')
print(' ')

