# takes single load vs indentation graph, splits into loading and unloading and does 
# exponential fit

import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy
from lmfit import Model, Parameters, report_fit


def funccat(x,A1,b,c,D2, e, f, x1start, x1end, x2start, x2end,dx2, join_point):
    xunscal = np.where(x<=join_point,x1start + x*(x1end-x1start),x2start + (x-join_point-dx2)*(x2end-x2start))
    rval = np.where(x<=join_point,exponential_load(xunscal, A1, b, c), exponential_unload(xunscal,D2,e,f))
    return rval

def function_load(x, A1, b, c):
	return(A1 * b*(x**2) + c)

def function_unload(x, D2, e, f):
	return(D2 * e*(x**2) + f)

def exponential_load(x, A1, b, c):
	return(A1 * np.exp(x*b) + c)

def exponential_unload(x, D2, e, f):
	return(D2 * np.exp(x*e) + f)

#-------Data Storing--------
join_point = 10
join_point_over=join_point+0.1

loading_indent = np.linspace(1, join_point_over, 10)
loading_load = function_load(loading_indent, A1=30, b=2, c=1)
loading_load_noise = np.random.normal(size=len(loading_indent), scale=1)
loading_load = loading_load + loading_load_noise

# holding_indent = np.linspace(join_point,join_point, 10)
# holding_load = np.linspace(20, 50, 10)
# holding_load_noise = np.random.normal(size=len(holding_indent), scale=1)
# holding_load = holding_load + holding_load_noise

unloading_indent = np.linspace(1, join_point, 10)
unloading_load = function_unload(unloading_indent, D2=20, e=1, f=1)
unloading_load_noise = np.random.normal(size=len(unloading_indent), scale=1)
unloading_load = unloading_load + unloading_load_noise


indentation = loading_indent, unloading_indent[::-1]
indentation = np.concatenate(indentation).ravel()

load = loading_load, unloading_load[::-1]
load = np.concatenate(load).ravel()

# plt.plot(indentation, load)
# plt.show()

loading_indent, loading_load, unloading_indent, unloading_load, holding_indent, holding_load, holding_indent_value = [], [], [], [], [], [], []
length_indent = len(indentation)

for x in range(length_indent):
	if indentation[x] - 1 <= max(indentation) <= indentation[x] + 1:
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

x_load = loading_indent
y_load = loading_load
x_unload = unloading_indent
y_unload = unloading_load

unloading_indent_rev = unloading_indent[::-1]+loading_indent[-1]

indentationData_concat = loading_indent, unloading_indent_rev
indentationData_concat = np.concatenate(indentationData_concat).ravel()

loadingData_concat = loading_load, unloading_load[::-1]
loadingData_concat = np.concatenate(loadingData_concat).ravel()
loadingData_concat = np.float128(loadingData_concat)

# plt.plot(indentationData_concat, loadingData_concat, label='concat curve')
# plt.legend()
# plt.show()

#-------Shared Fitting--------


join_point = loading_indent[-1]
dx2 = 0.02+join_point

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

plt.plot(indentationData_concat, loadingData_concat, label='concat curve')
initJoin = funcJoin.eval(parsJoin,x=indentationData_concat)
plt.plot(indentationData_concat,initJoin,'b--', label='int guess')
fitJoin = funcJoin.fit(loadingData_concat,parsJoin,x=indentationData_concat)
finalJoin = funcJoin.eval(fitJoin.params,x=indentationData_concat)
plt.plot(indentationData_concat,finalJoin,'r-', label='finalfit')
plt.legend()
plt.show()


