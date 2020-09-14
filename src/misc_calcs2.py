from math import *
import numpy as np
import matplotlib.pyplot as plt

# import verbose flow
import HMC as h
from k_energies import *

#x_test = np.array([0.0])
#U_test = PE_family(4.0)
#K_test = PE_family(1.5)

#ESS_values = []

#for e in stepsizes:
#    K_test = PE_family(p)
#    samples = h.HMC(nits = 120000, x_input = x_test, U=U_test, K=K_test, L = 10, e = 0.2, \
#    BurnIn = 20000, Random = False, MH = True)
#    x_store = samples['x']
#    ESS_values.append(d.ess(x_store[0,20001:]))
#    print p

#print ESS_values

nits = 10
x_store = np.array([[2,1/2.0,3],[0,0,-1],[0,0,1],[0,0,0]])
a = abs(x_store)
maxes = a.max(axis=0)
tup_ind = np.where(maxes<=1)
ndarr_ind = np.array(tup_ind)
arr_ind = ndarr_ind.flatten()
if arr_ind.size == 0:
    time = nits
else:
    time = arr_ind[0] + 1

print time
