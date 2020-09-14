from math import *
import numpy as np
import matplotlib.pyplot as plt

# import verbose flow
import HMC as h
from k_energies import *
import MC_diag as d

p = 4.0
k = 5/4.0

U_test = PE_family(p)
K_test = GR_family(beta = k)
#K_test = PE_family(k)
#K_test = Gauss()
#K_test = Laplace()

# Evaluate point at which all points have magnitude less than 1

#x_test = U_test.sample(100)
x_test = np.random.uniform(low = -5, high = 5, size = 100)

nits = 10000

print "Kinetic power: %.3f" % k
print "Potential power: %.3f" % p
print "Iterations: %d" % nits
print " "
print "------------------------------------"
print "------------------------------------"
print " "

e = 0.24
delta_step = 0.02
for i in xrange(10):
    samples = h.HMC(nits, x_input = x_test, U=U_test, K=K_test, L = 10, e = e, \
    BurnIn = 0, Random = False, MH = True)
    x_store = samples['x']

    # calculate column maxes of absolute values
    a = abs(x_store)
    maxes = a.max(axis=0)
    tup_ind = np.where(maxes<=2)
    ndarr_ind = np.array(tup_ind)
    arr_ind = ndarr_ind.flatten()
    if arr_ind.size == 0:
        time = nits
    else:
        time = arr_ind[0] + 1

    #ESS = d.ess(x_store[0])
    print "Iterations to centre: %.3f" % time
    print " "
    print "------------------------------------"
    print " "
    e += delta_step
