from math import *
import numpy as np
import matplotlib.pyplot as plt

# import verbose flow
import HMC as h
from k_energies import *
import MC_diag as d

p = 4.0
k = 4.0/3

U_test = PE_family(p)
K_test = GR_family(beta = k)
#K_test = PE_family(k)
#K_test = Gauss()
#K_test = Laplace()

# Evaluate point at which all points have magnitude less than 1

x_test = U_test.sample(100)
#x_test = np.random.uniform(low = -5, high = 5, size = 100)

nits = 10000

print "Kinetic power: %.3f" % k
print "Potential power: %.3f" % p
print "Iterations: %d" % nits
print " "
print "------------------------------------"
print "------------------------------------"
print " "

e = 0.28
delta_step = 0.02
for i in xrange(10):
    samples = h.HMC(nits, x_input = x_test, U=U_test, K=K_test, L = 10, e = e, \
    BurnIn = 0, Random = False, MH = True)
    x_store = samples['x']
    ESS = d.ess(x_store[0])
    print "Effective sample size: %.3f" % float(ESS[0])
    print " "
    print "------------------------------------"
    print " "
    e += delta_step
