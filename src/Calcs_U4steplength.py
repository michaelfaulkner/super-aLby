from math import *
import numpy as np
import matplotlib.pyplot as plt

# import verbose flow
import HMC as h
from k_energies import *
import MC_diag as d

p = 4.0  # potential power
k = 4.0/3  # kinetic power
d = 1000 # dimension

U = PE_family(p)
K = PE_family(k)
K2 = PE_family(2)

LeapFrogSteps = 10
e = 0.11
e2 = 0.23

nits = 1000
x_store = np.empty(nits) # only store first element
p_store = np.empty(nits) # only store first element
x_store2 = np.empty(nits) # only store first element
p_store2 = np.empty(nits) # only store first element

a_rate = np.empty(nits)
a_rate2 = np.empty(nits)

for i in xrange(nits):
    # sample
    x = U.sample(d)
    p = K.sample(d)
    p2 = K2.sample(d)

    # flow
    cand = h.flow(x, p, U.grad, K.grad, LeapFrogSteps, e) # run dynamics
    x_can = cand[0,:]
    p_can = cand[1,:]
    delta_H = U.value(x) + K.value(p) - U.value(x_can) - K.value(p_can)
    a_rate[i] = exp( min(0,delta_H) )

    cand2 = h.flow(x,p2,U.grad, K2.grad, LeapFrogSteps, e2)
    x_can2 = cand2[0,:]
    p_can2 = cand2[1,:]
    delta_H2 = U.value(x) + K2.value(p2) - U.value(x_can2) - K2.value(p_can2)
    a_rate2[i] = exp( min(0,delta_H2) )

    # record
    x_store[i] = abs(x_can[0] - x[0])
    p_store[i] = abs(p_can[0] - p[0])

    x_store2[i] = abs(x_can2[0] - x[0])
    p_store2[i] = abs(p_can2[0] - p2[0])

plt.subplot(321)
plt.ylim((0,1))
plt.hist(x_store, normed=True, range=(0,3))
plt.title('PE43')

plt.subplot(322)
plt.ylim((0,1))
plt.hist(x_store2, normed=True, range=(0,3))
plt.title('Gauss')

plt.subplot(323)
plt.ylim((0,1))
plt.hist(p_store, normed=True, range=(0,3))
plt.title('PE43 kinetic')

plt.subplot(324)
plt.ylim((0,1))
plt.hist(p_store2, normed=True, range=(0,3))
plt.title('Gauss kinetic')

plt.subplot(325)
plt.ylim((0,4))
plt.hist(a_rate, normed=True, range=(0,1))
plt.title('Acceptance rate PE43')

plt.subplot(326)
plt.ylim((0,4))
plt.hist(a_rate2, normed=True, range=(0,1))
plt.title('Acceptance rate Gauss')

plt.show()

p_test = K.sample(nits)
sign = p_test / abs(p_test)
p_third = sign * (p_test**(1.0/3))

plt.subplot(211)
plt.hist(p_test, bins = 30, normed = True, range = ((-3,3)))

plt.subplot(212)
plt.hist(p_third, bins = 30, normed = True, range = ((-3,3)))

plt.show()
