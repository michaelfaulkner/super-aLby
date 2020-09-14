#!/usr/bin/env python

from math import *
import numpy as np
from matplotlib.pylab import *

# potential
def U(x):
    return np.dot(x,x)/ 2.0

# potential gradient
def dU(x):
    return x

# kinetic
def K(p):
    return np.dot(p,p)/ 2.0

# kinetic gradient
def dK(p):
    return p

# Hamiltonian Dynamics
def flow(x,p,L,e):
    p = p - e * dU(x) / 2.0
    for i in xrange(1,L-1):
        x = x + e * dK(p)
        p = p - e * dU(x)
    x = x + e * dK(p)
    p = p - e * dU(x) / 2.0
    return np.vstack((x,p))

nits = 10000
n = 2 # dimension of state space

L = 10 # leapfrog steps
e = 1.5 # step size
x = np.zeros(n) # initial value for x
accepted = 0.

# storage
x_store = np.empty((n,nits))
p_store = np.empty((n,nits))

for i in xrange(nits): # don't forget python arrays start at zero
    # propose
    p = np.random.normal(size = n) # resample momentum from (e^(-K(p))
    cand = flow(x,p,L,e) # run dynamics
    x_can = cand[0,:]
    p_can = cand[1,:]

    # accept-reject
    delta_H = U(x) + K(p) - U(x_can) - K(p_can)

    # update chain
    u = np.random.uniform(0,1)
    if log(u) < delta_H:
        x = x_can
        p = p_can
        accepted = accepted + 1
    x_store[:,i] = x
    p_store[:,i] = p

# print acceptance rate
accept_rate = accepted/nits
print "Acceptance rate: %f" % accept_rate
print "LF Steps: %d, Step-size: %.1f." % (L,e)



## plotting the results:
subplot(411)
#title('x trace')
plot(x_store[0,:])
ylabel('Position')

subplot(412)
plot(p_store[1,:], color='r')
ylabel('Momentum')
# subplot(212)
# plt.show()
subplot(413)

#theoretical curve
#x_th = arange(-3,3,.1)
#y_th = exp(-U(x))/sqrt(2*pi)
hist(x_store[0,:], bins=30,normed=1, range=(-3,3))
#plot(x_th,y_th,'ro')

subplot(414)
hist(p_store[1,:], bins=30,normed=1, range=(-3,3), color='r')


#ylabel('Frequency')
#xlabel('x')
#legend(('PDF','Samples'))
show()
