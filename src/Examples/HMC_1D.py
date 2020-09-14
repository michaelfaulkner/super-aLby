#!/usr/bin/env python

from math import *
import numpy as np
from matplotlib.pylab import *

# potential
def U(x):
    return x*x/2.

# potential gradient
def dU(x):
    return x

# kinetic
def K(p):
    return p*p/2.

# kinetic gradient
def dK(p):
    return p

# Hamiltonian Dynamics
def flow(x,p,L,e):
    p = p - e*dU(x)/2
    for i in xrange(1,L-1):
        x = x + e*dK(p)
        p = p - e*dU(x)
    x = x + e*dK(p)
    p = p - e*dU(x)/2
    return [x,p]


nits = 10000
alpha = 1
L = 10
e = 0.1
x = 0.
p = 0.
x_store = []
p_store = []
x_store.append(x)
p_store.append(p)

#innov = np.random.uniform(-alpha,alpha,n) #random innovation, uniform proposal distribution
for i in xrange(1,nits):
    ## propose
    p = np.random.normal(0,1,1) # resample momentum from (e^(-K(p))
    [x_can,p_can] = flow(x,p,L,e) # run dynamics

    # accept-reject
    aprob = min([1.,exp( U(x)+K(p) - U(x_can)-K(p_can) )])

    # update chain
    u = np.random.uniform(0,1)
    if u < aprob:
        x = x_can
        p = p_can
    x_store.append(x)
    p_store.append(p)



#plotting the results:
#theoretical curve
# x = arange(-3,3,.1)
#y = sdnorm(x)
subplot(411)
# title('Metropolis-Hastings')
plot(x_store)
# ylabel('Position')
subplot(412)
plot(p_store)
# ylabel('Momentum')
# subplot(212)
# plt.show()
subplot(413)
hist(x_store, bins=30,normed=1)
subplot(414)
hist(p_store, bins=30,normed=1)
#plot(x,y,'ro')
#ylabel('Frequency')
#xlabel('x')
#legend(('PDF','Samples'))
show()
