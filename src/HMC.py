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
    return np.dot(p,p) / 2.0

# kinetic gradient
def dK(p):
    return p

# Hamiltonian Dynamics
def flow(x, p, dU, dK, L, e):
    p = p - e * dU(x) / 2.0
    for i in xrange(1,L-1):
        x = x + e * dK(p)
        p = p - e * dU(x)
    x = x + e * dK(p)
    p = p - e * dU(x) / 2.0
    return np.vstack((x,p))

# Hamiltonian dynamics with intermediate times stored
def flow_verb(x_input, p_input, dU, dK, L, e):
    if type(x_input) is float or type(x_input) is int:
        x_input = [x_input]
    n = len(x_input)
    x = np.empty((n,L))
    p = np.empty((n,L))
    x[:,0] = x_input
    p[:,0] = p_input
    for i in xrange(0,L-1):
        p_int = p[:,i] - e * dU(x[:,i]) / 2.0
        x[:,i+1] = x[:,i] + e * dK(p_int)
        p[:,i+1] = p_int - e * dU(x[:,i+1]) / 2.0
    return np.vstack((x,p))

# HMC as a function INSERT
def HMC(nits, x_input, U, K, L, e, BurnIn = 0, Random = False, MH = True):

    # preamble
    #if type(x_input) is float or type(x_input) is int:
    #    n = 1
    #else:
    #    n = len(x_input)

    x = x_input # initial value for x
    n = len(np.atleast_1d(x))
    accepted = 0.
    divergences = 0
    LeapFrogSteps = L # initialise numer of LeapFrogSteps
    x_store = np.empty((n,nits))
    p_store = np.empty((n,nits))


    for i in xrange(nits): # don't forget python arrays start at zero
        # propose
        p = K.sample(n) # resample momentum from (e^(-K(p))
        #if Random == True: # randomise LeapFrogSteps
        #    LeapFrogSteps = np.random.randint(1,L+1)
        cand = flow(x, p, U.grad, K.grad, LeapFrogSteps, e) # run dynamics
        x_can = cand[0,:]
        p_can = cand[1,:]

        # accept-reject if MH = True (otherwise accept all points)
        if MH == True:
            delta_H = U.value(x) + K.value(p) - U.value(x_can) - K.value(p_can)
            if delta_H < -1000:
                divergences += 1 # count numerical divergences
                # update chain
            u = np.random.uniform(0,1)
            if log(u) < delta_H:
                x = x_can
                p = p_can
                accepted += 1
        else:
            x = x_can
            p = p_can

        x_store[:,i] = x
        p_store[:,i] = p

        # Adapt step-size if BurnIn is nonzero
        if i <= BurnIn - 1 and (i+1) % 100 == 0:
            accept_rate = accepted / 100.0
            if accept_rate > 0.8:
                e = e * 1.1
            elif accept_rate < 0.6:
                e = e * 0.9
            accepted = 0

    # print acceptance rate
    accept_rate = accepted/float(nits - BurnIn)
    print "Acceptance rate: %f" % accept_rate
    print "LF Steps: %d, Step-size: %.3f" % (L,e)
    print "Numerical divergences: %d" % divergences

    # bind initial values with MCMC output
    x_return = np.empty((n, nits + 1))
    p_return = np.empty((n, nits + 1))
    x_return[:,0] = x_input
    x_return[:,1:] = x_store
    p_return[:,0] = np.zeros(n)
    p_return[:,1:] = p_store

    return { 'x':x_return, 'p':p_return }
