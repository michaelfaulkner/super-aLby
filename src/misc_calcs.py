
from math import *
import numpy as np
import matplotlib.pyplot as plt


# import verbose flow
import HMC as h
from k_energies import *
import MC_diag as d


U = GR_family(beta = 4.0)
x_test = np.array([1.0,-2.0])

print U.value(x_test)
print U.grad(x_test)
draws = U.sample(1000)

plt.hist(draws)
plt.show()
#x_test = np.array([0.0])
#x_test = x_test.repeat(1000)

#samples = h.HMC(nits = 10000, x_input = x_test, U=U_test, K=K_test, L = 10, e = 0.11, \
#BurnIn = 0, Random = False, MH = True)
#x_store = samples['x']
#ESS = d.ess(x_store[0])

#print ESS
#print float(ESS[0])

#plt.plot(x_store[0])
#plt.show()

## plotting the results:
#plt.subplot(411)
#plt.plot(x_store[0,:])
#plt.ylabel('Position')

#plt.subplot(412)
#plt.plot(p_store[0,:], color='r')
#plt.ylabel('Momentum')

#plt.subplot(413)
#plt.hist(x_store[0,:], bins=30,normed=1, range=(-3,3))

#plt.subplot(414)
#plt.hist(p_store[0,:], bins=30,normed=1, range=(-3,3), color='r')

#plt.show()


## Doing diagnostics
#print d.accept_rate(x_store[0,:])
#print d.acf(x_store[0,2000:])
#print d.iat(x_store[0,:])
#print d.ess(x_store[0,:])
#print d.ess(x_store[0,:], trunc = False)
#plt.plot(x_store[0,:])
#plt.show()

#from statsmodels.graphics.tsaplots import plot_acf
#plot_acf(x_store[0,:])
#plt.title('Autocorrelations')
#plt.show()
