# Centre of space Calcs

from math import *
import numpy as np
import matplotlib.pyplot as plt
from k_energies import *

# Test to see typical size of jumps in centre of space

# quartic potential means near 0 gradient will be negligible.
# Hence jumps will be dictated by p_0^(a-1), where a is KE power



powers = [1, 1.2,1.4,1.6,1.8,2,2.2,2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6]
n = len(powers)
jumps = []
for p in powers:
    K = PE_family(p)
    values = K.sample(10000)
    temp = abs(values)**(p-1)
    jumps.append(np.mean(temp))
    print np.mean(temp)

plt.plot(powers,jumps, 'b.')
plt.show()
