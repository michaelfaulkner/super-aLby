import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')

def anharmonic(l, w2, dt, m, x):

    return l * dt * x**4 + 0.5 * m * w2 * dt * x**2



x = np.arange(-10.0, 10.1, 0.1)

anharmonic_arr = anharmonic(1.0, -150.0, 1.0, 1.0, x)
plt.plot(x, anharmonic_arr, color="#e85e9a", linewidth = 3)
plt.xlabel(r"$x$", fontsize = 15, weight = "bold")
plt.ylabel(r"$V(x)$", fontsize = 15, weight = "bold", labelpad=-7)

plt.tight_layout()
plt.savefig("anhar.pdf")