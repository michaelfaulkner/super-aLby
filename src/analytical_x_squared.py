import numpy as np
import matplotlib.pyplot as plt

dim_m = np.arange(0.1,1.1,0.01)

def analytical_x2(dim_m):
    dim_omega = dim_m
    N_tau = 120 / dim_m 
    auxilliary = 1 + dim_omega**2 / 2 - dim_omega * np.sqrt(1 + dim_omega**2 / 4)

    return (1 / (2 * dim_m * dim_omega * np.sqrt(1 + 0.25 * dim_omega**2))) * ((1 + auxilliary**N_tau) / (1 - auxilliary**N_tau))


exp = analytical_x2(dim_m)

plt.scatter(dim_m, exp)
plt.xlabel("m")
plt.ylabel("<x^2>")
plt.show()