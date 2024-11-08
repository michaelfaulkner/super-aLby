import numpy as np
import matplotlib.pyplot as plt

dim_m = 0.1
N_tau = [2000, 1200, 1000, 500, 400, 200, 100, 50]

def analytical_x2(dim_m, N_tau):
    dim_omega = dim_m
  
    auxilliary = 1 + dim_omega**2 / 2 - dim_omega * np.sqrt(1 + dim_omega**2 / 4)

    return (1 / (2 * dim_m * dim_omega * np.sqrt(1 + 0.25 * dim_omega**2))) * ((1 + auxilliary**N_tau) / (1 - auxilliary**N_tau))


exp = analytical_x2(dim_m, N_tau)

plt.scatter(N_tau, exp)
plt.xlabel("N")
plt.ylabel("<x^2>")
plt.show()