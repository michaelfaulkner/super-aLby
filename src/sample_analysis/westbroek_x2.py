import numpy as np
import matplotlib.pyplot as plt
import sys

def analytical_x2(m, omega, N_tau, dt):
    dim_m = m * dt
    dim_omega = omega * dt
    auxiliary = 1 + dim_omega ** 2 / 2 - dim_omega * np.sqrt(1 + dim_omega**2 / 4)
    return (1 / (2 * dim_m * dim_omega * np.sqrt(1 + 0.25 * dim_omega ** 2))) * (
            (1 + auxiliary ** N_tau) / (1 - auxiliary ** N_tau))

def main(m, omega, T, min_dt, max_dt):
    m = float(m)
    omega = float(omega)
    T = float(T)
    min_dt = float(min_dt)
    max_dt = float(max_dt)

    dt_arr = np.linspace(min_dt, max_dt)
    print(dt_arr)
    x2_arr = np.zeros(len(dt_arr))
    print(len(dt_arr))

    for index, dt in enumerate(dt_arr):
        N_tau = T/dt
        x2_arr[index] = analytical_x2(m, omega, N_tau, dt)

    max_fitting = 10
    coeffs = np.polyfit(np.log(dt_arr[:max_fitting]), np.log(x2_arr[:max_fitting]), deg=1)
    fitted = coeffs[1] + np.multiply(np.log(dt_arr[:max_fitting]), coeffs[0])
    print(coeffs)
    plt.scatter(dt_arr, x2_arr)
    plt.plot(dt_arr[:max_fitting], np.exp(fitted), color="#d129b8ff")
    plt.xlabel("dt")
    plt.ylabel("<x^2>")
    plt.xscale("log")
    plt.yscale("log")

    plt.savefig("analytical_x2_dt.png")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
