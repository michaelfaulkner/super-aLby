import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')



def func(dt, w2, m, l):
    return dt * m**2 * w2**2 * 3 / (16 * l)

w2_arr = np.array([-1.0, -2.5, -5.0, -7.5, -10.0, -15.0, -20.0, -25.0, -30.0, -40.0, -50.0, -60.0])

dt_arr = np.array([0.01, 0.025, 0.05, 0.075, 0.1, 0.5, 0.75, 1.0, 2.0, 3.0])

m = 1.0
l = 1.0

for index, dt in enumerate(dt_arr):
    barrier_height_arr = func(dt, w2_arr, m, l)

    plt.plot(w2_arr, barrier_height_arr, label = f"{dt}")

plt.ylabel("barrier height (anharmonic)")
plt.xlabel(r"$\omega ^2$")


plt.legend()
plt.savefig("barrier_height_w2.png")

plt.clf()

for index, w2 in enumerate(w2_arr):
    barrier_height_arr = func(dt_arr, w2, m, l)
    sorted_dt = dt_arr[np.argsort(dt_arr)]
    sorted_barrier_height = barrier_height_arr[np.argsort(dt_arr)]
    sorted_N = 50/sorted_dt
    plt.plot(sorted_N, sorted_barrier_height, label =f"{w2}")

plt.yscale("log")
plt.xscale("log")

plt.ylabel("barrier height (anharmonic)")
plt.xlabel(r"$N_{\tau}$")


plt.legend()
plt.savefig("barrier_height_Nt.png")