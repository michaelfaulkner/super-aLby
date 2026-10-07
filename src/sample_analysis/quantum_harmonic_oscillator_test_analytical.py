import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import sample_getter
import sys
from markov_chain_diagnostics import get_sample_mean_and_error
import matplotlib

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')
def analytical_x2(mass, omega, N_tau, timestep):
    # dim_omega = dim_m
    # auxiliary = 1 + dim_omega ** 2 / 2 - dim_omega * np.sqrt(1 + dim_omega**2 / 4)
    # return (1 / (2 * dim_m * dim_omega * np.sqrt(1 + 0.25 * dim_omega ** 2))) * (
    #         (1 + auxiliary ** N_tau) / (1 - auxiliary ** N_tau))

    auxilliary = 1.0 + 0.5 * timestep**2 * omega**2 - timestep * \
        omega * np.sqrt(1 + 0.25 * timestep**2 * omega**2)

    return (1.0 / (2.0 * mass * omega * np.sqrt(1.0 + 0.25 * timestep**2 * omega**2))) * ((1 + auxilliary**N_tau) /
                                                                                          (1 - auxilliary**N_tau))


def main(x2_data_path, x2_data_path_metrop, N, propertime, mass, omega):
    r"""
    Produces plots comparing the numerical and analytical values of <x^2> for the 1D quantum harmonic oscillator
        potential.
    """

    N = int(N)
    propertime = float(propertime)
    mass = float(mass)
    omega = float(omega)

    timestep_data = np.load(os.path.join(x2_data_path, "x2_ecmc_0.npy"))[:, 1]
    storage_arr = np.zeros((len(timestep_data), N))
    sorted_timestep = timestep_data[np.argsort(timestep_data)]

    for index in range(N):
        x2_timestep = np.load(os.path.join(
            x2_data_path, f"x2_ecmc_{index}.npy"))
        x2_data = x2_timestep[:, 0]
        timestep_data = x2_timestep[:, 1]
        argsorted_data = np.argsort(timestep_data)
        timestep_argsorted = timestep_data[argsorted_data]
        x2_data = x2_data[argsorted_data]
        storage_arr[:, index] = x2_data

    x2_mean_arr = np.mean(storage_arr, axis=1)

    x2_mean_arr = x2_mean_arr[sorted_timestep >= 0.01]
    err = np.std(storage_arr, axis=1)
    err = err[sorted_timestep >= 0.01]
    sorted_timestep = sorted_timestep[sorted_timestep >= 0.01]
    sorted_N = propertime / sorted_timestep
    analytical_data = np.zeros(len(x2_mean_arr))

    for t_index, timestep in enumerate(sorted_timestep):
        print(timestep)
        analytical_x2_val = analytical_x2(
            mass, omega, propertime/timestep, timestep)
        analytical_data[t_index] = analytical_x2_val

    fig, ax = plt.subplots(1,1)

    ax.errorbar(analytical_data, x2_mean_arr, err, color = "#5a0251ff", label = "ECMC", fmt='o', capsize=3, markersize=7)
    ax.set_xlabel(r"$\langle x^2 \rangle$", y = 0.05, fontsize = 30, weight = "bold")
    ax.set_ylabel(r"$\widebar{X^2}$", fontsize = 30, weight = "bold")


    timestep_data = np.load(os.path.join(x2_data_path_metrop, "x2_ecmc_0.npy"))[:, 1]
    storage_arr = np.zeros((len(timestep_data), N))
    sorted_timestep = timestep_data[np.argsort(timestep_data)]

    for index in range(N):
        x2_timestep = np.load(os.path.join(
            x2_data_path_metrop, f"x2_ecmc_{index}.npy"))
        x2_data = x2_timestep[:, 0]
        timestep_data = x2_timestep[:, 1]
        argsorted_data = np.argsort(timestep_data)
        timestep_argsorted = timestep_data[argsorted_data]
        x2_data = x2_data[argsorted_data]
        storage_arr[:, index] = x2_data

    x2_mean_arr_metrop = np.mean(storage_arr, axis=1)

    x2_mean_arr_metrop = x2_mean_arr_metrop[sorted_timestep >= 0.01]
    err = np.std(storage_arr, axis=1)
    err = err[sorted_timestep >= 0.01]
    sorted_timestep = sorted_timestep[sorted_timestep >= 0.01]
    sorted_N = propertime / sorted_timestep
    analytical_data = np.zeros(len(x2_mean_arr_metrop))

    for t_index, timestep in enumerate(sorted_timestep):
        print(timestep)
        analytical_x2_val = analytical_x2(
            mass, omega, propertime/timestep, timestep)
        analytical_data[t_index] = analytical_x2_val    

    ax.errorbar(analytical_data, x2_mean_arr_metrop, err, color = "#f6b67aff", label = "Metropolis MC", fmt='^', capsize=3, markersize=4)

    ax.tick_params(direction="in", left="off",labelleft="off", axis='both', which='both', labelsize=15)
    legend_properties = {'weight':'bold'}
    legend = ax.legend(loc='lower right', fontsize=10, prop=legend_properties)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.0)

    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

    #ax.set_xscale("log")
    #ax.set_yscale("log")
    #ax.set_ylim(2e-1, 6e-1)
    #ax.set_ylim(2e-1, 6e-1)
    plt.tight_layout()
    plt.savefig("qho_x2_analytical_FSEM_poster.pdf",)


    # print(f"analytical 0.01: {analytical_x2_arr[-1]}, ecmc: {numerical_x2_e[-1]}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
