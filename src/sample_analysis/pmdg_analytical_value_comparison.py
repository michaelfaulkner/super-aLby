import numpy as np
import os
import importlib
import matplotlib
import matplotlib.pyplot as plt
import os
import sample_getter
import sys

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")
matplotlib.rcParams['mathtext.fontset'] = 'cm'

def analytical_x2(dim_m, N_tau):
    dim_omega = dim_m
    auxiliary = 1 + dim_omega ** 2 / 2 - dim_omega * np.sqrt(1 + dim_omega**2 / 4)
    return (1 / (2 * dim_m * dim_omega * np.sqrt(1 + 0.25 * dim_omega ** 2))) * (
            (1 + auxiliary ** N_tau) / (1 - auxiliary ** N_tau))

def main(data_folder):

    timestep_arr = np.zeros(len(os.listdir(data_folder)))
    x2_analytical_arr = np.zeros(len(os.listdir(data_folder)))
    x2_numerical_arr = np.zeros(len(os.listdir(data_folder)))

    for i, folder in enumerate(os.listdir(data_folder)):
        if folder != "mean_squared_positions" and folder != "positions":
            
            temperature = 1
            print(folder)
            timestep = float(folder[0] + "." + folder[1:])
            timestep_arr[i] = timestep
            number_of_particles = 120 / timestep
            number_of_equilibration_iterations = 1001
            thinning_level = None

            try:
                sample_directory = os.path.join(data_folder, folder, "30k")
                mean_squared_position_sample = np.load(os.path.join(sample_directory, "temperature_00_sample_of_mean_positions.npy"))
                mean_squared_position_sample = mean_squared_position_sample[number_of_equilibration_iterations:]
            except:
                try:
                    sample_directory = os.path.join(data_folder, folder)
                    mean_squared_position_sample = np.load(os.path.join(sample_directory, "temperature_00_sample_of_mean_positions.npy"))
                except:
                    pass

            mean_squared_position_sample = mean_squared_position_sample[:30000]
            x2_numerical_arr[i] = np.mean(mean_squared_position_sample)
            print(timestep_arr)

    for index, timestep_value in enumerate(timestep_arr):
        x2_analytical_arr[index] = analytical_x2(1.0, timestep_value)
    plt.scatter(x2_analytical_arr, x2_numerical_arr/timestep_arr**2, marker = "x", color="orange", label = r"Metropolis MC with $3\times 10^4$ samples")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Analytical $\langle x^2 \rangle$", fontsize=15)
    plt.ylabel(r"Numerical $\langle x^2 \rangle$", fontsize=15)
    plt.legend()


    plt.tight_layout()
    plt.savefig("pmdg_analytical_numerical_metropolis.pdf")

    



if __name__ == '__main__':
    main(sys.argv[1])