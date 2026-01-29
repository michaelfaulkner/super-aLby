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

def main(metropolis_data_folder, ecmc_data_folder):

    timestep_arr = np.zeros(len(os.listdir(metropolis_data_folder)))
    x2_analytical_arr = np.zeros(len(os.listdir(metropolis_data_folder)))
    x2_numerical_arr = np.zeros(len(os.listdir(metropolis_data_folder)))

    ecmc_timestep_arr = np.zeros(len(os.listdir(ecmc_data_folder)))
    ecmc_x2_analytical_arr = np.zeros(len(os.listdir(ecmc_data_folder)))
    ecmc_x2_numerical_arr = np.zeros(len(os.listdir(ecmc_data_folder)))

    for i, folder in enumerate(os.listdir(metropolis_data_folder)):
        if folder != "mean_squared_positions" and folder != "positions":
            
            temperature = 1
            timestep = float(folder[0] + "." + folder[1:])
            timestep_arr[i] = timestep
            number_of_particles = 120 / timestep
            number_of_equilibration_iterations = 1001
            thinning_level = None

            try:
                sample_directory = os.path.join(metropolis_data_folder, folder, "30k")
                mean_squared_position_sample = np.load(os.path.join(sample_directory, "temperature_00_sample_of_mean_positions.npy"))
                mean_squared_position_sample = mean_squared_position_sample[number_of_equilibration_iterations:]
            except:
                try:
                    sample_directory = os.path.join(metropolis_data_folder, folder)
                    mean_squared_position_sample = np.load(os.path.join(sample_directory, "temperature_00_sample_of_mean_positions.npy"))
                except:
                    pass

            mean_squared_position_sample = mean_squared_position_sample[:30000]
            x2_numerical_arr[i] = np.mean(mean_squared_position_sample)

    for i, folder in enumerate(os.listdir(ecmc_data_folder)):
        if folder != "mean_squared_positions" and folder != "positions":
            
            temperature = 1
            timestep = float(folder[0] + "." + folder[1:])
            ecmc_timestep_arr [i] = timestep
            number_of_particles = 120 / timestep
            number_of_equilibration_iterations = 1001
            thinning_level = None

            try:
                sample_directory = os.path.join(ecmc_data_folder, folder, "30k")
                mean_squared_position_sample = np.load(os.path.join(sample_directory, "temperature_00_sample_of_mean_positions.npy"))
                mean_squared_position_sample = mean_squared_position_sample[number_of_equilibration_iterations:]
            except:
                try:
                    sample_directory = os.path.join(ecmc_data_folder, folder)
                    mean_squared_position_sample = np.load(os.path.join(sample_directory, "temperature_00_sample_of_mean_positions.npy"))
                except:
                    pass

            mean_squared_position_sample = mean_squared_position_sample[:30000]
            ecmc_x2_numerical_arr[i] = np.mean(mean_squared_position_sample)
    

    for index, timestep_value in enumerate(timestep_arr):
        x2_analytical_arr[index] = analytical_x2(1.0, timestep_value)
    
    for index, timestep_value in enumerate(ecmc_timestep_arr):
        ecmc_x2_analytical_arr[index] = analytical_x2(1.0, timestep_value)
    
    sorted_timestep = timestep_arr[np.argsort(timestep_arr)]
    argsorted_data = np.argsort(timestep_arr)
    x2_numerical_arr = x2_numerical_arr[argsorted_data]
    x2_analytical_arr = x2_analytical_arr[argsorted_data]

    ecmc_sorted_timestep = ecmc_timestep_arr[np.argsort(ecmc_timestep_arr)]
    ecmc_argsorted_data = np.argsort(ecmc_timestep_arr)
    ecmc_x2_numerical_arr =ecmc_x2_numerical_arr[ecmc_argsorted_data]
    ecmc_x2_analytical_arr = ecmc_x2_analytical_arr[ecmc_argsorted_data]
   
    fig, ax = plt.subplots(2,1, sharey = True, sharex= True)
    ax[0].scatter(x2_analytical_arr, x2_numerical_arr/sorted_timestep**2, marker = "x", color="#e16f04ff", label = r"Metropolis MC with $3\times 10^4$ samples")
    ax[0].legend(loc = 2)
    ax[1].scatter(ecmc_x2_analytical_arr, ecmc_x2_numerical_arr/ecmc_sorted_timestep**2, marker = "x", color="#e20acdff", label = r"ECMC with $3\times 10^4$ samples")

    coeffs = np.polyfit(np.log(x2_analytical_arr[1:]), np.log(x2_numerical_arr[1:]/sorted_timestep[1:]**2), deg=1)
    print(coeffs)
    fitted_line = np.multiply(coeffs[0], np.log(x2_analytical_arr)) + coeffs[1]
    #ax.plot(x2_analytical_arr, np.exp(fitted_line), color="#f9a37bff", alpha = 0.7)

    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[1].set_xlabel(r"Analytical $\langle x^2 \rangle$", fontsize=15)
    fig.supylabel(r"Numerical $\langle x^2 \rangle$", fontsize=15)
    #ax.set_ylabel(r"Numerical $\langle x^2 \rangle$", fontsize=15)
    ax[0] = plt.gca()
    ax[0].set_xlim((0.3, 200))
    ax[0].set_ylim((0.02, 10000))

    
    ax[1].legend(loc = 2)


    plt.tight_layout()
    plt.savefig("pmdg_analytical_numerical_ecmc.png")

    



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])