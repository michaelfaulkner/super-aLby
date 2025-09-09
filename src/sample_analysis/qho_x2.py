import importlib
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sample_getter
import sys


this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")


def main(data_folder):

#     config_file_string = config_file
 
#     config = parsing.read_config(
#         parsing.parse_options([config_file_string]).config_file)

#     (config_file_mediator, potential,_, samplers, sample_directories, temperatures,
#         number_of_equilibration_iterations, number_of_observations, number_of_particles,
#         _, _, _) = helper_methods.get_basic_config_data(config_file_string)

#     mass = parsing.get_value(
#         config, strings.to_camel_case(potential), "mass")
#     timestep = parsing.get_value(
#         config, strings.to_camel_case(potential), "timestep")
#     sample_directory = sample_directories[0]
#     temperature_index = 0
#     thinning_level = None

    # mean_sample = sample_getter.get_mean_squared_positions(sample_directory, temperatures[temperature_index],
    #                         temperature_index, 0, number_of_particles, number_of_equilibration_iterations,
    #                         thinning_level=thinning_level)
    x2_arr = np.zeros(len(os.listdir(data_folder)))
    dt_arr = np.zeros(len(os.listdir(data_folder)))
    
    for index, folder in enumerate(os.listdir(data_folder)):
        mean_sample = np.load(os.path.join(data_folder, folder,"temperature_00_checkpoint_00_sample_of_mean_squared_positions.npy"))
        print(np.mean(mean_sample))
        dt_arr[index] = float(folder[0] + "." + folder[1:])
        x2_arr[index] = np.mean(mean_sample)#/dt_arr[index]**2

    # max_fitting = 10
    # coeffs = np.polyfit(np.log(dt_arr[:max_fitting]), np.log(x2_arr[:max_fitting]), deg=1)
    # fitted = coeffs[1] + np.multiply(np.log(dt_arr[:max_fitting]), coeffs[0])
    # print(coeffs)
    Tau = 20
    plt.scatter(Tau/dt_arr, x2_arr)
   # plt.plot(dt_arr[:max_fitting], np.exp(fitted), color="#d129b8ff")
    plt.xlabel("N")
    plt.ylabel("<x^2>")
    plt.xscale("log")
    plt.yscale("log")

    plt.savefig("numerical_x2_dt.png")

if __name__ == '__main__':
    main(sys.argv[1])