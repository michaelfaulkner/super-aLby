import numpy as np
import os
import importlib
import matplotlib
import matplotlib.pyplot as plt
import sample_getter
import sys
from configparser import NoOptionError
from markov_chain_diagnostics import get_sample_mean_and_error


this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")

def main(config_file_string):
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
    number_of_observations, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)
    
    mass = parsing.get_value(config, strings.to_camel_case(potential), "mass")
    timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
    number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")
    
    sample_directory = sample_directories[0]
    temperature_index = 0
    thinning_level = None
    number_of_equilibration_iterations = None
    # mean_sample = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
    #                     temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)

    timestep_str = str(timestep).replace(".", "")
    # print(timestep_str)
    
    sub_arr_len = 11000
    num_sub_arrs = 50
    mean_sample = np.zeros(sub_arr_len * num_sub_arrs)
    for i in range(num_sub_arrs):
        mean_sample[i * sub_arr_len : (i+1) * sub_arr_len] = np.load(
        f"output/positions_001_hot_start/temperature_00_run_{i:02d}_sample_of_mean_positions.npy")[1:, 0]
    
    mean = np.mean(mean_sample)
    std = np.std(mean_sample)
    plt.plot(np.arange(np.shape(mean_sample)[0]), mean_sample)
    plt.xlabel("sample index")
    plt.ylabel("mean of x^2")
    plt.title(f"metropolis, hot start, positions data, timestep = {timestep}")
    #plt.ylim((-1, 0.75))
    #plt.xlim((-1, 0.1e6))

    plt.savefig(f"{timestep_str}_metropolis_w_positions.png")
    plt.xlim((-1000, 10000))
    plt.savefig(f"{timestep_str}_metropolis_w_positions_zoomed.png")
    
    
    # plt.savefig("zoomed_metropolis_10e6_mean_squared_positions.png")
    print(f"mean: {mean}, std: {std}")

if __name__ == '__main__':
    main(sys.argv[1])