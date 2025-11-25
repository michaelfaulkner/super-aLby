import numpy as np
import os
import importlib
import sample_getter
import sys
import time
from configparser import NoOptionError
import matplotlib.pyplot as plt

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")

def main(config_file_string):

    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, temperature,
    number_of_equilibration_iterations, number_of_observations, number_of_particles,
    _) = helper_methods.get_basic_config_data(config_file_string)
    
    #mass = parsing.get_value(config, strings.to_camel_case(potential), "mass")
    #timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
    number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")
    temperature_index = 0
    thinning_level = None

    checkpointing_indices = sample_getter.get_checkpointing_indices(sample_directory)
    if checkpointing_indices != 0:
        sub_arr_len = number_of_equilibration_iterations + number_of_observations
        mean_sample = np.zeros(sub_arr_len * (checkpointing_indices + 1))
        for i in range(checkpointing_indices + 1):
            mean_sample[i * sub_arr_len : (i+1) * sub_arr_len + 1] = sample_getter.get_mean_positions(
                sample_directory, temperature, i, number_of_particles,
                None, thinning_level=thinning_level)[:, 0]
            
    else:
        mean_sample = sample_getter.get_mean_positions(sample_directory, temperature, number_of_particles,
                                                       number_of_equilibration_iterations,
                                                       thinning_level=thinning_level)
        
    print(np.shape(mean_sample))

    # plt.scatter(np.arange(sub_arr_len * (checkpointing_indices + 1) + 1), mean_sample)
    # plt.savefig("test.png")

if __name__ == '__main__':
    main(sys.argv[1])