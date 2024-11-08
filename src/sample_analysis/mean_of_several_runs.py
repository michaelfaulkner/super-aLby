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


def main():

    # read in the k values from k_values.txt
    current_directory = os.path.dirname(__file__)
    k_values_filepath = os.path.join(os.path.split(current_directory)[0], "N_values.txt")
    k_data = np.loadtxt(k_values_filepath, dtype='str')
    k_values = k_data[:,1]

    analytical_x2_arr = np.zeros(len(k_values))
    numerical_x2 = np.zeros(len(k_values))
    N_arr = np.zeros(len(k_values))

    for index, string in enumerate(k_values):
        config_file_string = f"src/config_files/convergence_tests/one_dim_quantum_oscillator_potential/metropolis_01_{string}.ini"
        config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
        (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
        number_of_observations, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)
        
        dimensionless_mass = parsing.get_value(config, strings.to_camel_case(potential), "dimensionless_mass")
        number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")

        mass_as_string = str(dimensionless_mass)
        mass_string_split = mass_as_string.split(".")
        mass_as_string = ""
        for _ in mass_string_split:
            mass_as_string += _
        
        sample_directory = f"output/N/metropolis_01_{string}"
        temperature_index = 0
        if number_of_observations < 10000:
            thinning_level = 1
        else:
            thinning_level = 10
        mean_sample = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                            temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
        if len(mean_sample) != int(number_of_observations/thinning_level):
            mean_sample = mean_sample[int(number_of_equilibration_iterations/thinning_level):]

                
        mean_sample_mean = get_sample_mean_and_error(mean_sample)
        numerical_x2[index] = mean_sample_mean[0]
        N_arr[index] = number_of_particles
        #print(f"done for {number_of_particles}")

    np.save("output/N/x2_run2", numerical_x2)

if __name__ == '__main__':
    main()