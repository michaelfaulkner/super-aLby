import numpy as np
import os
import importlib
import matplotlib
import matplotlib.pyplot as plt
import sample_getter
import sys
from configparser import NoOptionError
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
    
    dimensionless_mass = parsing.get_value(config, strings.to_camel_case(potential), "dimensionless_mass")
    timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
    number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")

    sample_directory = sample_directories[0]
    sample_directory = "output/metropolis_01_50"
    temperature_index = 0

    thinning_level = 100

    mean_sample = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                    temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
    print(len(mean_sample))
    print(number_of_observations)
    # if len(mean_sample) != int(number_of_observations/thinning_level):
    #     mean_sample = mean_sample[int(number_of_equilibration_iterations/thinning_level):]
    
    print(len(np.arange(0,number_of_observations,thinning_level)))
    print(len(np.arange(0,10,3)))
    fig, ax = plt.subplots(1,1)
    ax.scatter(np.arange(0,number_of_observations,thinning_level), mean_sample * timestep**2)
    ax.set_title(f"expectation of {dimensionless_mass}")
    ax.set_xlabel("metropolis sweeps")
    ax.set_ylabel("<x^2>")
    fig.suptitle(f"dim_m = {dimensionless_mass}")
    plt.savefig(f"output/figs/trace_plots/x2_sweeps.png")


if __name__ == '__main__':
    main(sys.argv[1])
