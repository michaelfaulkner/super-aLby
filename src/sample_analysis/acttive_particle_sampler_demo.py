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

def main(config_file):

    config_file_string = config_file
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directories, temperatures,
        number_of_equilibration_iterations, number_of_observations, number_of_particles,
        _, _, _) = helper_methods.get_basic_config_data(config_file_string)
    
    sample_directory = sample_directories[0]
    temperature_index = 0
    thinning_level = None

    active_particle_sample = sample_getter.get_active_particle(sample_directory, temperatures[temperature_index],
                    temperature_index, 0, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
    
    plt.plot(np.arange(len(active_particle_sample[:1000])), active_particle_sample[:1000])
    plt.savefig("test.png")

if __name__ == '__main__':
    main(sys.argv[1])