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

def main(config_file_string, config_file_string_propertime, cutoff):
    cutoff = int(cutoff)

    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, temperature,
    number_of_equilibration_iterations, number_of_observations, number_of_particles,
    _) = helper_methods.get_basic_config_data(config_file_string)

    mass = parsing.get_value(config, "QuantumHarmonicOscillatorPotential", "mass")
    omega_squared = parsing.get_value(config, "QuantumHarmonicOscillatorPotential", "omega_squared")


    number_of_equilibration_iterations = 0
    thinning_level = None
    # position_sample = sample_getter.get_positions(sample_directory, temperature, 0, number_of_particles,
    #                                                    number_of_equilibration_iterations,
    #                                                    thinning_level=thinning_level)

    mean_squared_positon_sample = sample_getter.get_mean_squared_positions(sample_directory, temperature, 0,
                                                                           number_of_particles,
                                                                           number_of_equilibration_iterations,
                                                                           thinning_level=thinning_level)
    print(np.shape(mean_squared_positon_sample))

    plt.plot(np.arange(len(mean_squared_positon_sample[:cutoff])), mean_squared_positon_sample[:cutoff], label="normal")

    config = parsing.read_config(parsing.parse_options([config_file_string_propertime]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, temperature,
        number_of_equilibration_iterations, number_of_observations, number_of_particles,
        _) = helper_methods.get_basic_config_data(config_file_string_propertime)
    number_of_equilibration_iterations = 0
    mean_squared_positon_sample = sample_getter.get_mean_squared_positions(sample_directory, temperature, 0,
                                                                               number_of_particles,
                                                                               number_of_equilibration_iterations,
                                                                               thinning_level=thinning_level)

    position_sample = sample_getter.get_positions(sample_directory, temperature, 0,
                                                                               number_of_particles,
                                                                               number_of_equilibration_iterations,
                                                                               thinning_level=thinning_level)
    plt.plot(np.arange(len(mean_squared_positon_sample[:cutoff])), mean_squared_positon_sample[:cutoff], label="propertime")
    plt.legend()
    plt.savefig("trace.png")

    plt.clf()
    obs = np.shape(position_sample)[0]
    plt.plot(np.arange(obs)[:cutoff], position_sample[:cutoff, -1])

    plt.savefig("dt.png")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])