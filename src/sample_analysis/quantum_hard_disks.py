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
    number_of_quantum_particles = parsing.get_value(config, "ModelSettings", "number_of_quantum_particles")
    number_of_timeslices = parsing.get_value(config, "ModelSettings", "number_of_timeslices")
    number_of_particles = number_of_quantum_particles * number_of_timeslices

    sample_directory = sample_directories[0]
    temperature_index = 0
    thinning_level = None
    number_of_equilibration_iterations = None
    position_sample = sample_getter.get_positions(sample_directory, temperatures[temperature_index],
                    temperature_index, 0, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)

   
    # plt.scatter(np.arange(np.shape(position_sample)[1]), position_sample[0, :], label="0")
    # plt.scatter(np.arange(np.shape(position_sample)[1]), position_sample[1000, :], label="1000")
    # plt.scatter(np.arange(np.shape(position_sample)[1]), position_sample[5000, :], label="5000")
    # plt.scatter(np.arange(np.shape(position_sample)[1]), position_sample[10000, :], label="10000")
    # plt.legend()
    # particle_0_sample = position_sample[:,::2]
    
    # particle_0_mean = np.mean(particle_0_sample**2, axis=1)
    # print(np.shape(particle_0_mean))
    
    # plt.plot(np.arange(len(particle_0_mean)), particle_0_mean)

    worldline_0 = position_sample[:, 0:2]
    print(np.shape(worldline_0))
    plt.figure(figsize=[10,10])
    ax = plt.axes(ylim=(-5, 5))
    start = 1000
    num_of_samples = 1150
    print(np.shape(np.arange(0,num_of_samples)), np.shape(worldline_0[0:num_of_samples, :]))
    ax.scatter(np.arange(start,num_of_samples), worldline_0[start:num_of_samples, 0], s=100)
    ax.scatter(np.arange(start,num_of_samples), worldline_0[start:num_of_samples, 1], s=100)
    ax.errorbar(np.arange(start,num_of_samples), worldline_0[start:num_of_samples, 1], yerr=np.ones(np.shape(np.arange(start,num_of_samples))), fmt=".", capsize=5.0)
    ax.errorbar(np.arange(start,num_of_samples), worldline_0[start:num_of_samples, 0], yerr=np.ones(np.shape(np.arange(start,num_of_samples))), fmt=".", capsize=5.0)

    ax.set_yticks(np.arange(-5,6))
    plt.tight_layout()
    plt.savefig("test.png")
    with np.printoptions(threshold=np.inf):
        print(worldline_0[0:50,:])


   
if __name__ == '__main__':
    main(sys.argv[1])