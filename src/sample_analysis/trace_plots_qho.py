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

def main(config_file_string, ecmc_config_file_string):

    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, temperature,
    number_of_equilibration_iterations, number_of_observations, number_of_particles,
    _) = helper_methods.get_basic_config_data(config_file_string)

    mass = parsing.get_value(config, "QuantumHarmonicOscillatorPotential", "mass")
    omega_squared = parsing.get_value(config, "QuantumHarmonicOscillatorPotential", "omega_squared")
    anharmonicity = parsing.get_value(config, "QuantumHarmonicOscillatorPotential", "anharmonicity")


    thinning_level = None
    position_sample = sample_getter.get_positions(sample_directory, temperature, 0, number_of_particles,
                                                       number_of_equilibration_iterations,
                                                       thinning_level=thinning_level)
    
    magnitude_of_double_well_position = np.sqrt(-mass * omega_squared /
                                                               (4 * anharmonicity))
    
    print(np.shape(position_sample))
    plt.plot(np.arange(len(position_sample[:, 6])), position_sample[:, 6], color="orange")
    plt.hlines([0.0, -magnitude_of_double_well_position, magnitude_of_double_well_position], 0.0, len(position_sample[:, 6]))
    plt.xlim((0.0, 1000))
    plt.savefig("trace_metropolis.png") 

    plt.clf()
    plt.plot(np.arange(len(position_sample[100, :])), position_sample[100, :], color="magenta")
    plt.plot(np.arange(len(position_sample[1000, :])), position_sample[1000, :], color="purple")
    plt.hlines([0.0, -magnitude_of_double_well_position, magnitude_of_double_well_position], 0.0, len(position_sample[100, :]))
    plt.savefig("worldlines_metropolis.png")

    plt.clf()

    #well data
    avg_well_hop = 0
    for particle in range(number_of_particles):
        well_hop = 0
        for trace_sample in range(number_of_observations):
            if np.sign(position_sample[(trace_sample - 1) % number_of_observations, particle]) != \
                np.sign(position_sample[trace_sample, particle]):
                well_hop += 1
        
        avg_well_hop += well_hop
    
    avg_well_hop /= number_of_particles
    print(f"Average number of barrier jumps per particle for Metropolis =  {avg_well_hop}")

    config = parsing.read_config(parsing.parse_options([ecmc_config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, temperature,
    number_of_equilibration_iterations, number_of_observations, number_of_particles,
    _) = helper_methods.get_basic_config_data(ecmc_config_file_string)



    position_sample = sample_getter.get_positions(sample_directory, temperature, 0, number_of_particles,
                                                       number_of_equilibration_iterations,
                                                       thinning_level=thinning_level)
    
    print(np.shape(position_sample))
    plt.plot(np.arange(len(position_sample[:, 6])), position_sample[:, 6], color="orange")
    plt.hlines([0.0, -magnitude_of_double_well_position, magnitude_of_double_well_position], 0.0, len(position_sample[:, 6]))
    plt.xlim((0.0, 1000))

    plt.savefig("trace_ecmc.png") 

    plt.clf()
    plt.plot(np.arange(len(position_sample[100, :])), position_sample[100, :], color="magenta")
    plt.plot(np.arange(len(position_sample[1000, :])), position_sample[1000, :], color="purple")
    plt.hlines([0.0, -magnitude_of_double_well_position, magnitude_of_double_well_position], 0.0, len(position_sample[100, :]))
    plt.savefig("worldlines_ecmc.png")

    avg_well_hop = 0
    for particle in range(number_of_particles):
        well_hop = 0
        for trace_sample in range(number_of_observations):
            if np.sign(position_sample[(trace_sample - 1) % number_of_observations, particle]) != \
                np.sign(position_sample[trace_sample, particle]):
                well_hop += 1
        
        avg_well_hop += well_hop
    
    avg_well_hop /= number_of_particles
    print(f"Average number of barrier jumps per particle for ECMC =  {avg_well_hop}")






if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])