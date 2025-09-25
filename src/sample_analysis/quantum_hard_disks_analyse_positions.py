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
    (config_file_mediator, potential, _, samplers, sample_directories, temperature, number_of_equilibration_iterations,
    number_of_observations, number_of_particles, _) = helper_methods.get_basic_config_data(config_file_string)
    
    mass = parsing.get_value(config, strings.to_camel_case(potential), "mass")
    timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
    number_of_quantum_particles = parsing.get_value(config, "ModelSettings", "number_of_quantum_particles")
    number_of_timeslices = parsing.get_value(config, "ModelSettings", "number_of_timeslices")
    number_of_particles = number_of_quantum_particles * number_of_timeslices
    disk_radius = parsing.get_value(config, "QuantumHardDiskPotential", "disk_radius")
    disk_radius = parsing.get_value(config, "QuantumHardDiskPotential", "disk_radius")
    range_of_initial_particle_positions  = parsing.get_value(config, "ModelSettings",
                                                             "range_of_initial_particle_positions")
    

    sample_directory = sample_directories[0]
    temperature_index = 0
    thinning_level = None
    number_of_equilibration_iterations = None
    position_sample = sample_getter.get_positions(sample_directory, temperature, 0, number_of_particles,
                                                  number_of_equilibration_iterations, thinning_level=thinning_level)

    # test_positions = np.load("output/convergence_tests/quantum_hard_disk_potential/event_chain/positions_testing.npy")
    # sample_info = np.load("output/convergence_tests/quantum_hard_disk_potential/event_chain/sample_info.npy")




    worldline_0 = position_sample[:, 0:number_of_quantum_particles]

    plt.figure(figsize=[20,10])
    ax = plt.axes(ylim=range_of_initial_particle_positions)
    start = 0
    num_of_samples = 100
    for particle_index in range(number_of_quantum_particles):

        ax.scatter(np.arange(start,num_of_samples), worldline_0[start:num_of_samples, particle_index], s=100, label = particle_index)
        ax.errorbar(np.arange(start,num_of_samples), worldline_0[start:num_of_samples, particle_index],
                        yerr=disk_radius * np.ones(np.shape(np.arange(start,num_of_samples))), fmt=".", capsize=5.0)
        for index, point in enumerate(worldline_0[start:num_of_samples, particle_index]):
            if point > range_of_initial_particle_positions[1] - disk_radius:
                # loop round to -5
                leftover = point + disk_radius - range_of_initial_particle_positions[1]
                ax.errorbar(index+start, range_of_initial_particle_positions[0], yerr=leftover, fmt=".", capsize=5.0,
                            color = plt.gca().lines[-1].get_color())
            elif point < range_of_initial_particle_positions[0] + disk_radius:
                # loop round to +5
                leftover = np.abs(point - disk_radius - range_of_initial_particle_positions[0])
                ax.errorbar(index+start, range_of_initial_particle_positions[1], yerr=leftover, fmt=".", capsize=5.0,
                            color = plt.gca().lines[-1].get_color())

    ax.set_yticks(np.arange(-5,6))
    plt.tight_layout()
    plt.legend()
    plt.savefig("test.png")

    # avg_samples = len(test_positions[:,0,0])/len(position_sample[:,0])
    # print(avg_samples)
    
    # test_positions_0 = test_positions[:, 0:9, 0]
    # plt.figure(figsize=[20,10])
    # ax = plt.axes(ylim=range_of_initial_particle_positions)
    # start = int(np.round(2.5 * avg_samples, 0))
    # num_of_samples = int(np.round(5.0 * avg_samples, 0))
    # for particle_index in range(number_of_quantum_particles):
    #     ax.scatter(np.arange(start,num_of_samples), test_positions_0[start:num_of_samples, particle_index], s=100, label = particle_index)
    #     ax.errorbar(np.arange(start,num_of_samples), test_positions_0[start:num_of_samples, particle_index],
    #                 yerr=disk_radius * np.ones(np.shape(np.arange(start,num_of_samples))), fmt=".", capsize=5.0)
    #     for index, point in enumerate(test_positions_0[start:num_of_samples, particle_index]):
    #         if point > range_of_initial_particle_positions[1] - disk_radius:
    #             # loop round to -5
    #             leftover = point + disk_radius - range_of_initial_particle_positions[1]
    #             ax.errorbar(index+start, range_of_initial_particle_positions[0], yerr=leftover, fmt=".", capsize=5.0,
    #                         color = plt.gca().lines[-1].get_color())
    #         elif point < range_of_initial_particle_positions[0] + disk_radius:
    #             # loop round to +5
    #             leftover = np.abs(point - disk_radius - range_of_initial_particle_positions[0])
    #             ax.errorbar(index+start, range_of_initial_particle_positions[1], yerr=leftover, fmt=".", capsize=5.0,
    #                         color = plt.gca().lines[-1].get_color())
    #         if index == 1:
    #             ax.scatter(index+start, -4.9, s=100, color="black", marker= "x")
    
    # ax.set_yticks(np.arange(-5,6))
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig("positions.png")
   
if __name__ == '__main__':
    main(sys.argv[1])