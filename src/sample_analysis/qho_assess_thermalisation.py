import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import sample_getter
import sys
from markov_chain_diagnostics import get_cumulative_distribution
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")


def main(metrop_config_file_string, ecmc_config_file_string):
    config = parsing.read_config(
        parsing.parse_options([metrop_config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, metrop_sample_directory, temperature,
        number_of_equilibration_iterations, number_of_observations, number_of_particles,
        size_of_particle_space) = helper_methods.get_basic_config_data(metrop_config_file_string)

    print(f"sample from {metrop_sample_directory}")

    temperature_index = 0
    thinning_level = None
    number_of_equilibration_iterations = 0
    metrop_mean_squared_sample = sample_getter.get_mean_squared_positions(metrop_sample_directory, temperature, 0, number_of_particles, number_of_equilibration_iterations,
                                    thinning_level=thinning_level)

    metrop_mean_squared_sample = np.mean(metrop_mean_squared_sample, axis = 1)

    config = parsing.read_config(
        parsing.parse_options([ecmc_config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, temperature,
        number_of_equilibration_iterations, number_of_observations, number_of_particles,
        size_of_particle_space) = helper_methods.get_basic_config_data(ecmc_config_file_string)

    print(f"sample from {sample_directory}")


    temperature_index = 0
    thinning_level = None
    number_of_equilibration_iterations = 0
    ecmc_mean_squared_sample = sample_getter.get_mean_squared_positions(sample_directory, temperature, 0, number_of_particles, number_of_equilibration_iterations,
                                    thinning_level=thinning_level)

    ecmc_mean_squared_sample = np.mean(ecmc_mean_squared_sample, axis = 1)

    fig, ax = plt.subplots(2,1, sharex=True, sharey = True)
    sample_index_cutoff = 10000
    ax[0].plot(np.arange(sample_index_cutoff), metrop_mean_squared_sample[:sample_index_cutoff], color = "#e16f04ff", label ="Metropolis MC")

    ax[1].plot(np.arange(sample_index_cutoff), ecmc_mean_squared_sample[:sample_index_cutoff], color = "#e20acdff", label = "ECMC")

    ax[1].set_xlabel("sample index", fontsize = 15, weight = "bold")
    ax[0].set_ylabel(r"$\langle x^2 \rangle$", fontsize = 15, weight = "bold")
    ax[1].set_ylabel(r"$\langle x^2 \rangle$", fontsize = 15, weight = "bold")
    legend_properties = {'weight':'bold'}
    fig.legend(loc  = (0.65, 0.45), prop=legend_properties)
    plt.savefig("thermalisation.pdf", transparent = True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
