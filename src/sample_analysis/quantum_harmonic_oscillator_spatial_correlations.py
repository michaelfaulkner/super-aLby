import numpy as np
import os
import importlib
import matplotlib
import matplotlib.pyplot as plt
import os
import sample_getter
import sys


this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")
matplotlib.rcParams['mathtext.fontset'] = 'cm'

def spatial_correlation_function(positions, length):
    """
    returns spatial correlation function for the positions sample. 
    C(r) = <x_{0} x_{r}> - <x_{0}><x_{r}>
    
    Parameters
    -----------
    positons : np array
        The array containing the position data sample
    length: float
        The distance between the first and second indices of the correlation function.
    """
    return np.mean(positions[:, 0] * positions[:, length]) - np.mean(positions[:, 0]) * np.mean(positions[:, length])


def main(config_file_string, min_length, max_length):

    min_length = int(min_length)
    max_length = int(max_length)

    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, temperature, number_of_equilibration_iterations,
     _, number_of_particles, size_of_particle_space) = helper_methods.get_basic_config_data(config_file_string)
    

    thinning_level = None
    position_sample = sample_getter.get_positions(sample_directory, temperature, 0, number_of_particles,
                                                  number_of_equilibration_iterations, thinning_level=thinning_level)
    timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")

    lengths = np.arange(min_length, max_length)
    spatial_correlations = np.zeros(len(lengths))

    for index, length in enumerate(lengths):
        spatial_correlations[index] = spatial_correlation_function(position_sample, length)

    fig, ax = plt.subplots(1,1)

    ax.scatter(lengths, spatial_correlations)
    ax.set_xlabel(r"length of correlation function (units of $\delta \tau$)")
    ax.set_ylabel("C(r)")

    plt.savefig("correlation_func.png")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])