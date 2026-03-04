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

def spatial_correlation_function(positions, length, number_of_particles):
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
    corr_func = np.zeros(number_of_particles)
    for particle_index in range(number_of_particles):
        corr_func[particle_index] = np.mean(positions[:, particle_index] * positions[:, (particle_index + length)%number_of_particles]) - \
            np.mean(positions[:, particle_index]) * np.mean(positions[:, (particle_index + length)%number_of_particles])

    return np.mean(corr_func)


def main(config_file_string, min_length, max_length, N_repeats):

    min_length = int(min_length)
    max_length = int(max_length)
    N_repeats = int(N_repeats)

    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, temperature, number_of_equilibration_iterations,
     _, number_of_particles, size_of_particle_space) = helper_methods.get_basic_config_data(config_file_string)
    

    thinning_level = None
    timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")

    lengths = np.arange(min_length, max_length, step = 5)
    spatial_correlations = np.zeros((len(lengths), N_repeats))
    #spatial_correlations_err = np.zeros(len(lengths))

    for n in range(N_repeats):
        print(n)
        n_sample_directory = os.path.join(sample_directory, f"{n}")
        position_sample = sample_getter.get_positions(n_sample_directory, temperature, 0, number_of_particles,
                                                  number_of_equilibration_iterations, thinning_level=thinning_level)
        for index, length in enumerate(lengths):
            spatial_correlations[index, n]= spatial_correlation_function(position_sample, length, number_of_particles)
    
    spatial_correlations_err = np.std(spatial_correlations, axis = 1)
    spatial_correlations = np.mean(spatial_correlations, axis = 1)

    print(np.shape(spatial_correlations_err))
    print(np.shape(spatial_correlations))


    print(spatial_correlations[spatial_correlations < 0])
    fig, ax = plt.subplots(1,1)

    ax.errorbar(lengths[spatial_correlations_err>0], spatial_correlations[spatial_correlations_err>0], yerr = spatial_correlations_err[spatial_correlations_err>0], fmt="o", capsize=5)


    ax.set_xlabel(r"$\Delta \tau$")
    ax.set_ylabel("C(r)")
    ax.set_yscale("log")
    ax.set_xscale("linear")
    print(ax.get_ylim())
    #plt.legend()
    

    plt.savefig("correlation_func.png")

    fig, ax = plt.subplots(1,1)

    ax.errorbar(lengths, spatial_correlations, yerr = spatial_correlations_err, fmt="o", capsize=5)


    ax.set_xlabel(r"$\Delta \tau$")
    ax.set_ylabel("C(r)")
    ax.set_yscale("linear")
    ax.set_xscale("linear")
    print(ax.get_ylim())
    plt.savefig("correlation_func_linear.png")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])