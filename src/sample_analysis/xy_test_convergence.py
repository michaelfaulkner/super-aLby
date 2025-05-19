from configparser import NoOptionError
from markov_chain_diagnostics import get_cumulative_distribution, get_effective_sample_size, get_sample_mean_and_error
import importlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sample_getter
import sys

# Add the directory that contains the module plotting_functions to sys.path
this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")

def main(config_file_string):
    """config_file_string is the location of the config file"""
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    """nb, argument of parsing.parse_options() must be of type Sequence[str]"""
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
     _, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)
    
    reference_cdf = np.load('src/permanent_data/reference_data/xy_reference_potential_8x8_0.8.npy')
    
    sample = sample_getter.get_potential(sample_directories[0], temperatures[0], 0,
                                                      number_of_particles).flatten()
    effective_sample_size = get_effective_sample_size(sample)

    print(f"Effective sample size = {effective_sample_size} (from a total sample size of {len(sample)}).")
    sample_cdf = get_cumulative_distribution(sample)

    plt.plot(reference_cdf[0], reference_cdf[1], color='r', linewidth=3, linestyle='-', label='reference data')
    plt.plot(sample_cdf[0], sample_cdf[1], color='k', linewidth=2, linestyle='-', label='super-aLby data')

    plt.xlabel(r"$x$", fontsize=15, labelpad=10)
    plt.ylabel(r"$ F_n \left( X < x \right)$", fontsize=15, labelpad=10)
    plt.tick_params(axis='both', which='major', labelsize=14, pad=10)
    legend = plt.legend(loc='lower right', fontsize=10)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)
    plt.tight_layout()
    #plt.show()
    plt.savefig("output/potential_convergence_test.png")


if __name__ == '__main__':
    main(sys.argv[1])
