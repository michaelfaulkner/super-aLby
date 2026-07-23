from configparser import NoOptionError
from markov_chain_diagnostics import get_cumulative_distribution, get_sample_mean_and_error, get_iact
import importlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sample_getter
import sys

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")

def main(config_file_string_1, config_file_string_2):
    """Compare samples from two different config files"""
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    config_1 = parsing. read_config(parsing.parse_options([config_file_string_1]).config_file)
    (_, _, _, _, sample_directory_1, temperature_1, _, 
     _, number_of_particles_1, _) = helper_methods.get_basic_config_data(config_file_string_1)
    config_2 = parsing.read_config(parsing.parse_options([config_file_string_2]).config_file)
    (_, _, _, _, sample_directory_2, temperature_2, number_of_equilibration_iterations_2,
     _, number_of_particles_2, _) = helper_methods.get_basic_config_data(config_file_string_2)
    sample_1 = sample_getter.get_structure_factor(sample_directory_1, temperature_1, 0,
                                                   number_of_particles_1,
                                                   number_of_equilibration_iterations_2).flatten()
    sample_2 = sample_getter.get_structure_factor(sample_directory_2, temperature_2, 0,
                                                   number_of_particles_2,
                                                   number_of_equilibration_iterations_2).flatten()
    sample1_cdf = get_cumulative_distribution(sample_1)
    sample2_cdf = get_cumulative_distribution(sample_2)

    iact_1 = get_iact(sample_1)
    iact_2 = get_iact(sample_2)
    
    plt.plot(sample1_cdf[0], sample1_cdf[1], color='r', linewidth=3, linestyle='-', label=f'Event Chain\n IACT: {iact_1:.3f}')
    plt.plot(sample2_cdf[0], sample2_cdf[1], color='k', linewidth=2, linestyle='-', label=f'metropolis\n IACT: {iact_2:.3f}')
    plt.tick_params(axis='both', which='major', labelsize=14, pad=10)
    legend = plt.legend(loc='lower right', fontsize=10)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)
    plt.xlabel("$S(k)$", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF vs $S(k)$ | probability=0.5, packing_fraction=0.8, N=8, ratio=1:1")
    plt.tight_layout()
    plt.show()

    plt.savefig("binary_compare.png")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])