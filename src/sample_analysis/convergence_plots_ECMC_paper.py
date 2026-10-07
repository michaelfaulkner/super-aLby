from configparser import NoOptionError
from markov_chain_diagnostics import get_cumulative_distribution, get_sample_mean_and_error
import importlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sample_getter
import sys
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')

# Add the directory that contains the module plotting_functions to sys.path
this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")
matplotlib.use('Agg')

def main(qho_config_file_string, aho_config_file_string):

    """config_file_string is the location of the config file"""
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    """nb, argument of parsing.parse_options() must be of type Sequence[str]"""
    config = parsing.read_config(parsing.parse_options([qho_config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, qho_sample_directory, temperature, qho_number_of_equilibration_iterations,
     _, qho_number_of_particles, qho_size_of_particle_space) = helper_methods.get_basic_config_data(qho_config_file_string)



    config = parsing.read_config(parsing.parse_options([aho_config_file_string]).config_file)
    (config_file_mediator, potential, _,  samplers, aho_sample_directory, temperature, aho_number_of_equilibration_iterations,
     _, aho_number_of_particles, aho_size_of_particle_space) = helper_methods.get_basic_config_data(aho_config_file_string)
    

    
    
    qho_reference_sample = np.load("/home/raichkel/super-aLby-private/src/output/convergence_tests/quantum_harmonic_oscillator_potential/metropolis/checkpoint_00_sample_of_mean_squared_positions.npy").flatten()
    aho_reference_sample = np.load("/home/raichkel/super-aLby-private/src/output/convergence_tests/quantum_anharmonic_oscillator_potential/metropolis/checkpoint_00_sample_of_mean_squared_positions.npy").flatten()

    qho_reference_cdf = get_cumulative_distribution(qho_reference_sample)
    aho_reference_cdf = get_cumulative_distribution(aho_reference_sample)
       

    qho_sample = sample_getter.get_mean_squared_positions(qho_sample_directory, temperature, 0,
                                                              qho_number_of_particles, qho_number_of_equilibration_iterations
                                                              ).flatten()
    aho_sample = sample_getter.get_mean_squared_positions(aho_sample_directory, temperature, 0,
                                                            aho_number_of_particles, aho_number_of_equilibration_iterations
                                                            ).flatten()

    qho_sample_cdf = get_cumulative_distribution(qho_sample)
    aho_sample_cdf = get_cumulative_distribution(aho_sample)
    
    fig, ax = plt.subplots(1,1)
    ax.plot(qho_reference_cdf[0], qho_reference_cdf[1], color="#e61267", linewidth=4, linestyle='-',
                label=f'Metropolis MC - QHO')
    ax.plot(qho_sample_cdf[0], qho_sample_cdf[1], color="#352626", linewidth=3, linestyle='-',
                label=f'ECMC - QHO')
    ax.plot(aho_reference_cdf[0], aho_reference_cdf[1], color="#29e218", linewidth=3, linestyle='-',
                label=f'Metropolis MC - Anharmonic')
    ax.plot(aho_sample_cdf[0], aho_sample_cdf[1], color="#5D0B7B", linewidth=3, linestyle='-',
                label=f'ECMC - QHO')
   

    ax.set_xlabel(r"$x$", fontsize=40, weight='bold')
    ax.set_ylabel(r"$ F_n \left( X < x \right)$", fontsize=25, weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=16)


    legend_properties = {'weight':'bold'}
    legend = ax.legend(fontsize=10, prop=legend_properties)
    legend.get_frame().set_edgecolor('k')
    plt.tight_layout()
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

    #plt.show()

    ax.tick_params(direction="in", left="off",labelleft="off")
    plt.tight_layout()
    plt.savefig("convergence_paper.pdf")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])