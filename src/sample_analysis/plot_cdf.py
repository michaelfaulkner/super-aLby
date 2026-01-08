import importlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from markov_chain_diagnostics import get_cumulative_distribution

# Add the directory that contains the module plotting_functions to sys.path
this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")


def main(config_file_strings, sample_name, figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    sample_directories = []
    colours = ['firebrick', 'black', 'grey', 'blue', 'green', 'yellow']
    for i, config_file_string in enumerate(config_file_strings):
        config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
        (config_file_mediator, potential, _, samplers, sample_directory, temperature, number_of_equilibration_iterations,
         _, number_of_particles, size_of_particle_space) = helper_methods.get_basic_config_data(config_file_string)
        sample_directories.append(sample_directory)
        try:
            sample = np.load(os.path.join(sample_directory, f"checkpoint_00_sample_of_{sample_name}.npy")).flatten()
        except FileNotFoundError:
            continue
        sample, cdf_sample = get_cumulative_distribution(sample[number_of_equilibration_iterations:])
        plt.plot(sample, cdf_sample, alpha=0.6, color=colours[i], linewidth=2.5, label=os.path.basename(sample_directory))

    plt.xlabel(sample_name, fontsize=14)
    plt.ylabel("CDF", fontsize=14)

    plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="upper right")

    plt.tight_layout()
    for samp_directory in sample_directories:
        plt.savefig(os.path.join(samp_directory, f"{sample_name}_cdf.png"))

    plt.show()


if __name__ == '__main__':
    *configs, samp_name = sys.argv[1:]
    main(configs, samp_name)
