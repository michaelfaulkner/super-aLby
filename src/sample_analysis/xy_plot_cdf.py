import os 
import re
import sys 
import glob 
import importlib
import numpy as np
import matplotlib.pyplot as plt
from markov_chain_diagnostics import get_cumulative_distribution, get_effective_sample_size

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
parsing = importlib.import_module("base.parsing")
helper_methods = importlib.import_module("helper_methods")

plt.style.use('ggplot')

def main(config_file_string):
    """
    Test convergence of XY model simulation by plotting CDF with.
    
    Parameters
    ----------
    config_file_string : str 
        Location of the config file.
    """
    config_file_path = parsing.parse_options([config_file_string]).config_file
    config = parsing.read_config(config_file_path)
    (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
     _, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)
    
    directory_path = sample_directories[0]
    file_path = os.path.join(directory_path, 'temperature_00_checkpoint_*_sample_of_magnetisation_norm.npy') # Creates wildcard condition
    just_file_path = file_path.split('/')[-1].split('.')[0]
    sample_paths = glob.glob(file_path) # Finds all filepaths satisfying wildcard condition 
    sample_paths = sorted(sample_paths, key=lambda fname: int(re.search(r"checkpoint_(\d{2})", fname).group(1)))

    sample = []
    for sample_path in sample_paths:
        sample.append(np.load(sample_path).flatten())
    sample = np.concatenate(sample)

    sample_cdf = get_cumulative_distribution(sample)
    
    eff_sample_size = get_effective_sample_size(sample)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(sample_cdf[0], sample_cdf[1], color='k', linestyle='-', alpha=0.8, label=f'Simulation\nN_eff={eff_sample_size:.2f}')
    ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10, loc='lower right')
    ax.set_title(file_path, fontsize=10)
    plt.savefig(os.path.join(directory_path, f'compare_cdf_{just_file_path}.png'))

if __name__ == '__main__':
    main(sys.argv[1])


