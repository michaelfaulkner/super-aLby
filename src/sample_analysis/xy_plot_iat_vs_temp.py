import os 
import re
import sys 
import glob 
import math
import importlib
import numpy as np
import matplotlib.pyplot as plt
from markov_chain_diagnostics import get_integrated_autocorrelation_time

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
parsing = importlib.import_module("base.parsing")
helper_methods = importlib.import_module("helper_methods")

plt.style.use('ggplot')

def main(config_file_string):
    """
    Plot magnetisation temperature sweep and fit for critial exponent beta. 

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

    iats = []

    for temp_index, temperature in enumerate(temperatures):
        file_path = os.path.join(directory_path, f'temperature_{temp_index:02d}_checkpoint_*_sample_of_magnetisation_norm.npy')
        sample_paths = glob.glob(file_path)
        sample_paths = sorted(sample_paths, key=lambda fname: int(re.search(r"checkpoint_(\d{2})", fname).group(1)))

        sample = []
        for sample_path in sample_paths:
            sample.append(np.load(sample_path).flatten())
        sample = np.concatenate(sample)
        sample = sample[number_of_equilibration_iterations:] # Remove burn-in
        
        iat = get_integrated_autocorrelation_time(sample, cutoff=math.e ** (-4))
        iats.append(iat)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(temperatures, iats, color='firebrick', alpha=0.6, label=f'{potential} {config_file_mediator}')
    ax.set_xlabel('T', fontsize=14, color='black')
    ax.set_ylabel('IAT', fontsize=14, color='black')
    ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10, loc='upper right')
    ax.set_title('IAT vs T', fontsize=18, color='black')
    plt.savefig(os.path.join(directory_path, f'iat_vs_temp.png'))


if __name__ == '__main__':
    main(sys.argv[1])
