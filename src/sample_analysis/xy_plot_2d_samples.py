import os 
import re
import sys 
import glob 
import importlib
import numpy as np
import matplotlib.pyplot as plt

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
parsing = importlib.import_module("base.parsing")
helper_methods = importlib.import_module("helper_methods")

plt.style.use('ggplot')

def main(config_file_string, temperature_index='00'):
    """
    Plot 2D samples.
    
    Parameters
    ----------
    config_file_string : str 
        Location of the config file.
    temperature_index : str
        Index of simulation temperature.
    """
    config_file_path = parsing.parse_options([config_file_string]).config_file
    config = parsing.read_config(config_file_path)
    (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
     _, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)

    directory_path = sample_directories[0]
    file_path = os.path.join(directory_path, f'temperature_{temperature_index}_checkpoint_*_sample_of_magnetisation_vector.npy')
    just_file_path = file_path.split('/')[-1].split('.')[0]
    sample_paths = glob.glob(file_path) # Find all filepaths satisfying wildcard condition 
    sample_paths = sorted(sample_paths, key=lambda fname: int(re.search(r"checkpoint_(\d{2})", fname).group(1)))

    sample = []
    for sample_path in sample_paths:
        sample.append(np.load(sample_path))
    sample = np.concatenate(sample, axis=0).T

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(sample[0, :], sample[1, :], color='firebrick', alpha=0.6)
    ax.set_title(file_path, fontsize=10)
    plt.savefig(os.path.join(directory_path, f'{just_file_path}.png'))

if __name__ == '__main__':
    main(sys.argv[1])


