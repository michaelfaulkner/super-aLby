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

def main(config_file_string):
    """
    Plot 2D samples.
    
    Parameters
    ----------
    config_file_path : str 
        Location of the config file.
    """
    config_file_path = parsing.parse_options([config_file_string]).config_file
    config = parsing.read_config(config_file_path)
    (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
     _, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)

    directory_path = sample_directories[0]
    file_path = os.path.join(directory_path, 'temperature_00_checkpoint_*_sample_of_magnetisation_vector.npy')
    samples_paths = glob.glob(file_path)
    samples_paths = sorted(samples_paths, key=lambda fname: int(re.search(r"checkpoint_(\d{2})", fname).group(1)))

    samples = []
    for samples_path in samples_paths:
        run_samples = np.load(samples_path)
        samples.append(run_samples)
    samples = np.array(samples).reshape(-1, 2)

    just_file_path = file_path.split('/')[-1].split('.')[0]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(samples[:, 0], samples[:, 1], color='firebrick', alpha=0.8)
    ax.set_title(file_path, fontsize=10)
    plt.savefig(os.path.join(directory_path, f'{just_file_path}.png'))

if __name__ == '__main__':
    main(sys.argv[1])


