import re
import os
import sys
import glob
import importlib
import numpy as np
import configparser
import matplotlib.pyplot as plt
from markov_chain_diagnostics import get_iact_and_acf

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")

def extract_index(path):
    match = re.search(r'_(\d+)$', os.path.basename(path))
    return int(match.group(1)) if match else -1

def main(config_directory_path):
    prefactors = []
    config = configparser.ConfigParser()
    config.optionxform = str
    config_directories = [d for d in glob.glob(f"{config_directory_path}/*")]
    config_directories = sorted(config_directories, key=extract_index)
    for config_directory in config_directories:
        config_file_path = os.path.join(config_directory, 'job_00.ini')
        config.read(config_file_path)
        prefactor = float(config.get("XyFactorField", "prefactor"))
        prefactors.append(prefactor)
    prefactors = np.array(prefactors)
    config.read(os.path.join(config_directories[0], 'job_00.ini'))
    number_of_equilibration_iterations = int(config.get("EventChainMediator", "number_of_equilibration_iterations"))
    output_directory_path = os.path.dirname(os.path.dirname(config.get("XyMagnetisationNormSampler", "output_directory")))

    directories = [directory for directory in glob.glob(f"{output_directory_path}/*") if "png" not in directory
                   and "npy" not in directory]
    directories = sorted(directories, key=extract_index)
    all_samples = []
    for directory in directories:
        subdirectories = [os.path.join(directory, subdirectory) for subdirectory in os.listdir(directory) if "png" not in subdirectory]
        all_subdirectoy_samples = []
        for subdirectory in subdirectories:
            try:
                samples = np.load(os.path.join(subdirectory, 'xy_16x16_T=1.1_mag_norm_ref.npy')).flatten()
                samples = samples[number_of_equilibration_iterations:]
                all_subdirectoy_samples.append(samples)
            except IOError:
                pass
        all_samples.append(all_subdirectoy_samples)

    iacts = []
    for samples in all_samples:
        samples_iacts = []
        for sample in samples:
            iact = get_iact_and_acf(sample)[0]
            samples_iacts.append(iact)
        iacts.append(np.mean(samples_iacts))
    iacts = np.array(iacts)

    temp = float(config.get("EventChainMediator", "minimum_temperature"))
    L = float(config.get("ModelSettings", "number_of_particles")) ** 0.5
    np.save(os.path.join(output_directory_path, "iact_vs_prefactor.npy"), np.vstack([prefactors, iacts]))

    plt.scatter(prefactors, iacts, color='firebrick', label=f'{L}x{L}\nT={temp}')
    plt.xlabel('L * Prefactor')
    plt.ylabel('IACT')
    plt.title('Optimising Factor Field Prefactor')
    plt.grid(True, alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(output_directory_path, "iact_vs_prefactor.png"))

    print("iact vs prefactor plot saved.")

if __name__ == "__main__":
    main(sys.argv[1])
