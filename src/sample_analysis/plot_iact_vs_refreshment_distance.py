import re
import os 
import sys
import glob 
import numpy as np 
import configparser
import matplotlib.pyplot as plt 
from markov_chain_diagnostics import get_iact_and_acf

def extract_index(path):
    match = re.search(r'_(\d+)$', os.path.basename(path))
    return int(match.group(1)) if match else -1

def main(directory_path, config_directory_path):
    directories = [directory for directory in glob.glob(f"{directory_path}/*") if "png" not in directory]
    directories = sorted(directories, key=extract_index)
    all_samples = []
    for directory in directories:
        subdirectories = [os.path.join(directory, subdirectory) for subdirectory in os.listdir(directory) if "png" not in subdirectory]
        all_subdirectoy_samples = []
        for subdirectory in subdirectories:
            try:
                samples = np.load(os.path.join(subdirectory, 'temperature_00_checkpoint_00_sample_of_magnetisation_norm.npy')).flatten()
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
    
    distance_between_velocity_refreshments = []
    config = configparser.ConfigParser()
    config.optionxform = str  
    config_directories = [d for d in glob.glob(f"{config_directory_path}/*")]
    config_directories = sorted(config_directories, key=extract_index)
    for config_directory in config_directories:
        config_file_path = os.path.join(config_directory, 'job_00.ini')
        config.read(config_file_path)
        distance_between_velocity_refreshment = float(config.get("EventChainMediator", "distance_between_velocity_refreshments"))
        distance_between_velocity_refreshments.append(distance_between_velocity_refreshment)
    distance_between_velocity_refreshments = np.array(distance_between_velocity_refreshments)

    config.read(os.path.join(config_directories[0], 'job_00'))
    temperature = config.get("EventChainMediator", "minimum_temperature")

    #optimal_distance = np.argmin(iacts)

    plt.scatter(distance_between_velocity_refreshments, iacts, color='firebrick', label=f"T={temperature}")
    #plt.axvline(x=optimal_distance, linestyle='--', color='k')
    plt.xscale('log')
    plt.xlabel('Velocity Refreshment Distance')
    plt.ylabel('IACT')
    plt.title('Optimising ECMC Velocity Refreshment Distance')
    plt.grid(True, alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(directory_path, "iact_vs_refreshment_dist.png"))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

