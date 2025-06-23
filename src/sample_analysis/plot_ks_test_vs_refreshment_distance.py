import re
import os 
import sys
import glob 
import importlib
import numpy as np 
import configparser
import matplotlib.pyplot as plt 
from markov_chain_diagnostics import get_ks_test

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")

def extract_index(path):
    match = re.search(r'_(\d+)$', os.path.basename(path))
    return int(match.group(1)) if match else -1

def main(config_directory_path, reference_sample_path):
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
    config.read(os.path.join(config_directories[0], 'job_00.ini'))
    number_of_equilibration_iterations = int(config.get("EventChainMediator", "number_of_equilibration_iterations"))
    output_directory_path = os.path.dirname(os.path.dirname(config.get("XyMagnetisationNormSampler", "output_directory")))

    directories = [directory for directory in glob.glob(f"{output_directory_path}/*") if "png" not in directory]
    directories = sorted(directories, key=extract_index)
    all_samples = []
    for directory in directories:
        subdirectories = [os.path.join(directory, subdirectory) for subdirectory in os.listdir(directory) if "png" not in subdirectory]
        all_subdirectoy_samples = []
        for subdirectory in subdirectories:
            try:
                samples = np.load(os.path.join(subdirectory, 'temperature_00_checkpoint_00_sample_of_magnetisation_norm.npy')).flatten()
                samples = samples[number_of_equilibration_iterations:]
                all_subdirectoy_samples.append(samples)
            except IOError: 
                pass
        all_samples.append(all_subdirectoy_samples)

    reference_sample = np.load(reference_sample_path).flatten()
    
    ks_tests = []
    for samples in all_samples:
        samples_ks_tests = []
        for sample in samples:
            ks_test = get_ks_test(sample, reference_sample)[0]
            samples_ks_tests.append(ks_test)
        ks_tests.append(np.mean(samples_ks_tests))
    ks_tests = np.array(ks_tests)

    config.read(os.path.join(config_directories[0], 'job_00'))
    temperature = config.get("EventChainMediator", "minimum_temperature")

    plt.scatter(distance_between_velocity_refreshments, ks_tests, color='firebrick', label=f"T={temperature}")
    plt.xscale('log')
    plt.xlabel('Velocity Refreshment Distance')
    plt.ylabel('KS Test')
    plt.title('Optimising ECMC Velocity Refreshment Distance')
    plt.grid(True, alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(output_directory_path, "ks_test_vs_refreshment_dist.png"))

    print("ks_test vs refreshment plot saved.")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

