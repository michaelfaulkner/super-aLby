from markov_chain_diagnostics import get_autocorrelation, get_integrated_autocorrelation_time
import importlib
import math
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



def main(config_folder):
    iact_arr = np.zeros(len(os.listdir(config_folder)))
    timestep_arr = np.zeros(len(os.listdir(config_folder)))

    for index, file in enumerate(os.listdir(config_folder)):
        config_file_string = os.path.join(config_folder, file)

        config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
        (config_file_mediator, potential, samplers, sample_directories, temperatures,
        number_of_equilibration_iterations, number_of_observations, number_of_particles,
        _, _, _) = helper_methods.get_basic_config_data(config_file_string)
        
        mass = parsing.get_value(config, strings.to_camel_case(potential), "mass")
        timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
        number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")
        sample_directory = sample_directories[0]
        temperature_index = 0
        thinning_level = None
    
        timestep_arr[index] = timestep
        print(f"current timestep = {timestep}, getting acf")

        
        mean_sample = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                            temperature_index, number_of_particles, number_of_equilibration_iterations,
                            thinning_level=thinning_level)
        #mean_sample = mean_sample[:]

        acf = get_autocorrelation(mean_sample[:,0])

        

        # plt.plot(np.arange(0, len(mean_sample)), acf)
        # plt.xlabel("sample index")
        # plt.ylabel("autocorrelation function")
        # plt.title("Autocorrelation Function for delta tau = 0.01")
        # plt.savefig("acf_qho_001.png")
        # plt.clf()

        iact = get_integrated_autocorrelation_time(acf)
        iact_arr[index] = iact
    
    save_arr = np.zeros((len(iact_arr), 2))
    save_arr[:, 0] = iact_arr
    save_arr[:, 1] = timestep_arr

    np.save("output/iact_ecmc.npy", save_arr)
    
    # fig, ax = plt.subplots(1, 1)
    # ax.scatter(timestep_arr, iact_arr)
    # ax.set_xlabel("delta tau")
    # ax.set_ylabel("iact")
    # plt.savefig("iact.png")

if __name__ == '__main__':
    main(sys.argv[1])