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



def main(config_folder, N):
    N = int(N)

    for i in range(N):

        iact_arr = np.zeros(len(os.listdir(config_folder)))
        timestep_arr = np.zeros(len(os.listdir(config_folder)))

        for index, folder in enumerate(os.listdir(config_folder)):
            config_file_string = os.path.join(config_folder, folder, f"{i}.ini")

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
            checkpointing_indices = sample_getter.get_checkpointing_indices(sample_directory)

            if checkpointing_indices != 0:
                sub_arr_len = number_of_equilibration_iterations + number_of_observations
                mean_sample = np.zeros(sub_arr_len * (checkpointing_indices + 1))
                for i in range(checkpointing_indices + 1):
                    mean_sample[i * sub_arr_len : (i+1) * sub_arr_len] = sample_getter.get_mean_positions(
                        sample_directory, temperatures[temperature_index], temperature_index, number_of_particles, 
                        None, thinning_level=thinning_level)[1:, 0]
                    
                    # np.load(
                    #     os.path.join(sample_directory, f"temperature_00_run_{i:02d}_sample_of_mean_positions.npy"))[1:, 0]
                mean_sample = mean_sample[50000:129999]
            
            else:
                mean_sample = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                                temperature_index, number_of_particles, number_of_equilibration_iterations,
                                thinning_level=thinning_level)
                mean_sample = mean_sample[:80000]

                
            iact, acf = get_integrated_autocorrelation_time(mean_sample[:])
            iact_arr[index] = iact
        
        save_arr = np.zeros((len(iact_arr), 2))
        save_arr[:, 0] = iact_arr
        save_arr[:, 1] = timestep_arr

        np.save(f"output/iact_metropolis_{i}.npy", save_arr)
    

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])