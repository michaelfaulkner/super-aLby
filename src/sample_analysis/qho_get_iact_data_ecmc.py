from markov_chain_diagnostics import get_iact_and_acf
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


def main(config_folder, min, max, min_timestep):
    min = int(min)
    max = int(max)
    min_timestep = float(min_timestep)

    for i in range(min, max):

        iact_arr = np.zeros(len(os.listdir(config_folder)))
        timestep_arr = np.zeros(len(os.listdir(config_folder)))

        for index, folder in enumerate(os.listdir(config_folder)):
            config_file_string = os.path.join(
                config_folder, folder, f"{i}.ini")
            print(config_file_string)
            config = parsing.read_config(
                parsing.parse_options([config_file_string]).config_file)
            (config_file_mediator, potential, samplers, sample_directories, temperatures,
             number_of_equilibration_iterations, number_of_observations, number_of_particles,
             _, _, _) = helper_methods.get_basic_config_data(config_file_string)

            mass = parsing.get_value(
                config, strings.to_camel_case(potential), "mass")
            timestep = parsing.get_value(
                config, strings.to_camel_case(potential), "timestep")
            sample_directory = sample_directories[0]
            temperature_index = 0
            thinning_level = None
            if timestep >= min_timestep:
                print(timestep)
                timestep_arr[index] = timestep

                checkpointing_index = sample_getter.get_checkpointing_indices(sample_directory)
                if checkpointing_index != 0:
                    sub_arr_len = number_of_equilibration_iterations + number_of_observations
                    mean_sample = np.zeros(sub_arr_len * (checkpointing_index + 1))
                    for i in range(checkpointing_index + 1):
                        sub_arr = sample_getter.get_mean_squared_positions(
                            sample_directory, temperatures[temperature_index], temperature_index, i, number_of_particles, 
                            None, thinning_level=thinning_level)[:, 0]
                        try:
                            mean_sample[i * sub_arr_len : (i+1) * sub_arr_len] = sub_arr
                        except:
                            mean_sample[i * sub_arr_len : (i+1) * sub_arr_len] = sub_arr[1:]
                        
                    mean_sample = mean_sample[3000:]
                        
                else:
                    mean_sample = sample_getter.get_mean_squared_positions(sample_directory, temperatures[temperature_index],
                                    temperature_index, 0, number_of_particles, number_of_equilibration_iterations,
                                    thinning_level=thinning_level)
                
                if len(np.shape(mean_sample)) > 1:
                    mean_sample = mean_sample[:,0]
        

                iact, acf = get_iact_and_acf(mean_sample[:])

                iact_arr[index] = iact

        save_arr = np.zeros((len(iact_arr), 2))
        save_arr[:, 0] = iact_arr
        save_arr[:, 1] = timestep_arr

        np.save(f"output/iact_ecmc_{i}.npy", save_arr)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
