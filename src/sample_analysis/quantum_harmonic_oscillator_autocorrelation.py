from markov_chain_diagnostics import get_iact_and_acf
import importlib
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
        (config_file_mediator, potential, _, samplers, sample_directories, temperature,
         number_of_equilibration_iterations, number_of_observations, number_of_particles,
         _) = helper_methods.get_basic_config_data(config_file_string)
        
        mass = parsing.get_value(config, strings.to_camel_case(potential), "mass")
        timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
        number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")
        sample_directory = sample_directories[0]
        temperature_index = 0
        thinning_level = None
        timestep_arr[index] = timestep

        if timestep == 0.01:
            print(f"Timestep = {timestep}, getting files from checkpoints")
            sub_arr_len = 10000
            num_sub_arrs = 13
            mean_sample = np.zeros(sub_arr_len * num_sub_arrs)
            for i in range(num_sub_arrs):
                mean_sample[i * sub_arr_len:(i+1) * sub_arr_len] = np.load(
                    f"output/metropolis_001_checkpoints/run_{i:02d}_sample_of_mean_positions.npy")[1:, 0]
            mean_sample = mean_sample[50000:129]
        else:
            print(f"current timestep = {timestep}, getting acf")
            mean_sample = sample_getter.get_mean_positions(sample_directory, temperature, number_of_particles,
                                                           number_of_equilibration_iterations,
                                                           thinning_level=thinning_level)
            mean_sample = mean_sample[:80000]
        iact = get_iact_and_acf(mean_sample[:])[0]
        iact_arr[index] = iact
    
    save_arr = np.zeros((len(iact_arr), 2))
    save_arr[:, 0] = iact_arr
    save_arr[:, 1] = timestep_arr
    np.save("output/iact_metropolis.npy", save_arr)
    
    # fig, ax = plt.subplots(1, 1)
    # ax.scatter(timestep_arr, iact_arr)
    # ax.scatter(timestep_arr, iact_arr)
    # ax.set_xlabel("delta tau")
    # ax.set_ylabel("iact")
    # plt.savefig("iact.png")


if __name__ == '__main__':
    main(sys.argv[1])
