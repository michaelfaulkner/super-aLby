import numpy as np
import os
import importlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sample_getter
import sys
from configparser import NoOptionError
from markov_chain_diagnostics import get_sample_mean_and_error

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")

def main(config_file_string):
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
    number_of_observations, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)
    
    mass = parsing.get_value(config, strings.to_camel_case(potential), "mass")
    timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
    number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")
    
    sample_directory = sample_directories[0]
    temperature_index = 0
    thinning_level = None
    number_of_equilibration_iterations = None

    timestep_str = str(timestep).replace(".", "")
    
    sub_arr_len = 11000
    num_sub_arrs = 2
    divisor = 2
    mean_sample = np.zeros(sub_arr_len * num_sub_arrs)
    position_sample_0 = np.zeros((int(sub_arr_len * num_sub_arrs / divisor), number_of_particles))
    position_sample_1 = np.zeros((int(sub_arr_len * num_sub_arrs / divisor), number_of_particles))


    for i in range(int(num_sub_arrs / divisor)):
        print(f"sample file {i}")
        mean_sample[i * sub_arr_len : (i+1) * sub_arr_len] = np.load(
        f"output/positions_001_hot_start/temperature_00_run_{i:02d}_sample_of_mean_positions.npy")[1:, 0]

        position_sample_0[i * sub_arr_len : (i+1) * sub_arr_len, :] = np.load(
        f"output/positions_001_hot_start/temperature_00_run_{i:02d}_sample_of_positions.npy")[1:, :]
    
    for i in range(int(num_sub_arrs / divisor), num_sub_arrs):
        print(f"positions_1, sample file {i}")
        mean_sample[i * sub_arr_len : (i+1) * sub_arr_len] = np.load(
        f"output/positions_001_hot_start/temperature_00_run_{i:02d}_sample_of_mean_positions.npy")[1:, 0]

        position_sample_1[(i- int(num_sub_arrs/divisor)) * sub_arr_len  : (i+1- int(num_sub_arrs/divisor)) * sub_arr_len, :] = np.load(
        f"output/positions_001_hot_start/temperature_00_run_{i:02d}_sample_of_positions.npy")[1:, :]
        print(f"sample file {i}, indices {(i- int(num_sub_arrs/divisor)) * sub_arr_len} : {(i+1- int(num_sub_arrs/divisor)) * sub_arr_len}")
    
    fig, ax = plt.subplots(1, 1, figsize = (15, 10))
    artists =[]

    for i in range(0,20000,20):
        if i < 11000:
            container = ax.scatter(np.arange(0, number_of_particles), position_sample_0[i, :], color = "purple")
        else: 
            container = ax.scatter(np.arange(0, number_of_particles), position_sample_1[i - sub_arr_len, :], color = "purple")
        artists.append([container])

    # for i in range(int(num_sub_arrs / divisor) * sub_arr_len):
    #     container1 = ax.scatter(np.arange(0, number_of_particles), position_sample_1[i, :], color = "purple")
    #     artists.append([container1])


    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=50)

    ani.save(filename="positions_metropolis.mp4", writer="ffmpeg")

if __name__ == '__main__':
    main(sys.argv[1])