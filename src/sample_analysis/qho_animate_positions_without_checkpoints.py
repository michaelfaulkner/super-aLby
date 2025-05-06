import numpy as np
import os
import importlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sample_getter
import sys
import time
from configparser import NoOptionError
from markov_chain_diagnostics import get_sample_mean_and_error

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")
matplotlib.rcParams['mathtext.fontset'] = 'cm'

def main(config_file_string):
    initial_t = time.time()
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

    mean_sample = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                    temperature_index, number_of_particles, number_of_equilibration_iterations,
                    thinning_level=thinning_level)
    
    position_sample = sample_getter.get_positions(sample_directory, temperatures[temperature_index],
            temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
    print(np.shape(position_sample))

    fig, ax = plt.subplots(2,1, figsize = (15, 10), sharex = True)
    purple = True
    for i in range(10000, 10501, 500):
        
        if purple == True:
            colour = "#59029F"
        else:
            colour = "#e3a710"
        ax[1].plot(np.arange(0, number_of_particles), position_sample[i, :], color = colour)
        ax[1].set_title("ECMC Trajectories", fontsize = 20)
        ax[1].set_xlabel(r"$\tau$", fontsize = 40)
        ax[1].set_ylabel(r"$x$", fontsize = 40)
        ax[1].tick_params(axis='both', which='major', labelsize=20)
      
        purple = False
        # plt.savefig(f"ecmc_test_figs/{i}_positions.png")
        # plt.cla()

    sub_arr_len = 11000
    num_sub_arrs = 14
    position_sample = np.zeros((sub_arr_len, number_of_particles))

    position_sample[: sub_arr_len, :] = np.load(
        f"output/positions_001_hot_start/temperature_00_run_10_sample_of_positions.npy")[1:, :]



   
    purple = True
    for i in range(5000, 5501, 500):
        
        if purple == True:
            colour = "#59029F"
        else:
            colour = "#e3a710"
        ax[0].plot(np.arange(0, number_of_particles), position_sample[i, :], color = colour)
        ax[0].set_title("Metropolis Trajectories", fontsize = 20)
        ax[0].set_ylabel(r"$x$", fontsize = 40)
        ax[0].tick_params(axis='both', which='major', labelsize=20)

     
        purple = False
    
    plt.tight_layout()
    plt.savefig("trajectories.png")
    # ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=20)

    # ani.save(filename="positions_metropolis.mp4", writer="ffmpeg")

    final_t = time.time()

    print(f"took {(final_t - initial_t)/60} mins")

if __name__ == '__main__':
    main(sys.argv[1])