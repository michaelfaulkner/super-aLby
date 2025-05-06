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
    
    sub_arr_len = 11000
    num_sub_arrs = 14
    mean_sample = np.zeros(sub_arr_len * num_sub_arrs)

    position_sample_0 = np.zeros((sub_arr_len, number_of_particles))
    position_sample_1 = np.zeros((sub_arr_len, number_of_particles))
    position_sample_2 = np.zeros((sub_arr_len, number_of_particles))
    position_sample_3 = np.zeros((sub_arr_len, number_of_particles))
    position_sample_4 = np.zeros((sub_arr_len, number_of_particles))
    position_sample_5 = np.zeros((sub_arr_len, number_of_particles))
    position_sample_6 = np.zeros((sub_arr_len, number_of_particles))
    position_sample_7 = np.zeros((sub_arr_len, number_of_particles))
    position_sample_8 = np.zeros((sub_arr_len, number_of_particles))
    position_sample_9 = np.zeros((sub_arr_len, number_of_particles))
    position_sample_10 = np.zeros((sub_arr_len, number_of_particles))
    position_sample_11 = np.zeros((sub_arr_len, number_of_particles))
    position_sample_12 = np.zeros((sub_arr_len, number_of_particles))
    position_sample_13 = np.zeros((sub_arr_len, number_of_particles))
    

    # for i in range(num_sub_arrs):
    #     print(f"sample file {i}")
    #     mean_sample[i * sub_arr_len : (i+1) * sub_arr_len] = np.load(
    #     f"output/positions_001_hot_start/temperature_00_run_{i:02d}_sample_of_mean_positions.npy")[1:, 0]

    #     position_sample_0[i * sub_arr_len : (i+1) * sub_arr_len, :] = np.load(
    #     f"output/positions_001_hot_start/temperature_00_run_{i:02d}_sample_of_positions.npy")[1:, :]
    
    # for i in range((i+1) * sub_arr_len, num_sub_arrs):
    #     print(f"positions_1, sample file {i}")
    #     mean_sample[i * sub_arr_len : (i+1) * sub_arr_len] = np.load(
    #     f"output/positions_001_hot_start/temperature_00_run_{i:02d}_sample_of_mean_positions.npy")[1:, 0]

    #     position_sample_1[(i- int(num_sub_arrs/divisor)) * sub_arr_len  : (i+1- int(num_sub_arrs/divisor)) * sub_arr_len, :] = np.load(
    #     f"output/positions_001_hot_start/temperature_00_run_{i:02d}_sample_of_positions.npy")[1:, :]
    #     print(f"sample file {i}, indices {(i- int(num_sub_arrs/divisor)) * sub_arr_len} : {(i+1- int(num_sub_arrs/divisor)) * sub_arr_len}")
    # position_sample_0[: sub_arr_len, :] = np.load(
    #     f"output/positions_001_hot_start/temperature_00_run_00_sample_of_positions.npy")[1:, :]
    # position_sample_1[: sub_arr_len, :] = np.load(
    #     f"output/positions_001_hot_start/temperature_00_run_01_sample_of_positions.npy")[1:, :]
    # position_sample_2[: sub_arr_len, :] = np.load(
    #     f"output/positions_001_hot_start/temperature_00_run_02_sample_of_positions.npy")[1:, :]
    # position_sample_3[: sub_arr_len, :] = np.load(
    #     f"output/positions_001_hot_start/temperature_00_run_03_sample_of_positions.npy")[1:, :]
    # position_sample_4[: sub_arr_len, :] = np.load(
    #     f"output/positions_001_hot_start/temperature_00_run_04_sample_of_positions.npy")[1:, :]
    # position_sample_5[: sub_arr_len, :] = np.load(
    #     f"output/positions_001_hot_start/temperature_00_run_05_sample_of_positions.npy")[1:, :]
    # position_sample_6[: sub_arr_len, :] = np.load(
    #     f"output/positions_001_hot_start/temperature_00_run_06_sample_of_positions.npy")[1:, :]
    # position_sample_7[: sub_arr_len, :] = np.load(
    #     f"output/positions_001_hot_start/temperature_00_run_07_sample_of_positions.npy")[1:, :]
    # position_sample_8[: sub_arr_len, :] = np.load(
    #     f"output/positions_001_hot_start/temperature_00_run_08_sample_of_positions.npy")[1:, :]
    # position_sample_9[: sub_arr_len, :] = np.load(
    #     f"output/positions_001_hot_start/temperature_00_run_09_sample_of_positions.npy")[1:, :]
    # position_sample_10[: sub_arr_len, :] = np.load(
    #     f"output/positions_001_hot_start/temperature_00_run_10_sample_of_positions.npy")[1:, :]
    # position_sample_11[: sub_arr_len, :] = np.load(
    #     f"output/positions_001_hot_start/temperature_00_run_11_sample_of_positions.npy")[1:, :]
    position_sample_12[: sub_arr_len, :] = np.load(
        f"output/positions_001_hot_start/temperature_00_run_12_sample_of_positions.npy")[1:, :]
    position_sample_13[: sub_arr_len, :] = np.load(
        f"output/positions_001_hot_start/temperature_00_run_13_sample_of_positions.npy")[1:, :]
    


    fig, ax = plt.subplots(1, 1, figsize = (15, 10))
    ax.set_ylim(-3,3)
    artists =[]

    for i in range(132000, 154000):

        if i % 1000 == 0:
            print(f"getting plot number {i}")
        # if i < 11000:
        #     container = ax.scatter(np.arange(0, number_of_particles), position_sample_0[i, :], color = "purple")
        # else: 
        #     container = ax.scatter(np.arange(0, number_of_particles), position_sample_1[i - sub_arr_len, :], color = "purple")
        # artists.append([container])
        if i < 11000:
            ax.set_ylim(-4,4)
            ax.scatter(np.arange(0, number_of_particles), position_sample_0[i, :], color = "purple")
            plt.savefig(f"test_figs/{i}_positions.png")
            plt.cla()
           
        elif i < 22000:
            ax.set_ylim(-4,4)
            ax.scatter(np.arange(0, number_of_particles), position_sample_1[i - sub_arr_len, :], color = "purple")
            plt.savefig(f"test_figs/{i}_positions.png")
            plt.cla()
        
        elif i < 33000:
            ax.set_ylim(-4,4)
            ax.scatter(np.arange(0, number_of_particles), position_sample_2[i - 2 *sub_arr_len, :], color = "purple")
            plt.savefig(f"test_figs/{i}_positions.png")
            plt.cla()

        elif i < 44000:
            ax.set_ylim(-4,4)
            ax.scatter(np.arange(0, number_of_particles), position_sample_3[i - 3 *sub_arr_len, :], color = "purple")
            plt.savefig(f"test_figs/{i}_positions.png")
            plt.cla()

        elif i < 55000:
            ax.set_ylim(-4,4)
            ax.scatter(np.arange(0, number_of_particles), position_sample_4[i - 4 * sub_arr_len, :], color = "purple")
            plt.savefig(f"test_figs/{i}_positions.png")
            plt.cla()

        elif i < 66000:
            ax.set_ylim(-4,4)
            ax.scatter(np.arange(0, number_of_particles), position_sample_5[i - 5 * sub_arr_len, :], color = "purple")
            plt.savefig(f"test_figs/{i}_positions.png")
            plt.cla()
        
        elif i < 77000:
            ax.set_ylim(-4,4)
            ax.scatter(np.arange(0, number_of_particles), position_sample_6[i - 6 * sub_arr_len, :], color = "purple")
            plt.savefig(f"test_figs/{i}_positions.png")
            plt.cla()
        
        elif i < 88000:
            ax.set_ylim(-4,4)
            ax.scatter(np.arange(0, number_of_particles), position_sample_7[i - 7 * sub_arr_len, :], color = "purple")
            plt.savefig(f"test_figs/{i}_positions.png")
            plt.cla()
        
        elif i < 99000:
            ax.set_ylim(-4,4)
            ax.scatter(np.arange(0, number_of_particles), position_sample_8[i - 8 * sub_arr_len, :], color = "purple")
            plt.savefig(f"test_figs/{i}_positions.png")
            plt.cla()
        
        elif i < 110000:
            ax.set_ylim(-4,4)
            ax.scatter(np.arange(0, number_of_particles), position_sample_9[i - 9 * sub_arr_len, :], color = "purple")
            plt.savefig(f"test_figs/{i}_positions.png")
            plt.cla()
        
        elif i < 121000:
            ax.set_ylim(-4,4)
            ax.scatter(np.arange(0, number_of_particles), position_sample_10[i - 10 * sub_arr_len, :], color = "purple")
            plt.savefig(f"test_figs/{i}_positions.png")
            plt.cla()
        
        elif i < 132000:
            ax.set_ylim(-4,4)
            ax.scatter(np.arange(0, number_of_particles), position_sample_11[i - 11 * sub_arr_len, :], color = "purple")
            plt.savefig(f"test_figs/{i}_positions.png")
            plt.cla()
        
        elif i < 143000:
            ax.set_ylim(-4,4)
            ax.scatter(np.arange(0, number_of_particles), position_sample_12[i - 12 * sub_arr_len, :], color = "purple")
            plt.savefig(f"test_figs/{i}_positions.png")
            plt.cla()
        
        else:
            ax.set_ylim(-4,4)
            ax.scatter(np.arange(0, number_of_particles), position_sample_13[i - 13 * sub_arr_len, :], color = "purple")
            plt.savefig(f"test_figs/{i}_positions.png")
            plt.cla()







    # ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=20)

    # ani.save(filename="positions_metropolis.mp4", writer="ffmpeg")

    final_t = time.time()

    print(f"took {(final_t - initial_t)/60} mins")

if __name__ == '__main__':
    main(sys.argv[1])