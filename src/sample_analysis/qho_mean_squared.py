import numpy as np
import os
import importlib
import matplotlib
import matplotlib.pyplot as plt
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

def main(config_file_string, data_path):
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
    number_of_observations, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)
    
    mass = parsing.get_value(config, strings.to_camel_case(potential), "mass")
    timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
    number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")
    
    sample_directory = sample_directories[0]
    sample_directory = data_path
    temperature_index = 0
    thinning_level = None
    number_of_equilibration_iterations = None
    mean_sample = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                        temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)

    timestep_str = str(timestep).replace(".", "")
    # print(timestep_str)
    
    # sub_arr_len = 51000
    # num_sub_arrs = 20
    # mean_sample_m = np.zeros(sub_arr_len * num_sub_arrs)
    # for i in range(num_sub_arrs):
    #     mean_sample_m[i * sub_arr_len : (i+1) * sub_arr_len] = np.load(
    #     f"output/metropolis_001_checkpoints/temperature_00_run_{i:02d}_sample_of_mean_positions.npy")[1:, 0]

    # fig, ax = plt.subplots(2, 1, sharex=True)
    # ax[0].plot(np.arange(np.shape(mean_sample_m)[0]), mean_sample_m, color = "purple")
    # ax[1].plot(np.arange(np.shape(mean_sample)[0]), mean_sample, color="#d11596")

    # ax[0].set_title(rf"Metropolis, $\delta \tau = 0.01$", fontsize=12)
    # ax[1].set_title(rf"ECMC, $\lambda = 50.0$, $\delta \tau = 0.01$", fontsize=12)
    
    # ax[1].set_xlabel("sample index", fontsize=15)
    # ax[0].set_ylabel(r"$\langle x^2 \rangle$",  fontsize=15)
    # ax[1].set_ylabel(r"$\langle x^2 \rangle$",  fontsize=15)


    # #plt.ylim((-1, 0.75))
    # #plt.xlim((-1, 0.1e6))

    # #plt.savefig(f"{timestep_str}_metropolis_w_positions.png")
    # plt.xlim((-1000, 50000))
    # plt.tight_layout()
    # plt.savefig(f"trace.png", transparent = True)
    print(len(mean_sample))
    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(np.shape(mean_sample)[0]), mean_sample, color="purple")
    ax.set_xlim(0, 50000)
    # ax.set_xlabel("sample index", fontsize=15)
    # ax.set_ylabel(r"$\langle x^2 \rangle$",  fontsize=15)
    # ax.set_title(rf"Metropolis, $\delta \tau = 0.01$", fontsize=15)
    #plt.xlim((-1000, 10000))
    plt.savefig("test.png")
    
    # plt.savefig("zoomed_metropolis_10e6_mean_squared_positions.png")
    #print(f"mean: {mean}, std: {std}")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])