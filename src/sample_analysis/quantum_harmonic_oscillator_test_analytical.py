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

def analytical_x2(dim_m, N_tau):
    dim_omega = dim_m
    auxilliary = 1 + dim_omega**2 / 2 - dim_omega * np.sqrt(1 + dim_omega**2 / 4)

    return (1 / (2 * dim_m * dim_omega * np.sqrt(1 + 0.25 * dim_omega**2))) * ((1 + auxilliary**N_tau) / (1 - auxilliary**N_tau))


def main(values_filepath, config_folder):
    r"""
    Produces plots comparing the numerical and analytical values of <x^2> for the 1D quantum harmonic oscillator potential.

    Parameters
        ----------
        values_filepath : str
            The path to the N and \delta \tau values file. This file is expected to be a .txt file, formatted like:
            1 001 10000 
            2 005 2000 
            etc. This is an artefact of using SLURM array jobs to produce several simulations with differing input values.
        config_folder : str
            The path to the folder containing the corresponding configuration files. These are expected to be named like
            metropolis_001_10000.ini

    """
    N_tau_data = np.loadtxt(values_filepath, dtype='str')
    tau_values = N_tau_data[:,1]
    N_values = N_tau_data[:,2]

    analytical_x2_arr = np.zeros(len(tau_values))
    numerical_x2 = np.zeros(len(tau_values))
    timestep_arr = np.zeros(len(tau_values))
    N_arr = np.zeros(len(tau_values))

    for index, string in enumerate(tau_values):
        #config_file_string = os.path.join(config_folder, f"metropolis_{string}_{N_values[index]}.ini")
        config_file_string = config_folder
        config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
        (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
        number_of_observations, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)
        
        dimensionless_mass = parsing.get_value(config, strings.to_camel_case(potential), "dimensionless_mass")
        timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
        number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")
        
        sample_directory = sample_directories[0]
        temperature_index = 0
        thinning_level = None

        mean_sample = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                            temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
        position_sample = sample_getter.get_positions(sample_directory, temperatures[temperature_index],
                            temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
        mean_sample_mean = get_sample_mean_and_error(mean_sample)
        numerical_x2[index] = mean_sample_mean[0]
        timestep_arr[index] = timestep
        analytical_x2_arr[index] = analytical_x2(dimensionless_mass, number_of_particles)
        N_arr[index] = number_of_particles
        ######################################
        #print(np.shape(position_sample[0]))
        fig, ax = plt.subplots(1,1)
        ax.scatter(np.arange(0,len(position_sample[0])), position_sample[0])
        ax.scatter(np.arange(0,len(position_sample[0])), position_sample[10])
        ax.scatter(np.arange(0,len(position_sample[0])), position_sample[20])
        ax.scatter(np.arange(0,len(position_sample[0])), position_sample[30])
        ax.scatter(np.arange(0,len(position_sample[0])), position_sample[40])
        ax.scatter(np.arange(0,len(position_sample[0])), position_sample[50])
        ax.scatter(np.arange(0,len(position_sample[0])), position_sample[60])
        ax.scatter(np.arange(0,len(position_sample[0])), position_sample[70])
        ax.scatter(np.arange(0,len(position_sample[0])), position_sample[80])
        ax.scatter(np.arange(0,len(position_sample[0])), position_sample[90])
        ax.scatter(np.arange(0,len(position_sample[0])), position_sample[99])
        ax.set_xlabel("site index")
        ax.set_ylabel("position")
        
        # ax.set_ylim(-2.5,2.5)
        plt.savefig("test.png")
        
        indices_movement_sample = np.load("output/event_chain_mediator/temperature_00_sample_of_indices.npy")
        indices_sample = indices_movement_sample[:,0]
        moves_sample = indices_movement_sample[:,1]
        fig, ax = plt.subplots(1,1)
        ax.plot(np.arange(0,len(indices_sample)), indices_sample, marker="x")
        ax.set_xlabel("nth choice")
        ax.set_ylabel("chosen active particle index")
        plt.savefig("test1.png")
        fig, ax = plt.subplots(1,1)
        ax.scatter(indices_sample, moves_sample, marker="x")
        ax.set_xlabel("index")
        ax.set_ylabel("move made")
        plt.savefig("test2.png")
        ######################################



    fig1, ax1 = plt.subplots(1,1)
    ax1.scatter(timestep_arr, analytical_x2_arr, marker="x", color="red", label="analytical")
    ax1.scatter(timestep_arr, numerical_x2, marker="x", color="blue", label="numerical")
    ax1.set_xlabel(r"$\delta \tau$")
    ax1.set_ylabel(r"$\langle x^2 \rangle$")
    ax1.legend()
    plt.tight_layout()

    fig2, ax2 = plt.subplots(1,1)
    ax2.scatter(N_arr, analytical_x2_arr, marker="x", color="red", label="analytical")
    ax2.scatter(N_arr, numerical_x2, marker="x", color="blue", label="numerical")
    ax2.set_xlabel("N")
    ax2.set_ylabel(r"$\langle x^2 \rangle$")
    ax2.legend()
    plt.tight_layout()
    #plt.show()
    

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])