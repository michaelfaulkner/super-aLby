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
    m_arr = np.zeros(len(tau_values))

    for index, string in enumerate(tau_values):
        #config_file_string = os.path.join(config_folder, f"event_chain_{N_values[index]}.ini")
        config_file_string = config_folder
        config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
        (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
        number_of_observations, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)
        
        mass = parsing.get_value(config, strings.to_camel_case(potential), "mass")
        timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
        number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")
        #lambda_value = parsing.get_value(config, "EventChainMediator", "distance_between_measurements")
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
        analytical_x2_arr[index] = analytical_x2(mass, number_of_particles)
        
        #print(f"analytical x^2 for m={dimensionless_mass}: {analytical_x2_arr[index]}")
        N_arr[index] = number_of_particles
        m_arr[index] = mass
        ######################################
        #print(np.shape(position_sample[0]))
        fig, ax = plt.subplots(1,1)
        ax.scatter(np.arange(0,len(position_sample[0])), position_sample[0], label = "0")
        # ax.scatter(np.arange(0,len(position_sample[0])), position_sample[10], label = "10")
        # ax.scatter(np.arange(0,len(position_sample[0])), position_sample[50], label = "50")
        # ax.scatter(np.arange(0,len(position_sample[0])), position_sample[90], label = "90")
        ax.scatter(np.arange(0,len(position_sample[0])), position_sample[1000], label = "1000")
        ax.scatter(np.arange(0,len(position_sample[0])), position_sample[10000], label = "10000")
        # # # ax.scatter(np.arange(0,len(position_sample[0])), position_sample[30], label = "30")
        # # # ax.scatter(np.arange(0,len(position_sample[0])), position_sample[40], label = "40")
        ax.scatter(np.arange(0,len(position_sample[0])), position_sample[50000], label = "50000")
        # # # ax.scatter(np.arange(0,len(position_sample[0])), position_sample[60], label = "60")
        # # # ax.scatter(np.arange(0,len(position_sample[0])), position_sample[70], label = "70")
        # # # ax.scatter(np.arange(0,len(position_sample[0])), position_sample[80], label = "80")
        # # # ax.scatter(np.arange(0,len(position_sample[0])), position_sample[90], label = "90")
        ax.scatter(np.arange(0,len(position_sample[0])), position_sample[79000], label = "79000")
        ax.set_xlabel("site index")
        ax.set_ylabel("position")
        plt.legend()
        plt.savefig(f"output/figs/positions_resampled.png")

        fig, ax = plt.subplots(1,1)
        ax.scatter(np.arange(0,len(mean_sample)), mean_sample)
        ax.set_xlabel("simulation progress/time")
        ax.set_ylabel("<x^2>")
        plt.savefig(f"output/figs/mean_positions_squared_resampled.png")
        
        # indices_movement_sample = np.load("output/event_chain_mediator/temperature_00_sample_of_indices.npy")
        # indices_sample = indices_movement_sample[:,0]
        # moves_sample = indices_movement_sample[:,1]
        # eta_sample = indices_movement_sample[:,2]
        # x_sample = indices_movement_sample[:,3]
        # x_m1_sample = indices_movement_sample[:,4]
        # x_p1_sample = indices_movement_sample[:,5]
        # action_sample = indices_movement_sample[:,6]
        # # i = np.nonzero(position_sample==np.max(position_sample))
        # # print(np.shape(i))
        # # print(np.shape(indices_sample))
        # # indices_for_action = np.nonzero(indices_sample==i)[1]
        # # subset_action = action_sample[indices_for_action]

        # fig, ax = plt.subplots(1,1)
        # ax.plot(np.arange(0,len(indices_sample[:40])), indices_sample[:40], marker="x")
        # ax.set_xlabel("nth choice")
        # ax.set_ylabel("chosen active particle index")
        # plt.tight_layout()
        # plt.savefig(f"output/figs/active_particle.png")
        # fig, ax = plt.subplots(1,1)
        # ax.scatter(indices_sample, moves_sample, marker="x")
        # ax.set_xlabel("active_particle_index")
        # ax.set_ylabel("move made")
        # plt.tight_layout()
        # plt.savefig(f"output/figs/moves_at_a.png")
        # fig, ax = plt.subplots(1,1)
        # ax.plot(np.arange(0,len(moves_sample)), moves_sample, marker="x")
        # ax.set_xlabel("time in simulation (no units)")
        # ax.set_ylabel("move made")
        # plt.tight_layout()
        # plt.savefig(f"output/figs/moves_in_time.png")
        # fig, ax = plt.subplots(1,1)
        # ax.scatter(x_sample, eta_sample, marker="x")
        # ax.set_xlabel("x_a")
        # ax.set_ylabel("eta")
        # plt.tight_layout()
        # plt.savefig(f"output/figs/eta.png")
        # fig, ax = plt.subplots(1,1)
        # ax.scatter(np.arange(0,len(mean_sample)), mean_sample)
        # ax.set_xlabel("time in simulation")
        # ax.set_ylabel("<x^2>")
        # plt.tight_layout()
        # plt.savefig(f"output/figs/x2_mean.png")

        # ######################################



    # fig1, ax1 = plt.subplots(1,1)
    # ax1.scatter(timestep_arr, analytical_x2_arr, marker="x", color="red", label="analytical")
    # ax1.scatter(timestep_arr, numerical_x2, marker="x", color="blue", label="numerical")
    # ax1.set_xlabel(r"$\delta \tau$")
    # ax1.set_ylabel(r"$\langle x^2 \rangle$")
    # ax1.legend()
    # plt.tight_layout()
   

    # fig2, ax2 = plt.subplots(1,1)
    # ax2.scatter(N_arr, analytical_x2_arr, marker="x", color="red", label="analytical")
    # ax2.scatter(N_arr, numerical_x2, marker="x", color="blue", label="numerical")
    # ax2.set_xlabel("N")
    # ax2.set_ylabel(r"$\langle x^2 \rangle$")
    # ax2.legend()
    # plt.tight_layout()
    # #plt.show()
    # print(analytical_x2_arr)
    # fig1, ax1 = plt.subplots(1,1)
    # ax1.scatter(m_arr[:3], analytical_x2_arr[:3], marker="x", color="red", label="analytical")
    # ax1.scatter(m_arr[:3], numerical_x2[:3], marker="x", color="blue", label="numerical")
    # ax1.set_xlabel(r"$\tilde{m}$")
    # ax1.set_ylabel(r"$\langle x^2 \rangle$")
    # plt.ticklabel_format(style="plain")
    # ax1.legend()
    
    # plt.tight_layout()
    # plt.savefig("m_arr.png")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])