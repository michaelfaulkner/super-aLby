import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import sample_getter
import sys
from markov_chain_diagnostics import get_sample_mean_and_error


this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")


def analytical_x2(dim_m, N_tau):
    dim_omega = dim_m
    auxiliary = 1 + dim_omega ** 2 / 2 - dim_omega * np.sqrt(1 + dim_omega**2 / 4)
    return (1 / (2 * dim_m * dim_omega * np.sqrt(1 + 0.25 * dim_omega ** 2))) * (
            (1 + auxiliary ** N_tau) / (1 - auxiliary ** N_tau))


def main(values_filepath, config_folder):
    r"""
    Produces plots comparing the numerical and analytical values of <x^2> for the 1D quantum harmonic oscillator
        potential.

    Parameters
        ----------
        values_filepath : str
            The path to the N and \delta \tau values file. This file is expected to be a .txt file, formatted like:
            1 001 10000 
            2 005 2000 
            etc.
            This is an artefact of using SLURM array jobs to produce several simulations with differing input values.
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
        if string != "001":
       
            config_file_string = os.path.join(config_folder, f"metropolis/metropolis_{string}.ini")
            print(config_file_string)
            config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
            (config_file_mediator, potential, samplers, sample_directories, temperatures,
             number_of_equilibration_iterations, number_of_observations, number_of_particles,
             _, _, _) = helper_methods.get_basic_config_data(config_file_string)
            
            mass = parsing.get_value(config, strings.to_camel_case(potential), "mass")
            timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
            number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")
            # lambda_value = parsing.get_value(config, "EventChainMediator", "normalised_distance_between_measurements")
            sample_directory = sample_directories[0]
            temperature_index = 0
            thinning_level = None
            mean_sample = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                                                           temperature_index, number_of_particles,
                                                           number_of_equilibration_iterations, thinning_level)
            # position_sample = sample_getter.get_positions(sample_directory, temperatures[temperature_index],
            #                                               temperature_index, number_of_particles,
            #                                               number_of_equilibration_iterations, thinning_level)
            mean_sample_mean = get_sample_mean_and_error(mean_sample)
            numerical_x2[index] = mean_sample_mean[0] / timestep**2
   
        timestep_arr[index] = timestep
        # print(f"timestep was {timestep_arr[index]}, mean {numerical_x2[index]}")

    numerical_x2_e = np.zeros(len(tau_values))
    for index, string in enumerate(tau_values):

        config_file_string = os.path.join(config_folder, f"ecmc_lambda_50/event_chain_{string}.ini")
        print(config_file_string)
        config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
        (config_file_mediator, potential, samplers, sample_directories, temperatures,
         number_of_equilibration_iterations, number_of_observations, number_of_particles, _, _, _
         ) = helper_methods.get_basic_config_data(config_file_string)
        
        mass = parsing.get_value(config, strings.to_camel_case(potential), "mass")
        timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
        number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")
        # lambda_value = parsing.get_value(config, "EventChainMediator", "normalised_distance_between_measurements")
        sample_directory = sample_directories[0]
        temperature_index = 0
        thinning_level = None
 
        mean_sample = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                                                       temperature_index, number_of_particles,
                                                       number_of_equilibration_iterations, thinning_level)
        mean_sample = mean_sample[:30000]

        mean_sample_mean = get_sample_mean_and_error(mean_sample)
        numerical_x2_e[index] = mean_sample_mean[0] / timestep ** 2
        timestep_arr[index] = timestep
        analytical_x2_arr[index] = analytical_x2(mass * timestep, number_of_particles)
        
    timestep = 0.01
    number_of_equilibration_iterations = 1000
    metropolis_001_31k = np.load(
        "output/metropolis/mean_squared_positions/31000/temperature_00_sample_of_mean_squared_positions_001_0.npy")
    metropolis_001_31k = get_sample_mean_and_error(metropolis_001_31k)
    metropolis_001_31k = metropolis_001_31k[0] / timestep **2

    metropolis_001_51k = np.load(
        "output/metropolis/mean_squared_positions/51000/temperature_00_sample_of_mean_squared_positions_001_0.npy")
    metropolis_001_51k = metropolis_001_51k[number_of_equilibration_iterations + 1:]
    metropolis_001_51k = get_sample_mean_and_error(metropolis_001_51k)
    metropolis_001_51k = metropolis_001_51k[0] / timestep**2

    metropolis_001_81k = np.load(
        "output/metropolis/mean_squared_positions/81000/temperature_00_sample_of_mean_squared_positions_001_0.npy")
    metropolis_001_81k = get_sample_mean_and_error(metropolis_001_81k)
    metropolis_001_81k = metropolis_001_81k[0] / timestep**2

    metropolis_001_101k = np.load(
        "output/metropolis/mean_squared_positions/101000/temperature_00_sample_of_mean_squared_positions_001_0.npy")
    metropolis_001_101k = get_sample_mean_and_error(metropolis_001_101k)
    metropolis_001_101k = metropolis_001_101k[0] / timestep**2

    sub_arr_len = 51000
    num_sub_arrs = 20
    metropolis_001_10e6 = np.zeros(sub_arr_len * num_sub_arrs)
    for i in range(num_sub_arrs):
        metropolis_001_10e6[i * sub_arr_len : (i+1) * sub_arr_len] = np.load(
        f"output/metropolis_001_checkpoints/temperature_00_run_{i:02d}_sample_of_mean_positions.npy")[1:, 0]

    metropolis_001_10e6 = metropolis_001_10e6[number_of_equilibration_iterations + 1:-18999]
    metropolis_001_10e6 = get_sample_mean_and_error(metropolis_001_10e6)
    metropolis_001_10e6 = metropolis_001_10e6[0] / timestep**2

    fig1, ax1 = plt.subplots(1,2, sharey = True, figsize = (10.0, 7.0))
    ax1[0].set_title(r"Metropolis with $3\times 10^4$ samples",  fontsize=15)
    ax1[0].scatter(timestep_arr[:], analytical_x2_arr[:], marker=".", s = 200.0, color="purple", label="analytical")
    ax1[0].scatter(timestep_arr[:], numerical_x2[:], marker="x", s = 200.0, color="#f974ef", label="numerical")

    ax1[0].scatter(timestep, metropolis_001_31k,  marker="x", s = 200.0, color="#f974ef", label=r"$3\times 10^4$ ")
    ax1[0].scatter(timestep, metropolis_001_51k,  marker="v", s = 200.0, color="#bd178b", label=r"$5\times 10^4$ ")
    ax1[0].scatter(timestep, metropolis_001_81k,  marker="s", s = 200.0, color="#eb102e", label=r"$8\times 10^4$ ")
    ax1[0].scatter(timestep, metropolis_001_101k,  marker="p", s = 200.0, color="#f0601d", label=r"$1\times 10^5$ ")
    ax1[0].scatter(timestep, metropolis_001_10e6,  marker="*", s = 200.0, color="#f5d20f", label=r"$1\times 10^6$ ")

    ax1[1].set_title(r"ECMC, with $\lambda = 50.0$ and $3\times 10^4$  samples",  fontsize=15)
    ax1[1].scatter(timestep_arr[:], analytical_x2_arr[:], marker=".", s = 200.0, color="purple", label="analytical")
    ax1[1].scatter(timestep_arr[:], numerical_x2_e[:], marker="x", s = 200.0, color="#f974ef", label="numerical")
    ax1[0].set_yscale('log')
    ax1[1].set_yscale('log')
    ax1[0].set_xscale('log')
    ax1[1].set_xscale('log')

    ax1[0].set_xlabel(r"$\delta \tau$",  fontsize=20)
    ax1[1].set_xlabel(r"$\delta \tau$",  fontsize=20)

    ax1[0].set_ylabel(r"$\langle x^2 \rangle$",  fontsize=20)
    ax1[0].legend()
    ax1[1].legend()

    plt.tight_layout()
    plt.savefig("tau_arr.pdf")
  
    print(f"analytical 0.01: {analytical_x2_arr[-1]}, ecmc: {numerical_x2_e[-1]}")
   

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
