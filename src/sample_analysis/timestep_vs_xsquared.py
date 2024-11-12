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


def main():

    # read in the N and tau values
    current_directory = os.path.dirname(__file__)
    values_filepath = os.path.join(os.path.split(current_directory)[0], "N_tau_values_50.txt")
    N_tau_data = np.loadtxt(values_filepath, dtype='str')
    tau_values = N_tau_data[:,1]
    N_values = N_tau_data[:,2]

    analytical_x2_arr = np.zeros(len(tau_values))
    numerical_x2 = np.zeros(len(tau_values))
    timestep_arr = np.zeros(len(tau_values))
    m_arr = np.zeros(len(tau_values))
    N_arr = np.zeros(len(tau_values))

    for index, string in enumerate(tau_values):
        config_file_string = f"src/config_files/convergence_tests/one_dim_quantum_oscillator_potential/N_tau_50/metropolis_{string}_{N_values[index]}.ini"
        #config_file_string = f"src/config_files/convergence_tests/one_dim_quantum_oscillator_potential/metropolis_1_{string}.ini"
        config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
        (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
        number_of_observations, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)
        
        dimensionless_mass = parsing.get_value(config, strings.to_camel_case(potential), "dimensionless_mass")
        timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
        number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")
        
        sample_directory = f"output/convergence_tests/one_dim_quantum_oscillator_potential/N_tau_50/metropolis_{string}_{N_values[index]}"
        #sample_directory =  f"output/timestep/metropolis_1_{string}"
        temperature_index = 0
        if number_of_observations < 10000:
            thinning_level = 1
        else:
            thinning_level = 10
        mean_sample = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                            temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
        if len(mean_sample) != int(number_of_observations/thinning_level):
            mean_sample = mean_sample[int(number_of_equilibration_iterations/thinning_level):]
        
        mean_sample_mean = get_sample_mean_and_error(mean_sample)
        numerical_x2[index] = mean_sample_mean[0]
        timestep_arr[index] = timestep
        m_arr[index] = dimensionless_mass / timestep
        analytical_x2_arr[index] = analytical_x2(dimensionless_mass, number_of_particles)
        N_arr[index] = number_of_particles



    fig1, ax1 = plt.subplots(1,1)
    ax1.scatter(timestep_arr, analytical_x2_arr, marker="x", color="red", label="analytical")
    ax1.scatter(timestep_arr, numerical_x2, marker="x", color="blue", label="numerical")
    ax1.set_xlabel("timestep")
    ax1.set_ylabel("<x^2> - dimensionless")
    ax1.legend()
    plt.tight_layout()
    plt.savefig("output/figs/x2_timestep_prod50.pdf")
    plt.savefig("output/figs/x2_timestep_prod50.png")

    fig2, ax2 = plt.subplots(1,1)
    ax2.scatter(N_arr, analytical_x2_arr, marker="x", color="red", label="analytical")
    ax2.scatter(N_arr, numerical_x2, marker="x", color="blue", label="numerical")
    ax2.set_xlabel("N")
    ax2.set_ylabel("<x^2> - dimensionless")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("output/figs/N_timestep_prod50.pdf")
    plt.savefig("output/figs/N_timestep_prod50.png")
    plt.show()
    


if __name__ == '__main__':
    main()