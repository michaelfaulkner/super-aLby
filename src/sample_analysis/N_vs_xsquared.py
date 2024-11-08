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

    # read in the k values from k_values.txt
    current_directory = os.path.dirname(__file__)
    k_values_filepath = os.path.join(os.path.split(current_directory)[0], "N_values.txt")
    k_data = np.loadtxt(k_values_filepath, dtype='str')
    k_values = k_data[:,1]

    analytical_x2_arr = np.zeros(len(k_values)+3)
    numerical_x2 = np.zeros(len(k_values)+3)
    N_arr = np.zeros(len(k_values)+3)
    N_obsv = np.zeros(3)

    for index, string in enumerate(k_values):
        config_file_string = f"src/config_files/convergence_tests/one_dim_quantum_oscillator_potential/metropolis_01_{string}.ini"
        config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
        (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
        number_of_observations, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)
        
        dimensionless_mass = parsing.get_value(config, strings.to_camel_case(potential), "dimensionless_mass")
        number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")

        mass_as_string = str(dimensionless_mass)
        mass_string_split = mass_as_string.split(".")
        mass_as_string = ""
        for _ in mass_string_split:
            mass_as_string += _
        
        sample_directory = f"output/N/metropolis_01_{string}"
        temperature_index = 0

        analytical_x2_arr[index] = analytical_x2(dimensionless_mass, number_of_particles)
        N_arr[index] = number_of_particles
        if number_of_particles == 50:
            N_obsv[0] = number_of_observations

    config_file_string = f"src/config_files/convergence_tests/one_dim_quantum_oscillator_potential/metropolis_1_01.ini"
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
    number_of_observations, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)
    
    dimensionless_mass = parsing.get_value(config, strings.to_camel_case(potential), "dimensionless_mass")
    number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")

    
    sample_directory = f"output/timestep/metropolis_1_01"
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
    numerical_x2[-1] = mean_sample_mean[0]
    analytical_x2_arr[-1] = analytical_x2(dimensionless_mass, number_of_particles)
    N_arr[-1] = number_of_particles

    config_file_string = f"src/config_files/convergence_tests/one_dim_quantum_oscillator_potential/metropolis_01_50_50kruns.ini"
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
    number_of_observations, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)
    
    dimensionless_mass = parsing.get_value(config, strings.to_camel_case(potential), "dimensionless_mass")
    number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")

    
    sample_directory = f"output/metropolis_01_50_50kruns"
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
    numerical_x2[-2] = mean_sample_mean[0]
    analytical_x2_arr[-2] = analytical_x2(dimensionless_mass, number_of_particles)
    N_arr[-2] = number_of_particles
    N_obsv[1] = number_of_observations

    config_file_string = f"src/config_files/convergence_tests/one_dim_quantum_oscillator_potential/metropolis_01_50_100kruns.ini"
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
    number_of_observations, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)

    dimensionless_mass = parsing.get_value(config, strings.to_camel_case(potential), "dimensionless_mass")
    number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")

    sample_directory = f"output/metropolis_01_50"
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
    numerical_x2[-3] = mean_sample_mean[0]
    analytical_x2_arr[-3] = analytical_x2(dimensionless_mass, number_of_particles)
    N_arr[-3] = number_of_particles
    N_obsv[2] = number_of_observations

        
    numerical0_loaded = np.load("output/N/x2_run0.npy")
    numerical1_loaded = np.load("output/N/x2_run1.npy")
    numerical2_loaded = np.load("output/N/x2_run2.npy")

    mean_numerical = np.mean(([numerical0_loaded],
                              [numerical1_loaded],
                              [numerical2_loaded]), axis=0)
    numerical_x2[:-3] = mean_numerical
    

    fig2, ax2 = plt.subplots(1,1)
    ax2.scatter(N_arr, analytical_x2_arr, marker="x", color="red", label="analytical")
    ax2.scatter(N_arr, numerical_x2, marker="x", color="blue", label="numerical")
    ax2.set_xlabel("N")
    ax2.set_ylabel("<x^2> - dimensionless")
    ax2.legend()

    n_runs = [numerical_x2[-3], numerical_x2[-2], numerical_x2[8]]
    fig, ax = plt.subplots(1,1)
    ax.scatter(N_obsv, n_runs)
    ax.hlines(analytical_x2_arr[8], color = "red", xmin=np.min(N_obsv), xmax = np.max(N_obsv))
    plt.tight_layout()

    plt.show()
    


if __name__ == '__main__':
    main()