import numpy as np
import os
import importlib
import matplotlib
import matplotlib.pyplot as plt
import sample_getter
import sys
from configparser import NoOptionError
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


def main(config_file_string):
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
    number_of_observations, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)
    
    dimensionless_mass = parsing.get_value(config, strings.to_camel_case(potential), "dimensionless_mass")
    timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
    number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")

    #sample_directory = sample_directories[0]
    sample_directory = "output/convergence_tests/one_dim_quantum_oscillator_potential/thermalisation/metropolis_01_50_10k_0"
    temperature_index = 0
    thinning_level = None
    number_of_equilibration_iterations = None
    mean_sample_10_0 = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                    temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
    
    sample_directory = "output/convergence_tests/one_dim_quantum_oscillator_potential/thermalisation/metropolis_01_50_10k_1"
    mean_sample_10_1 = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                    temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
    
    sample_directory = "output/convergence_tests/one_dim_quantum_oscillator_potential/thermalisation/metropolis_01_50_10k_2"
    mean_sample_10_2 = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                    temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
    mean_sample_10 = np.mean((mean_sample_10_0,mean_sample_10_1,mean_sample_10_2), axis=0)

    config_file_string = "src/config_files/convergence_tests/one_dim_quantum_oscillator_potential/thermalisation/metropolis_01_50_50kruns.ini"
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
    number_of_observations, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)
    
    dimensionless_mass = parsing.get_value(config, strings.to_camel_case(potential), "dimensionless_mass")
    timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
    number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")

    sample_directory = "output/convergence_tests/one_dim_quantum_oscillator_potential/thermalisation/metropolis_01_50_50k_0"
    mean_sample_50_0 = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                    temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
    
    sample_directory = "output/convergence_tests/one_dim_quantum_oscillator_potential/thermalisation/metropolis_01_50_50k_1"
    mean_sample_50_1 = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                    temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
    
    sample_directory = "output/convergence_tests/one_dim_quantum_oscillator_potential/thermalisation/metropolis_01_50_50k_2"
    mean_sample_50_2 = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                    temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
    mean_sample_50 = np.mean((mean_sample_50_0,mean_sample_50_1,mean_sample_50_2), axis=0)
    
    config_file_string = "src/config_files/convergence_tests/one_dim_quantum_oscillator_potential/thermalisation/metropolis_01_50_100kruns.ini"
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
    number_of_observations, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)
    
    dimensionless_mass = parsing.get_value(config, strings.to_camel_case(potential), "dimensionless_mass")
    timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
    number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")

    sample_directory = "output/convergence_tests/one_dim_quantum_oscillator_potential/thermalisation/metropolis_01_50_100k_0"
    mean_sample_100_0 = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                    temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
    
    sample_directory = "output/convergence_tests/one_dim_quantum_oscillator_potential/thermalisation/metropolis_01_50_100k_1"
    mean_sample_100_1 = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                    temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
    
    sample_directory = "output/convergence_tests/one_dim_quantum_oscillator_potential/thermalisation/metropolis_01_50_100k_2"
    mean_sample_100_2 = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                    temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
    mean_sample_100 = np.mean((mean_sample_100_0,mean_sample_100_1,mean_sample_100_2), axis=0)

    print(np.shape(mean_sample_10))
    print(np.shape(mean_sample_50))
    print(np.shape(mean_sample_100))

    # if len(mean_sample) != int(number_of_observations/thinning_level):
    #     mean_sample = mean_sample[int(number_of_equilibration_iterations/thinning_level):]
    
    analytical = analytical_x2(dimensionless_mass, number_of_particles)

    sweeps = np.arange(0,len(mean_sample_10))
    fig, ax = plt.subplots(1,3, sharey=True)
    ax[0].scatter(sweeps[:5000], mean_sample_10[:5000])
    ax[1].scatter(sweeps[:5000], mean_sample_50[:5000])
    ax[2].scatter(sweeps[:5000], mean_sample_100[:5000])
    ax[0].set_xlabel("metropolis sweeps")
    ax[1].set_xlabel("metropolis sweeps")
    ax[2].set_xlabel("metropolis sweeps")
    ax[0].set_ylabel("<x^2>")
    fig.set_size_inches(12.5, 8.5)
    plt.tight_layout()
    plt.savefig(f"output/figs/trace_plots/x2_sweeps_avgd.png")

    mean_x2 = [np.mean(mean_sample_10), np.mean(mean_sample_50), np.mean(mean_sample_100)]
    fig1, ax1 = plt.subplots(1,1)
    ax1.scatter([10000, 50000, 100000], mean_x2)
    ax1.hlines(analytical, 9000, 100000, color = "red")
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1])

