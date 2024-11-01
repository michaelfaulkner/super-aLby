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

def main():

    # read in the k values from k_values.txt
    current_directory = os.path.dirname(__file__)
    k_values_filepath = os.path.join(os.path.split(current_directory)[0], "m_values.txt")
    k_data = np.loadtxt(k_values_filepath, dtype='str')
    k_values = k_data[:,1]

    for index, string in enumerate(k_values):
        config_file_string = f"src/config_files/convergence_tests/one_dim_quantum_oscillator_potential/metropolis_{string}.ini"
        config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
        (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
        number_of_observations, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)
        
        dimensionless_mass = parsing.get_value(config, strings.to_camel_case(potential), "dimensionless_mass")
        timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
        number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")

        mass_as_string = str(dimensionless_mass)
        mass_string_split = mass_as_string.split(".")
        mass_as_string = ""
        for _ in mass_string_split:
            mass_as_string += _
        
        sample_directory = f"output/convergence_tests/one_dim_quantum_oscillator_potential/metropolis_{mass_as_string}"
        temperature_index = 0
        thinning_level = 10
        mean_sample = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                            temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
        position_sample = sample_getter.get_positions(sample_directory, temperatures[temperature_index],
                            temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=300)

        position_sample_squared = np.square(position_sample)
        fig, ax = plt.subplots(1,1)
        
        ax.scatter(np.arange(0,number_of_observations,thinning_level), mean_sample)
        ax.set_title(f"expectation of {dimensionless_mass}")
        ax.set_xlabel("metropolis sweeps")
        ax.set_ylabel("<x^2>")
        fig.suptitle(f"dim_m = {dimensionless_mass}")

        fig1, ax1 = plt.subplots(1,1)
        for index in range(len(position_sample)):
            
            ax1.scatter(np.arange(number_of_particles), position_sample[index,:], label=f"{index}", alpha=0.5)
            ax1.axhline(np.mean(position_sample[index,:]), color="red")
            ax1.set_xlabel("site")
            ax1.set_xlabel("site")
            ax1.set_xlabel("site")
            ax1.set_ylabel("Position")
            ax1.set_ylim(-1,1)
            fig1.suptitle(f"dim_m = {dimensionless_mass}")

            # fig2, ax2 = plt.subplots(1,3, sharey=True)
            # ax2[0].scatter(np.arange(number_of_particles), position_sample_squared[index,:], label=f"{index}", alpha=0.5)
            # ax2[0].axhline(np.mean(position_sample_squared[index,:]), color="purple")
            # ax2[0].set_xlabel("site")
            # ax2[1].set_xlabel("site")
            # ax2[2].set_xlabel("site")
            # ax2[0].set_ylabel("Position^2")
            # fig2.suptitle(f"dim_m = {dimensionless_mass}")

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
