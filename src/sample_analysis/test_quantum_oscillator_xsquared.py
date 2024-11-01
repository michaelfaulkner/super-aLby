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

def main(config_file_string):

    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
     _, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)
    
    mass = parsing.get_value(config, strings.to_camel_case(potential), "mass")
    timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
    number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")

    mass_as_string = str(mass)
    mass_string_split = mass_as_string.split(".")
    mass_as_string = ""
    for _ in mass_string_split:
        mass_as_string += _
    
    sample_directory = f"output/convergence_tests/one_dim_quantum_oscillator_potential/metropolis_{mass_as_string}"
    temperature_index = 0
    mean_sample = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                        temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=None)
    position_sample = sample_getter.get_positions(sample_directory, temperatures[temperature_index],
                        temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=None)

    #position_sample = np.square(position_sample)
    fig, ax = plt.subplots(1,1)
    ax.scatter(np.arange(1000), mean_sample)

    fig1, ax1 = plt.subplots(1,3, sharey=True)
    x = 0
    for index in range(1000):
        if index%200 == 0:
            if x<2:
                ax1[0].scatter(np.arange(1200), position_sample[index,:], label=f"{index}", alpha=0.5)
                ax1[0].axhline(np.mean(position_sample[index,:]), color="red")
            elif x<3:
                ax1[1].scatter(np.arange(1200), position_sample[index,:], label=f"{index}", alpha=0.5)
                ax1[1].axhline(np.mean(position_sample[index,:]), color="red")
            else:
                ax1[2].scatter(np.arange(1200), position_sample[index,:], label=f"{index}", alpha=0.5)
                ax1[2].axhline(np.mean(position_sample[index,:]), color="red")
            x+=1
    ax1[0].set_xlabel("site")
    ax1[1].set_xlabel("site")
    ax1[2].set_xlabel("site")
    ax1[0].set_ylabel("Position")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1])
