import numpy as np
import matplotlib.pyplot as plt
import os
import importlib
import matplotlib
import sample_getter
import sys
from configparser import NoOptionError
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")

def main(config_file_folder):

    mean_neighbour_displacement = np.zeros((len(os.listdir(config_file_folder)),2))

    for index, config_file in enumerate(os.listdir(config_file_folder)):
        config_file = os.path.join(config_file_folder, config_file)
        config = parsing.read_config(parsing.parse_options([config_file]).config_file)
        (config_file_mediator, potential, _, samplers, sample_directories, temperatures,
            number_of_equilibration_iterations, number_of_observations, number_of_particles,
            _, _, _) = helper_methods.get_basic_config_data(config_file)
        
        sample_directory = sample_directories[0]
        temperature_index = 0
        thinning_level = None
        number_of_equilibration_iterations = 0
        try:
            neighbour_sample = sample_getter.get_neighbour_displacement_squared(sample_directory, temperatures[temperature_index],
                            temperature_index, 0, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
            mean_neighbour_displacement[index, 0] = np.mean(neighbour_sample)
            print(np.mean(neighbour_sample), parsing.get_value(config, strings.to_camel_case(potential), "timestep"))
            mean_neighbour_displacement[index, 1] = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
        except:
            print(f"No sample was found for timestep {parsing.get_value(config, strings.to_camel_case(potential),
                            "timestep")}, this value will be skipped")
            
    print(mean_neighbour_displacement)
    print(np.shape(mean_neighbour_displacement))
    mean_neighbour_displacement = mean_neighbour_displacement[np.nonzero(mean_neighbour_displacement[:,0])]
    mean_neighbour_displacement = mean_neighbour_displacement[~np.isnan(mean_neighbour_displacement[:,0])]
    fit_index = -1
    e_coeffs = np.polyfit(np.log(120/mean_neighbour_displacement[:fit_index, 1]), np.log(mean_neighbour_displacement[:fit_index, 0]), deg=1)
    fitted_e = e_coeffs[1] + np.multiply(np.log(120/mean_neighbour_displacement[:fit_index, 1]), e_coeffs[0])
    print(f"coeffs: {e_coeffs}")


    plt.plot(120/mean_neighbour_displacement[:fit_index, 1], np.exp(fitted_e), color="#d97dd9ff")
    print("---------------------------")
    print(mean_neighbour_displacement)
    print(np.shape(mean_neighbour_displacement))
    plt.scatter(120/mean_neighbour_displacement[:,1], mean_neighbour_displacement[:,0], color = "#9e0ebeff")
    plt.xlabel(r"$N_{\tau}$", fontsize=17)
    plt.ylabel(r"$<(x_{i+1} - x_i)^2>$", fontsize=17)
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("neighbour_displacement.pdf")

if __name__ == '__main__':
    main(sys.argv[1])