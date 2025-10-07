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

def main(config_file_folder_stem, number_of_repeats):
    number_of_repeats = int(number_of_repeats)

    mean_neighbour_displacement = np.zeros((len(os.listdir(config_file_folder_stem + f"_0")), number_of_repeats))
    timestep_arr = np.zeros((len(os.listdir(config_file_folder_stem + f"_0")), number_of_repeats))

    for repeat in range(number_of_repeats):
        config_file_folder = config_file_folder_stem + f"_{repeat}"

        for index, config_file in enumerate(os.listdir(config_file_folder)):
            config_file = os.path.join(config_file_folder, config_file)
            print(f"config file {config_file}")
            config = parsing.read_config(parsing.parse_options([config_file]).config_file)
            (config_file_mediator, potential, _, samplers, sample_directories, temperatures,
                number_of_equilibration_iterations, number_of_observations, number_of_particles,
                _, _, _) = helper_methods.get_basic_config_data(config_file)
            
            sample_directory = sample_directories[0]
            temperature_index = 0
            thinning_level = None
            number_of_equilibration_iterations = 0
            timestep_arr[index, repeat] = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
            try:
                neighbour_sample = sample_getter.get_neighbour_displacement_squared(sample_directory, temperatures[temperature_index],
                                temperature_index, 0, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
                mean_neighbour_displacement[index, repeat] = np.mean(np.mean(neighbour_sample, axis = 0))
        
            except:
                print(f"No sample was found for timestep {parsing.get_value(config, strings.to_camel_case(potential),
                                "timestep")}, this value will be skipped")

    timestep_arr = np.mean(timestep_arr, axis=1)
    #timestep_arr = timestep_arr[~np.isnan(mean_neighbour_displacement)]
    err = np.std(mean_neighbour_displacement, axis = 1)
    neighbour_displacement = np.mean(mean_neighbour_displacement, axis = 1)
    print(mean_neighbour_displacement)
    print(err)

    timestep_arr = timestep_arr[np.nonzero(neighbour_displacement)]
    err = err[np.nonzero(neighbour_displacement)]
    neighbour_displacement = neighbour_displacement[np.nonzero(neighbour_displacement)]
    

    #mean_neighbour_displacement = mean_neighbour_displacement[~np.isnan(mean_neighbour_displacement)]
    
    #fit_index = -1
    #e_coeffs = np.polyfit(np.log(120/mean_neighbour_displacement[:fit_index, 1, 0]), np.log(neighbour_displacement[:fit_index]), deg=1)
    #fitted_e = e_coeffs[1] + np.multiply(np.log(120/mean_neighbour_displacement[:fit_index, 1, 0]), e_coeffs[0])
    #print(f"coeffs: {e_coeffs}") 
    N = 120/timestep_arr
    neighbour_displacement = neighbour_displacement

    #plt.plot(120/mean_neighbour_displacement[:fit_index, 1, 0], np.exp(fitted_e), color="#d97dd9ff")
    plt.errorbar(N, neighbour_displacement, err, fmt='o', capsize=3, markersize=3, color = "#9e0ebeff")
    plt.xlabel(r"$N_{\tau}$", fontsize=17)
    plt.ylabel(r"$<(x_{i+1} - x_i)^2>$", fontsize=17)
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("neighbour_displacement.pdf")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])