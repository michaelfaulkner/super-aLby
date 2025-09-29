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

    displacement = np.zeros((len(os.listdir(config_file_folder_stem + f"_0")), number_of_repeats))
    N = np.zeros((len(os.listdir(config_file_folder_stem + f"_0")), number_of_repeats))

    for repeat in range(number_of_repeats):
        config_file_folder = config_file_folder_stem + f"_{repeat}"
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
            N[index, repeat] = 120/parsing.get_value(config, strings.to_camel_case(potential), "timestep")
            if repeat == 0:
                checkpoint_index = 0
            else:
                checkpoint_index = 1
            try: 
                position_sample = sample_getter.get_positions(sample_directory, temperatures[temperature_index],
                                    temperature_index, checkpoint_index, number_of_particles, number_of_equilibration_iterations, thinning_level=thinning_level)
                neighbour_displacement = np.zeros((np.shape(position_sample)))

                for sample_index, sample in enumerate(position_sample):
                    for position_index, position_value in enumerate(position_sample[sample_index, :]):
                        neighbour_displacement[sample_index, position_index] = \
                            (position_sample[sample_index, helper_methods.get_east_neighbour(position_index, number_of_particles)] - position_value)**2
                
                displacement[index, repeat] = np.mean(neighbour_displacement)
                
            except:
                print(f"No sample was found for timestep {parsing.get_value(config, strings.to_camel_case(potential),
                                    "timestep")}, this value will be skipped")

    print(displacement)
    print(N)
    print(np.shape(displacement))
    print(np.shape(N))
    #e_coeffs = np.polyfit(np.log(N), np.log(displacement), deg=1)
    #fitted_e = e_coeffs[1] + np.multiply(np.log(N), e_coeffs[0])


    err = np.std(displacement, axis=1)
    displacement = np.mean(displacement, axis=1)

    N = N[np.nonzero(displacement)]
    displacement = displacement[np.nonzero(displacement)]
    err = err[np.nonzero(displacement)]

    
    #plt.plot(N, np.exp(fitted_e), color="#d97dd9ff")
    #plt.scatter(N, displacement)
    plt.errorbar(N[:,0], displacement, err, fmt='o', capsize=3, markersize=3, color = "#9e0ebeff")

    plt.xlabel(r"$N_{\tau}$", fontsize=17)
    plt.ylabel(r"$<(x_{i+1} - x_i)^2>$", fontsize=17)
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("q.pdf")
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])