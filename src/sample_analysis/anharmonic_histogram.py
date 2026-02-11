import numpy as np
import os
import importlib
import matplotlib
import matplotlib.pyplot as plt
import os
import sample_getter
import sys

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")
matplotlib.rcParams['mathtext.fontset'] = 'cm'

def main(config_file_string):

    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, temperature, number_of_equilibration_iterations,
     _, number_of_particles, size_of_particle_space) = helper_methods.get_basic_config_data(config_file_string)
    

    thinning_level = None
    position_sample = sample_getter.get_positions(sample_directory, temperature, 0, number_of_particles,
                                                  number_of_equilibration_iterations, thinning_level=thinning_level)
    
    mod_squared_position = np.square(np.abs(position_sample))
    
    timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
    mass = parsing.get_value(config, strings.to_camel_case(potential), "mass")
    omega_squared = parsing.get_value(config, strings.to_camel_case(potential), "omega_squared")
    anharmonicity = parsing.get_value(config, strings.to_camel_case(potential), "anharmonicity")
    bottom_of_well = np.sqrt(-mass * omega_squared/ 4 * anharmonicity)
    
   

    fig, ax = plt.subplots(1,1)
    ax.hist(position_sample.flatten(), bins = 50, density = True)
    ax.set_xlabel(r"$x$", fontsize = 16)
    ax.set_ylabel(r"$|\psi_0 (x)|^2$", fontsize = 16)
    lims = ax.get_ylim()
    ax.vlines(bottom_of_well, lims[0], lims[1], color = "gray")
    ax.vlines(-bottom_of_well, lims[0], lims[1], color = "gray")



    plt.savefig("anharmonic_hist.png")


if __name__ == '__main__':
    main(sys.argv[1])
