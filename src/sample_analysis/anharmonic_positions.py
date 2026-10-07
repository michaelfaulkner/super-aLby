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

def main(positions_data_path):
    # config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    # (config_file_mediator, potential, _, samplers, sample_directory, temperature, number_of_equilibration_iterations,
    #  number_of_observations, number_of_particles, size_of_particle_space) = helper_methods.get_basic_config_data(config_file_string)
    

    # thinning_level = None
    # position_sample = sample_getter.get_positions(sample_directory, temperature, 0, number_of_particles,
    #                                               number_of_equilibration_iterations, thinning_level=thinning_level)
    
    # print(np.shape(position_sample))

    # timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
    # mass = parsing.get_value(config, strings.to_camel_case(potential), "mass")
    # omega_squared = parsing.get_value(config, strings.to_camel_case(potential), "omega_squared")
    # anharmonicity = parsing.get_value(config, strings.to_camel_case(potential), "anharmonicity")
    # number_of_timeslices = parsing.get_value(config, "ModelSettings", "number_of_timeslices")



    # bottom_of_well = np.sqrt(-mass * omega_squared * anharmonicity) \
    #         / 2 * anharmonicity 
            
    # barrier_height = np.abs( -(timestep * anharmonicity * bottom_of_well **4 + 
    #                             0.5 * mass * timestep * omega_squared * bottom_of_well **2))

    position_sample = np.load(positions_data_path)
    mean_positions = np.mean(position_sample, axis = 1)
    plt.scatter(np.arange(len(mean_positions)), mean_positions)
    plt.savefig("test.png")
    plt.clf()

    number_of_timeslices = len(position_sample[0, :])

    for sample_index in [0, 10, 20, 50, 100, 2000, 3000]:
        fig0, ax0 = plt.subplots(1,1)
        ax0.plot(np.arange(0, number_of_timeslices), position_sample[sample_index, :],linestyle="-", color="black")
        #ax0.hlines(bottom_of_well, xmin=0, xmax = len(position_sample[sample_index, :]),linestyle=":", color = "gray")
        #ax0.hlines(-bottom_of_well, xmin=0, xmax = len(position_sample[sample_index, :]),linestyle=":", color = "gray")
        #ax0.hlines(0.0, xmin=0, xmax = len(position_sample[sample_index, :]),linestyle=":", color = "gray")

        plt.tight_layout()
        plt.savefig(f"worldline_{sample_index}.png")

  

    
if __name__ == '__main__':
    main(sys.argv[1])