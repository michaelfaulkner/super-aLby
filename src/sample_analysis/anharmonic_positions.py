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

    position_sample = np.load("output/anharmonic/checkpoint_00_sample_of_positions.npy")
    number_of_observations = 90000
    number_of_timeslices = 1000
    number_of_particles = 1000
    len_sample = 500
   
    for sample_index in range(0, number_of_observations, 20000):
        fig0, ax0 = plt.subplots(1,1)
        ax0.plot(np.arange(0, number_of_timeslices), position_sample[sample_index, :],linestyle="-", color="black")
        #ax0.hlines(bottom_of_well, xmin=0, xmax = len(position_sample[sample_index, :]),linestyle=":", color = "gray")
        #ax0.hlines(-bottom_of_well, xmin=0, xmax = len(position_sample[sample_index, :]),linestyle=":", color = "gray")
        #ax0.hlines(0.0, xmin=0, xmax = len(position_sample[sample_index, :]),linestyle=":", color = "gray")

        plt.tight_layout()
        plt.savefig(f"worldline_{sample_index}.png")

    plt.clf()

    for index in range(0, number_of_particles, 100):
        fig1, ax1 = plt.subplots(1,1)
        ax1.plot(np.arange(0, len(position_sample[:len_sample,index])), position_sample[:len_sample,index],linestyle="-", color="black")
        ax1.scatter(np.arange(0, len(position_sample[:len_sample,index]))[np.nonzero(position_sample[:len_sample,index]>0.0)], position_sample[:len_sample,index][np.nonzero(position_sample[:len_sample,index]>0.0)], marker = "x", color = "blue")
        ax1.scatter(np.arange(0, len(position_sample[:len_sample,index]))[np.nonzero(position_sample[:len_sample,index]<0.0)], position_sample[:len_sample,index][np.nonzero(position_sample[:len_sample,index]<0.0)], marker = "x", color = "red")

       # ax1.hlines(bottom_of_well, xmin=0, xmax = len(position_sample[:len_sample,index]),linestyle=":", color = "gray")
        #ax1.hlines(-bottom_of_well, xmin=0, xmax = len(position_sample[:len_sample,index]),linestyle=":", color = "gray")
        #ax1.hlines(0.0, xmin=0, xmax = len(position_sample[:len_sample,index]),linestyle=":", color = "gray")
        #ax1.set_xlim(200, 400)


            #ax.scatter(np.arange(0, number_of_particles)[np.nonzero(subsample>0.0)], subsample[np.nonzero(subsample>0.0)], color="red")
            #ax.scatter(np.arange(0, number_of_particles)[np.nonzero(subsample<0.0)], subsample[np.nonzero(subsample<0.0)], color="blue")
            

        #ax.plot(np.arange(0, number_of_particles), np.mean(position_sample, axis=0), color='black')

        plt.tight_layout()
        plt.savefig(f"single_particle_trajectory_{index}.png")

if __name__ == '__main__':
    main(sys.argv[1])