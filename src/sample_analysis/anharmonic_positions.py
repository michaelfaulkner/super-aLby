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
    
    print(np.shape(position_sample))

    timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
    mass = parsing.get_value(config, strings.to_camel_case(potential), "mass")
    omega_squared = parsing.get_value(config, strings.to_camel_case(potential), "omega_squared")
    anharmonicity = parsing.get_value(config, strings.to_camel_case(potential), "anharmonicity")
    bottom_of_well = np.sqrt(-mass * omega_squared * anharmonicity) \
            / 2 * anharmonicity 
            
    barrier_height = np.abs( -(timestep * anharmonicity * bottom_of_well **4 + 
                                0.5 * mass * timestep * omega_squared * bottom_of_well **2))
    fig, ax = plt.subplots(1,1)
   
    # for i in range(40,50,20):
    #     subsample = position_sample[i, :]
    #     ax.plot(np.arange(0, number_of_particles), subsample, linestyle="-", color="black")
    #     ax.scatter(np.arange(0, number_of_particles), subsample, marker = "x", color="black")
    #     ax.hlines(np.mean(subsample), xmin=0, xmax = number_of_particles, color="purple", label = "mean of x")
    #     ax.hlines(np.mean(np.abs(subsample)), xmin=0, xmax = number_of_particles, color="pink", label = "mean of |x|")
    #     ax.hlines(bottom_of_well, xmin=0, xmax = number_of_particles,linestyle=":", color = "gray")
    #     ax.hlines(-bottom_of_well, xmin=0, xmax = number_of_particles,linestyle=":", color = "gray")
    #     ax.hlines(0.0, xmin=0, xmax = number_of_particles,linestyle=":", color = "gray")

    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("metropolis_trajectories_2_start_05.png")

    for index in range(0, number_of_particles, 10):
        fig1, ax1 = plt.subplots(1,1)
        ax1.plot(np.arange(0, len(position_sample)), position_sample[:,20],linestyle="-", color="black")
        ax1.scatter(np.arange(0, len(position_sample))[np.nonzero(position_sample[:,20]>0.0)], position_sample[:,20][np.nonzero(position_sample[:,20]>0.0)], marker = "x", color = "blue")
        ax1.scatter(np.arange(0, len(position_sample))[np.nonzero(position_sample[:,20]<0.0)], position_sample[:,20][np.nonzero(position_sample[:,20]<0.0)], marker = "x", color = "red")

        ax1.hlines(bottom_of_well, xmin=0, xmax = len(position_sample),linestyle=":", color = "gray")
        ax1.hlines(-bottom_of_well, xmin=0, xmax = len(position_sample),linestyle=":", color = "gray")
        ax1.hlines(0.0, xmin=0, xmax = len(position_sample),linestyle=":", color = "gray")
        #ax1.set_xlim(200, 400)


            #ax.scatter(np.arange(0, number_of_particles)[np.nonzero(subsample>0.0)], subsample[np.nonzero(subsample>0.0)], color="red")
            #ax.scatter(np.arange(0, number_of_particles)[np.nonzero(subsample<0.0)], subsample[np.nonzero(subsample<0.0)], color="blue")
            

        #ax.plot(np.arange(0, number_of_particles), np.mean(position_sample, axis=0), color='black')

        plt.tight_layout()
        plt.savefig(f"metropolis_single_particle_trajectory_{index}.png")

if __name__ == '__main__':
    main(sys.argv[1])