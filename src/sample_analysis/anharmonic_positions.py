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
    fig, ax = plt.subplots(1,1)
   
    for i in range(50,70,20):
        subsample = position_sample[i, :]
        ax.plot(np.arange(0, number_of_particles), subsample, linestyle="-", alpha=0.5, color="gray")
        ax.scatter(np.arange(0, number_of_particles)[np.nonzero(subsample>0.0)], subsample[np.nonzero(subsample>0.0)], color="red")
        ax.scatter(np.arange(0, number_of_particles)[np.nonzero(subsample<0.0)], subsample[np.nonzero(subsample<0.0)], color="blue")

    #ax.plot(np.arange(0, number_of_particles), np.mean(position_sample, axis=0), color='black')
    

    plt.tight_layout()
    plt.savefig("ecmc_trajectories.png")




if __name__ == '__main__':
    main(sys.argv[1])