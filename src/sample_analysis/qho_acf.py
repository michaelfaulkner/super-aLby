from markov_chain_diagnostics import get_iact_and_acf
import importlib
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sample_getter
import sys
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')



this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")

def main(config_file_string):

    config = parsing.read_config(
        parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, temperature,
        number_of_equilibration_iterations, number_of_observations, number_of_particles,
        size_of_particle_space) = helper_methods.get_basic_config_data(config_file_string)

    mass = parsing.get_value(
        config, strings.to_camel_case(potential), "mass")
    timestep = parsing.get_value(
        config, strings.to_camel_case(potential), "timestep")
    omega_squared = parsing.get_value(
        config, strings.to_camel_case(potential), "omega_squared")
    sample_directory = sample_directory
    temperature_index = 0
    thinning_level = None
    try:
        sample = sample_getter.get_mean_squared_positions(sample_directory, temperature, 0, number_of_particles, number_of_equilibration_iterations,
                                    thinning_level=thinning_level)
    except:
        sample = sample_getter.get_positions(sample_directory, temperature, 0, number_of_particles, number_of_equilibration_iterations,
                                    thinning_level=thinning_level)
        sample = np.mean(np.square(sample-2), axis = 1)
    
    #sample = sample[30000:50000]
                    
    if len(np.shape(sample)) > 1:
        sample = sample[:,0]

    sample = sample[:300]
    iact, acf = get_iact_and_acf(sample[:])


    print(np.shape(acf))
    print(iact)

    np.save(f"acf_1_x0_2_x-a.npy", acf)

    plt.scatter(np.arange(len(acf)), acf)
    plt.ylim(-1.75, 1.75)

    plt.annotate(f"IACT = {iact}", (1000, 1.0))

    plt.savefig(f"acf_1_x0_2_x-a.png")


if __name__ == '__main__':
    main(sys.argv[1])

