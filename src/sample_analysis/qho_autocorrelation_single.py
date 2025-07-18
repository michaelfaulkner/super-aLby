from markov_chain_diagnostics import get_autocorrelation, get_iact_and_acf
import importlib
import numpy as np
import os
import sys


this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")


def main(config_file_string):
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directories, temperatures,
     number_of_equilibration_iterations, number_of_observations, number_of_particles,
     _, _, _) = helper_methods.get_basic_config_data(config_file_string)
    
    mass = parsing.get_value(config, strings.to_camel_case(potential), "mass")
    timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
    number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")
    sample_directory = sample_directories[0]
    temperature_index = 0
    thinning_level = None

    """
    mean_sample = sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index], temperature_index,
                                                   number_of_particles, number_of_equilibration_iterations,
                                                   thinning_level=thinning_level)
    """
    
    sub_arr_len = 51000
    num_sub_arrs = 20
    mean_sample = np.zeros(sub_arr_len * num_sub_arrs)
    for i in range(num_sub_arrs):
        mean_sample[i * sub_arr_len : (i+1) * sub_arr_len] = np.load(
            f"output/metropolis_001_checkpoints/temperature_00_run_{i:02d}_sample_of_mean_positions.npy")[1:, 0]
    mean_sample = mean_sample[50000:]
    acf = get_autocorrelation(mean_sample[:])
    iact = get_iact_and_acf(mean_sample[:])[0]
    np.save("acf_001_metropolis_thermalised.npy", acf)
    print(iact)


if __name__ == '__main__':
    main(sys.argv[1])
