from markov_chain_diagnostics import get_iact_and_acf
import importlib
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sample_getter
import sys


this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")


def main(config_folder, min, max, save_folder, sampler_name):
    min = int(min)
    max = int(max)

    iact_repeats = np.zeros((max - min))
    for iact_index in range(min, max):

        config_file_string = os.path.join(
            config_folder, f"{iact_index}.ini")
        print(config_file_string)
        config = parsing.read_config(
            parsing.parse_options([config_file_string]).config_file)
        (config_file_mediator, potential, _, samplers, sample_directory, temperature,
        number_of_equilibration_iterations, number_of_observations, number_of_particles,
        size_of_particle_space) = helper_methods.get_basic_config_data(config_file_string)

        mass = parsing.get_value(
            config, strings.to_camel_case(potential), "mass")
        timestep = parsing.get_value(
            config, strings.to_camel_case(potential), "timestep")
        sample_directory = sample_directory
        temperature_index = 0
        thinning_level = None

        if "fourier_mode_amplitude_sampler" in sampler_name:
            sample_for_iact = sample_getter.get_fourier_mode_amplitude(sample_directory, temperature, 0, number_of_particles,
                            number_of_equilibration_iterations, thinning_level=thinning_level)
        elif "standard_mean_position_sampler" in sampler_name:
            sample_for_iact = sample_getter.get_mean_positions(sample_directory, temperature, 0, number_of_particles,
                            number_of_equilibration_iterations, thinning_level=thinning_level)
        
        if len(np.shape(sample_for_iact)) > 1:
            sample_for_iact = sample_for_iact[:,0]


        iact, acf = get_iact_and_acf(sample_for_iact[:])

        print(f"{iact_index} iact = {iact}")

        iact_repeats[iact_index] = iact



        np.save(f"output/{save_folder}/iact.npy", iact_repeats)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
