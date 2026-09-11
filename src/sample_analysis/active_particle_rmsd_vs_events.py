import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import importlib
import sample_getter

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")


def main(config_file_string, save_folder):


    config = parsing.read_config(
        parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, temperature,
        number_of_equilibration_iterations, number_of_observations, number_of_particles,
        size_of_particle_space) = helper_methods.get_basic_config_data(config_file_string)

    timestep = parsing.get_value(
        config, strings.to_camel_case(potential), "timestep")
    sample_directory = sample_directory
    temperature_index = 0
    thinning_level = None

        
    checkpointing_index = sample_getter.get_checkpointing_indices(
        sample_directory)

    if checkpointing_index != 0:
        max_len = 0
        for i in range(checkpointing_index + 1):
            new_len = len(sample_getter.get_event_active_particle_index(
                sample_directory, temperature, i, number_of_particles,
                None, thinning_level=thinning_level)[:, 0])
            if new_len > max_len:
                max_len = new_len

        active_particle_sample = np.zeros(max_len * (checkpointing_index + 1))
        
        for i in range(checkpointing_index + 1):
            sub_arr = sample_getter.get_event_active_particle_index(
                sample_directory, temperature, i, number_of_particles,
                None, thinning_level=thinning_level)[:, 0]
            try:
                active_particle_sample[i *
                                        max_len: (i) * max_len + len(sub_arr)] = sub_arr
            except:
                active_particle_sample[i * max_len: (
                    i) * max_len + len(sub_arr)] = sub_arr[1:]

        active_particle_sample = active_particle_sample[np.nonzero(
            active_particle_sample)]
        active_particle_sample = active_particle_sample[number_of_equilibration_iterations:]

    else:
        active_particle_sample = sample_getter.get_event_active_particle_index(sample_directory, temperature, 0,
                                                                               number_of_particles,
                                                                               number_of_equilibration_iterations,
                                                                                thinning_level=thinning_level)

    initial_index = active_particle_sample[0]
    rmsd_events_array = np.zeros((len(active_particle_sample)-1))

    msd = 0.0
    boundaries = 0

    for particle_index in range(len(active_particle_sample)):
        if particle_index != 0:
            current_active_particle_index = active_particle_sample[particle_index]
            boundaries = check_boundary(active_particle_sample[particle_index-1], current_active_particle_index,
                                        number_of_particles, boundaries)
            if boundaries == 0:
                msd += (current_active_particle_index -
                        initial_index)**2
            else:
                msd += (np.sign(boundaries) * current_active_particle_index +
                        np.abs(boundaries) * number_of_particles - np.sign(boundaries) * initial_index)**2

            rmsd_events_array[particle_index-1] = np.sqrt(msd / particle_index)    



    np.save(f"output/{save_folder}/rmsd_vs_events_ecmc.npy", rmsd_events_array)


def check_boundary(index_before, index_after, number_of_particles, boundaries):

    if index_before == number_of_particles -1 and index_after == 0:
        return boundaries + 1
    elif index_before == 0 and index_after == number_of_particles - 1:
        return boundaries - 1
    else:
        return boundaries


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
