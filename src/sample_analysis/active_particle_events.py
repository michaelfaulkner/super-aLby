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


def main(config_file_string, cutoff):

    cutoff = int(cutoff)

    config = parsing.read_config(
                    parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, temperature,
    number_of_equilibration_iterations, number_of_observations, number_of_particles,
    size_of_particle_space) = helper_methods.get_basic_config_data(config_file_string)

    timestep = parsing.get_value(
        config, strings.to_camel_case(potential), "timestep")
    temperature_index = 0
    thinning_level = None

    active_particle_sample = sample_getter.get_event_active_particle_index(sample_directory, temperature, 0,
                                number_of_particles, 0, thinning_level)

    unwrapped_active_particle_sample = np.zeros(np.shape(active_particle_sample))

    boundaries = 0
    msd = np.zeros(np.shape(active_particle_sample))

    initial_index =  active_particle_sample[0]

    for index, active_particle in enumerate(active_particle_sample[:cutoff]):

        if index != 0:
            boundaries = check_boundary(active_particle_sample[index-1], active_particle, number_of_particles,
                                        boundaries)

        active_particle_unwrapped = active_particle + boundaries * number_of_particles

        msd[index] = (active_particle_unwrapped - initial_index)**2 / number_of_particles

        unwrapped_active_particle_sample[index] = active_particle_unwrapped

        

    cutoff_unwrapped = unwrapped_active_particle_sample[:cutoff]


    print(np.shape(cutoff_unwrapped))

    fig, ax = plt.subplots(1,1)

    ax.scatter(np.arange(len(cutoff_unwrapped)), cutoff_unwrapped, s= 5.0)
    ax.plot(np.arange(len(cutoff_unwrapped)), msd[:cutoff], color = "red")


    ax.set_xlabel("Event number")
    ax.set_ylabel("Active particle index (unwrapped)")


    plt.savefig("active_particle_events_asymm.png")


            






def check_boundary(index_before, index_after, number_of_particles, boundaries):

    if index_before == number_of_particles - 1 and index_after == 0:
        return boundaries + 1
    elif index_before == 0 and index_after == number_of_particles - 1:
        return boundaries - 1
    else:
        return boundaries



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])