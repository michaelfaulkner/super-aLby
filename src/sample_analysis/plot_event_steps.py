import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import importlib

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")


def main(config_file_string):
    """Plot event steps u(s) ∈ {-1, 1}."""
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
     _, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)

    event_step_index = 0
    for i, sampler in enumerate(samplers):
        if "event_step_sampler" in sampler:
            event_step_index = i
    event_step_directory = sample_directories[event_step_index]
    event_step_path = os.path.join(event_step_directory, 'temperature_00_checkpoint_00_sample_of_event_step.npy')
    sample = np.load(event_step_path).flatten()

    distances = []
    u = sample[0]
    distance = 1
    for obs in sample:
        if obs == u:
            distance += 1
        else:
            distances.append(distance)
            u = obs
            distance = 1
    print(np.mean(distances))

    plt.plot(sample[5000:5300])
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1])
