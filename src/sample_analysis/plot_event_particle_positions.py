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


def main(config_file_string, n):
    """Plot event steps u(s) ∈ {-1, 1}."""
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
     _, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)

    event_step_index = 0
    event_particle_position_index = 0
    for i, sampler in enumerate(samplers):
        if "event_particle_position_sampler" in sampler:
            event_particle_position_index = i
        if "event_step_sampler" in sampler:
            event_step_index = i
    event_particle_position_directory = sample_directories[event_particle_position_index]
    event_particle_position_path = os.path.join(event_particle_position_directory,
                                                'temperature_00_checkpoint_00_sample_of_event_particle_position.npy')
    event_step_directory = sample_directories[event_step_index]
    event_step_path = os.path.join(event_step_directory, 'temperature_00_checkpoint_00_sample_of_event_step.npy')

    event_particle_position_sample = np.load(event_particle_position_path)
    event_step_sample = np.load(event_step_path)
    n_cols = np.shape(event_particle_position_sample)[1]

    fig, ax = plt.subplots(2, 1)
    for i in range(n_cols):
        ax[0].plot(event_particle_position_sample[:, i][1:n+1])

    ax[1].plot(event_step_sample[:n])

    for x in range(0, n + 1):
        for axis in ax:
            axis.axvline(x, color="black", linestyle=":", linewidth=0.8, alpha=0.4)

    fig.subplots_adjust(hspace=0.0)
    fig_path = os.path.join(event_particle_position_directory, 'event_particle_positions.png')
    plt.savefig(fig_path)
    plt.show()


if __name__ == '__main__':
    n = int(sys.argv[2])
    main(sys.argv[1], n)
