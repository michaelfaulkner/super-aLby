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


def find_mean_and_max_chain_length(event_step_sample):
    lengths = []
    u = event_step_sample[0]
    length = 1
    max_length = 1
    for obs in event_step_sample:
        if obs == u:
            length += 1
        else:
            lengths.append(length)
            u = obs
            max_length = max(max_length, length)
            length = 1
    mean_chain_length = np.mean(lengths) if lengths else len(event_step_sample)
    return mean_chain_length, max_length if lengths else mean_chain_length


def main(config_file_string, n, m, vlines=1, remove_eq=0):
    """Plot the event particle positions, active particle indices, and event steps from index n to m."""
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, factor_field, samplers, sample_directory, temperature,
     number_of_equilibration_iterations, number_of_observations, number_of_particles, size_of_particle_space) = (
        helper_methods.get_basic_config_data(config_file_string))

    event_particle_position_path = os.path.join(sample_directory, 'checkpoint_00_sample_of_event_particle_position.npy')
    event_particle_position_sample = np.load(event_particle_position_path)[remove_eq *
                                                                           number_of_equilibration_iterations:]

    event_step_path = os.path.join(sample_directory, 'checkpoint_00_sample_of_event_step.npy')
    event_step_sample = np.load(event_step_path)[remove_eq * number_of_equilibration_iterations:]

    event_active_particle_index_path = (
        os.path.join(sample_directory, 'checkpoint_00_sample_of_event_active_particle_index.npy'))
    event_active_particle_index_sample = (
        np.load(event_active_particle_index_path))[remove_eq * number_of_equilibration_iterations:]

    n_cols = np.shape(event_particle_position_sample)[1]

    mean_chain_length, max_chain_length = find_mean_and_max_chain_length(event_step_sample)

    num_plus = sum(1 if step == 1 else 0 for step in event_step_sample)
    prop_plus = num_plus / len(event_step_sample)
    mean_pointer_velocity = np.mean(event_step_sample)

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))

    for i in range(n_cols):
        ax[0].plot(range(n, m+1), event_particle_position_sample[:, i][n:m+1], alpha=0.7, label=f'Particle {i}')
    ax[0].legend(fontsize=6)
    ax[0].set_ylabel('Particle Position', fontsize=8)

    ax[1].plot(range(n, m+1), event_active_particle_index_sample[n:m+1], marker='o', markersize=3, color='firebrick')
    ax[1].set_ylabel('Active Particle Index', fontsize=8)

    ax[2].plot(range(n+1, m+1 + 1), event_step_sample[n:m+1], color='black')
    ax[2].set_ylim(-1.2, 1.2)
    ax[2].set_ylabel('Event Step', fontsize=8)

    if vlines == 1:
        for x in range(n, m + 1):
            for axis in ax:
                axis.axvline(x, color="black", linestyle=":", linewidth=0.8, alpha=0.4)

    fig.subplots_adjust(hspace=0.0)

    plt.savefig(os.path.join(sample_directory, f'event_particle_positions_{n}_{m}.png'))
    plt.show()


if __name__ == '__main__':
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    vert_lines = int(sys.argv[4])
    remove_equ = int(sys.argv[5])
    main(sys.argv[1], start, end, vert_lines, remove_equ)
