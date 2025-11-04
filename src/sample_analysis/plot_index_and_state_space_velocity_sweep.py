import importlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json

# Add the directory that contains the module plotting_functions to sys.path
this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")


def main(config_file_string, same_plot=1):
    fig, ax = plt.subplots(figsize=(10, 8)) if same_plot == 1 else plt.subplots(1, 2, figsize=(14, 8))
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, temperature, number_of_equilibration_iterations,
     _, number_of_particles, size_of_particle_space) = helper_methods.get_basic_config_data(config_file_string)
    sh_file_string = f"{os.path.splitext(config_file_string)[0]}.sh"
    num_jobs = int(helper_methods.read_variable_from_sh_file(sh_file_string, "NUM_JOBS"))
    sweep_name = helper_methods.read_variable_from_sh_file(sh_file_string, "CONFIG_VARIABLE")
    sweep_values = helper_methods.get_temps_from_bash_file(sh_file_string)
    sample_paths = [[os.path.join(f"{sample_directory}/{sweep_name}_{temperature_index:02d}", f"job_{i:02d}")
                     for i in range(num_jobs)] for temperature_index in range(len(sweep_values))]

    state_space_velocities, index_space_velocities = [], []
    for sweep_index, sweep_sample in enumerate(sample_paths):
        sweep_state_space_velocities, sweep_index_space_velocities = [], []
        for sample_path in sweep_sample:
            try:
                with open(os.path.join(sample_path, "state_and_index_space_velocities.json"), 'r') as f:
                    sample = json.load(f)
            except FileNotFoundError:
                continue
            state_space_velocity, index_space_velocity = sample["state_space_velocity"], sample["index_space_velocity"]
            (sweep_state_space_velocities.append(state_space_velocity),
             sweep_index_space_velocities.append(index_space_velocity))

        state_space_velocities.append(np.mean(sweep_state_space_velocities)) if (
            sweep_state_space_velocities) else state_space_velocities.append(float('nan'))
        index_space_velocities.append(np.mean(sweep_index_space_velocities)) if (
            sweep_index_space_velocities) else index_space_velocities.append(float('nan'))

    if same_plot:
        ax.scatter(sweep_values, state_space_velocities, color='firebrick', marker='o', linestyle='-', alpha=0.7,
                   linewidth=1.8, label="Mean State Space Velocity")
        ax.scatter(sweep_values, index_space_velocities, color='grey', marker='o', linestyle='-', alpha=0.7,
                   linewidth=1.8, label="Mean Index Space Velocity")
        ax.set_xlabel(sweep_name, fontsize=14)
        ax.set_ylabel("Mean Velocity", fontsize=14)
        ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)

    else:

        ax[0].scatter(sweep_values, state_space_velocities, color='firebrick', marker='8', linestyle='-', alpha=0.7,
                      linewidth=1.8)
        ax[1].scatter(sweep_values, index_space_velocities, color='grey', marker='8', linestyle='-', alpha=0.7,
                      linewidth=1.8)

        ax[0].set_xlabel(sweep_name, fontsize=14)
        ax[0].set_ylabel("Mean State Space Velocity", fontsize=14)
        ax[0].grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)

        ax[1].set_xlabel(sweep_name, fontsize=14)
        ax[1].set_ylabel("Mean Index Space Velocity", fontsize=14)
        ax[1].grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)

    plt.title(f"N={number_of_particles} L={size_of_particle_space} T={temperature}")
    plt.legend()
    plt.savefig(os.path.join(sample_directory, f"index_and_state_space_velocity_{sweep_name}_sweep.png"))
    plt.show()
    np.save(os.path.join(sample_directory, f"index_and_state_space_velocity_{sweep_name}_sweep.npy"),
            np.vstack([sweep_values, state_space_velocities, index_space_velocities]))

    print(f"Plot saved: {os.path.join(sample_directory, f'index_and_state_space_velocity_{sweep_name}_sweep.png')}")

    return fig, ax


if __name__ == '__main__':
    if len(sys.argv) <= 2:
        main(sys.argv[1])
    else:
        main(sys.argv[1], int(sys.argv[2]))
