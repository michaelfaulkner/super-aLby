import importlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json
import glob

# Add the directory that contains the module plotting_functions to sys.path
this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")


def main(config_file_string):
    fig, ax = plt.subplots(figsize=(10, 8))
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, temperature, number_of_equilibration_iterations,
     _, number_of_particles, size_of_particle_space) = helper_methods.get_basic_config_data(config_file_string)
    sh_file_string = f"{os.path.splitext(config_file_string)[0]}.sh"
    
    num_jobs = int(helper_methods.read_variable_from_sh_file(sh_file_string, "NUM_JOBS"))
    sweep_name = helper_methods.read_variable_from_sh_file(sh_file_string, "CONFIG_VARIABLE")
    sweep_values = helper_methods.get_temps_from_bash_file(sh_file_string)
    
    sample_paths = [sorted(glob.glob(os.path.join(f"{sample_directory}/{sweep_name}_{temperature_index:02d}", "job_*")))
                    for temperature_index in range(len(sweep_values))]

    mean_rates = []
    sweeps_values = []
    for sweep_index, sweep_sample in enumerate(sample_paths):
        sweep_rates = []
        for sample_path in sweep_sample:
            try:
                with open(os.path.join(sample_path, "sim_params.json"), 'r') as f:
                    sample = json.load(f)
            except FileNotFoundError:
                try:
                    with open(os.path.join(sample_path, 'state_and_index_space_velocities.json'), 'r') as f:
                        sample = json.load(f)
                except FileNotFoundError:
                    continue
            
            rate = sample["acceptance_rate"]
            sweep_rates.append(rate)

        if len(sweep_rates) > 0:
            mean_rates.append(np.mean(sweep_rates))
            sweeps_values.append(sweep_values[sweep_index])

    ax.scatter(sweeps_values, mean_rates, color='firebrick', marker='o', linestyle='-', alpha=0.7,
                linewidth=1.8)
    ax.set_xlabel(sweep_name, fontsize=14)
    ax.set_ylabel("Acceptance Rate", fontsize=14)
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)

    plt.title(f"N={number_of_particles} L={size_of_particle_space} T={temperature}")
    plt.legend()
    plt.savefig(os.path.join(sample_directory, f"acceptance_rate_sweep.png"))
    np.save(os.path.join(sample_directory, f"acceptance_rate_sweep.npy"),
            np.vstack([sweeps_values, mean_rates]))

    print(f"Plot saved: {os.path.join(sample_directory, f'acceptance_rate_sweep.png')}")

    return fig, ax


if __name__ == '__main__':
    main(sys.argv[1])
