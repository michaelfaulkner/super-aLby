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


def main(config_file_string, iact_sample_name='structure_factor'):
    fig, ax = plt.subplots(figsize=(10, 8))
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, temperature, number_of_equilibration_iterations,
     _, number_of_particles, size_of_particle_space) = helper_methods.get_basic_config_data(config_file_string)
    sh_file_string = f"{os.path.splitext(config_file_string)[0]}.sh"
    
    prefactors, iacts = np.load(os.path.join(sample_directory, f'{iact_sample_name}_iact_vs_prefactor.npy'))
    _, mean_rates = np.load(os.path.join(sample_directory, 'mean_event_rate_sweep.npy'))
    try:
        _, acc_rates = np.load(os.path.join(sample_directory, 'acceptance_rate_sweep.npy'))
    except FileNotFoundError:
        acc_rates = np.ones_like(iacts)

    print(len(mean_rates))
    print(len(iacts))
    print(len(acc_rates))
    comp_efforts = iacts * mean_rates / acc_rates

    ax.scatter(prefactors, comp_efforts, color='firebrick', marker='o', linestyle='-', alpha=0.7, linewidth=1.8)
    ax.set_xlabel('prefactor', fontsize=14)
    ax.set_ylabel("Comp Effort", fontsize=14)
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)

    plt.title(f"N={number_of_particles} L={size_of_particle_space} T={temperature}")
    plt.legend()
    plt.savefig(os.path.join(sample_directory, f"comp_effort_sweep.png"))
    np.save(os.path.join(sample_directory, f"comp_effort_sweep.npy"),
            np.vstack([prefactors, comp_efforts]))

    print(f"Plot saved: {os.path.join(sample_directory, f'comp_effort_sweep.png')}")

    return fig, ax


if __name__ == '__main__':
    main(*sys.argv[1:3])
