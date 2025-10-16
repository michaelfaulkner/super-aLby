from markov_chain_diagnostics import get_sample_mean_and_error
import importlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sample_getter
import sys

# Add the directory that contains the module plotting_functions to sys.path
this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")


def main(config_file_string):
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, temperatures, number_of_equilibration_iterations,
     _, number_of_particles, _) = helper_methods.get_basic_config_data(config_file_string)
    sh_file_string = f"{os.path.splitext(config_file_string)[0]}.sh"
    num_jobs = int(helper_methods.read_variable_from_sh_file(sh_file_string, "NUM_JOBS"))
    temp_values = helper_methods.get_temps_from_bash_file(sh_file_string)
    potential_sample_paths = [[os.path.join(f"{sample_directory}/temperature_{temperature_index:02d}", f"job_{i:02d}")
                               for i in range(num_jobs)] for temperature_index in range(len(temp_values))]

    specific_heats = []
    for temp_index, temp_sample in enumerate(potential_sample_paths):
        temp_specific_heats = []
        for potential_sample_path in temp_sample:
            try:
                specific_heat_sample = sample_getter.get_specific_heat(
                    potential_sample_path, temp_values[temp_index], 0, number_of_particles,
                    number_of_equilibration_iterations)
            except FileNotFoundError:
                continue
            specific_heat = np.mean(specific_heat_sample)
            temp_specific_heats.append(specific_heat)
        specific_heats.append(np.mean(temp_specific_heats))

    plt.scatter(temp_values, specific_heats, color='firebrick', marker='o', linestyle='-', alpha=0.7, linewidth=1.8)

    plt.xlabel("Temperature", fontsize=14)
    plt.ylabel("Specific Heat", fontsize=14)

    plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(sample_directory, "specific_heat_vs_temp.png"))
    plt.show()
    np.save(os.path.join(sample_directory, "specific_heat_vs_temp.npy"),
            np.vstack([temp_values, specific_heats]))


if __name__ == '__main__':
    main(sys.argv[1])
