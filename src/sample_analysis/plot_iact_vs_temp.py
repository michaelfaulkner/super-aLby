from markov_chain_diagnostics import get_iact_and_acf
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


def main(config_file_string, sample_name):
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
     _, number_of_particles, _) = helper_methods.get_basic_config_data(config_file_string)
    sh_file_string = f"{os.path.splitext(config_file_string)[0]}.sh"
    num_jobs = int(helper_methods.read_variable_from_sh_file(sh_file_string, "NUM_JOBS"))
    temp_values = helper_methods.get_temps_from_bash_file(sh_file_string)
    sample_directories = [[f"{sampler_directory}/temperature_{temperature_index:02d}/job_00"
                           for sampler_directory in helper_methods.get_basic_config_data(config_file_string)[4]]
                          for temperature_index in range(len(temp_values))]
    sampler_index = None
    for index, sampler in enumerate(samplers):
        if sample_name in sampler:
            sampler_index = index
    sample_directory_paths = [os.path.dirname(temp_sample[sampler_index]) for temp_sample in sample_directories]
    sample_paths = [[os.path.join(potential_directory_path, f"job_{i:02d}")
                     for i in range(num_jobs)] for potential_directory_path in sample_directory_paths]

    iacts = []
    for temp_index, temp_sample in enumerate(sample_paths):
        temp_iacts = []
        for sample_path in temp_sample:
            try:
                sample = np.load(os.path.join(sample_path, f"checkpoint_00_sample_of_{sample_name}.npy")).flatten()
            except FileNotFoundError:
                continue
            iact = get_iact_and_acf(sample[number_of_equilibration_iterations:])[0]
            temp_iacts.append(iact)
        iacts.append(np.mean(temp_iacts)) if temp_iacts else iacts.append(float('nan'))

    plt.scatter(temp_values, iacts, color='firebrick', marker='o', linestyle='-', alpha=0.7, linewidth=1.8)

    plt.xlabel("Temperature", fontsize=14)
    plt.ylabel("IACT", fontsize=14)

    plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(sample_directory_paths[0]), "iact_vs_temp.png"))
    plt.show()
    np.save(os.path.join(os.path.dirname(sample_directory_paths[0]), "iact_vs_temp.npy"),
            np.vstack([temp_values, iacts]))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
