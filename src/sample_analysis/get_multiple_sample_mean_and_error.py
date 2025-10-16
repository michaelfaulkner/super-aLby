from markov_chain_diagnostics import get_sample_mean_and_error
import importlib
import matplotlib
import numpy as np
import os
import sys

# Add the directory that contains the module plotting_functions to sys.path
this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")


def main(config_file_string, sample_name):
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, _, number_of_equilibration_iterations,
     _, number_of_particles, _) = helper_methods.get_basic_config_data(config_file_string)
    sh_file_string = f"{os.path.splitext(config_file_string)[0]}.sh"
    num_jobs = int(helper_methods.read_variable_from_sh_file(sh_file_string, "NUM_JOBS"))
    temp_values = helper_methods.get_temps_from_bash_file(sh_file_string)
    sample_paths = [[os.path.join(f"{sample_directory}/temperature_{temperature_index:02d}", f"job_{i:02d}")
                     for i in range(num_jobs)] for temperature_index in range(len(temp_values))]

    mean_value, mean_error = None, None
    for temp_index, temp_sample in enumerate(sample_paths):
        means = []
        errors = []
        for sample_path in temp_sample:
            try:
                sample = np.load(os.path.join(sample_path, f"checkpoint_00_sample_of_{sample_name}.npy")).flatten()
            except FileNotFoundError:
                continue
            mean, error = get_sample_mean_and_error(sample[number_of_equilibration_iterations:])
            means.append(mean)
            errors.append(error)

        mean_value = np.mean(means)
        mean_error = sum(error**2 for error in errors) ** 0.5 / len(errors)

        print(f"Temp: {temp_values[temp_index]} Mean value: {mean_value:.6f} +- {mean_error:.6f}")

    return mean_value, mean_error


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
