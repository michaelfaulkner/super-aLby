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


def main(number_of_system_sizes=5):
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    lattice_lengths = [4 + index * 4 for index in range(number_of_system_sizes)]
    config_file_4x4_wolff = ["config_files/sampling_algos/ising_figures/4x4_wolff.ini"]
    (config_file_mediator, _, samplers, sample_directories_4x4_wolff, temperatures, number_of_equilibration_iterations,
     number_of_observations, _, _) = helper_methods.get_basic_config_data(config_file_4x4_wolff)
    output_directory = sample_directories_4x4_wolff[0].replace("/4x4_wolff", "")
    sample_directories = [f"{output_directory}/{length}x{length}_wolff" for length in lattice_lengths]
    if config_file_mediator == "metropolis_mediator":
        thinning_level = 10
    else:
        thinning_level = None

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    figure.tight_layout(w_pad=5.0)
    [axis.spines[spine].set_linewidth(3) for spine in ["top", "bottom", "left", "right"] for axis in axes]
    transition_temperature = 2.269185314213
    for axis in axes:
        axis.tick_params(which='both', direction='in', width=3)
        axis.tick_params(which='major', length=7, labelsize=18, pad=5)
        axis.tick_params(which='minor', length=4)
        axis.axvline(x=transition_temperature, color='y', linestyle='--')
    [axis.set_xlabel(r"$1 / (\beta J)$", fontsize=20, labelpad=3) for axis in axes]
    axes[0].set_ylabel(r"$\mathbb{E} C_{\rm V}$ / $N^{1 / 2}$", fontsize=20, labelpad=1)
    axes[1].set_ylabel(r"$\mathbb{E} {|m|}$", fontsize=20, labelpad=1)
    axes[1].set_ylim([0.0, 1.05])
    figure.text(0.235, -0.11, "(a)", fontsize=20)
    figure.text(0.75, -0.11, "(b)", fontsize=20)
    colors = ["black", "red", "blue", "green", "yellow", "cyan", "magenta"][:number_of_system_sizes]
    colors.reverse()

    for lattice_length_index, lattice_length in enumerate(lattice_lengths):
        (magnetic_norm_density_vs_temp, magnetic_norm_density_errors_vs_temp, specific_heat_vs_temp,
         specific_heat_errors_vs_temp) = (None, None, None, None)
        for sampler in samplers:
            if sampler == "standard_mean_position_sampler":
                _, _ = get_observable_mean_and_error_vs_temperature(
                    "magnetic_density", config_file_mediator, output_directory,
                    sample_directories[lattice_length_index], temperatures, lattice_length ** 2,
                    number_of_equilibration_iterations, thinning_level)
                (magnetic_norm_density_vs_temp,
                 magnetic_norm_density_errors_vs_temp) = get_observable_mean_and_error_vs_temperature(
                    "magnetic_norm_density", config_file_mediator, output_directory,
                    sample_directories[lattice_length_index], temperatures, lattice_length ** 2,
                    number_of_equilibration_iterations, thinning_level)
                _, _ = get_observable_mean_and_error_vs_temperature(
                    "magnetic_susceptibility", config_file_mediator, output_directory,
                    sample_directories[lattice_length_index], temperatures, lattice_length ** 2,
                    number_of_equilibration_iterations, thinning_level)
                _, _ = get_observable_mean_and_error_vs_temperature(
                    "magnetic_norm_susceptibility", config_file_mediator, output_directory,
                    sample_directories[lattice_length_index], temperatures, lattice_length ** 2,
                    number_of_equilibration_iterations, thinning_level)
            elif sampler == "potential_sampler":
                _, _ = get_observable_mean_and_error_vs_temperature(
                    "potential", config_file_mediator, output_directory, sample_directories[lattice_length_index],
                    temperatures, lattice_length ** 2, number_of_equilibration_iterations, thinning_level)
                (specific_heat_vs_temp, specific_heat_errors_vs_temp) = get_observable_mean_and_error_vs_temperature(
                    "specific_heat", config_file_mediator, output_directory, sample_directories[lattice_length_index],
                    temperatures, lattice_length ** 2, number_of_equilibration_iterations, thinning_level)

        axes[0].errorbar(temperatures, specific_heat_vs_temp / lattice_length,
                         specific_heat_errors_vs_temp / lattice_length, marker=".", markersize=8,
                         color=colors[lattice_length_index], linestyle="None",
                         label=fr"$N$ = {lattice_length}x{lattice_length}")
        axes[1].errorbar(temperatures, magnetic_norm_density_vs_temp, magnetic_norm_density_errors_vs_temp, marker=".",
                         markersize=8, color=colors[lattice_length_index], linestyle="None",
                         label=fr"$N$ = {lattice_length}x{lattice_length}")
    legends = [axes[0].legend(loc="upper left", fontsize=12), axes[1].legend(loc="lower left", fontsize=12)]
    [legend.get_frame().set_edgecolor("k") for legend in legends]
    [legend.get_frame().set_lw(3) for legend in legends]
    plt.savefig(f"{output_directory}/2d_ising_model_spec_heat_and_mag_norm_density_vs_temperature_"
                f"{config_file_mediator.replace('_mediator', '')}.pdf", bbox_inches="tight")


def get_observable_mean_and_error_vs_temperature(observable_string, config_file_mediator, output_directory,
                                                 sample_directory, temperatures, number_of_particles,
                                                 number_of_equilibration_iterations, thinning_level):
    try:
        with open(
                f"{output_directory}/2d_ising_model_{observable_string}_vs_temperature_"
                f"{config_file_mediator.replace('_mediator', '')}_{number_of_particles}_sites.tsv", "r") as output_file:
            output_file_sans_header = np.array([np.fromstring(line, dtype=float, sep='\t') for line in output_file
                                                if not line.startswith('#')]).transpose()
            means_vs_temperature, errors_vs_temperature = output_file_sans_header[1], output_file_sans_header[2]
    except IOError:
        output_file = open(
            f"{output_directory}/2d_ising_model_{observable_string}_vs_temperature_"
            f"{config_file_mediator.replace('_mediator', '')}_{number_of_particles}_sites.tsv", "w")
        output_file.write("# temperature".ljust(30) + observable_string.ljust(35) + observable_string + " error" + "\n")
        means_vs_temperature, errors_vs_temperature = [], []
        get_sample_method = getattr(sample_getter, "get_" + observable_string)
        for temperature_index, temperature in enumerate(temperatures):
            sample_mean, sample_error = get_sample_mean_and_error(get_sample_method(
                sample_directory, temperature, temperature_index, number_of_particles,
                number_of_equilibration_iterations, thinning_level))
            output_file.write(
                f"{temperature:.14e}".ljust(30) + f"{sample_mean:.14e}".ljust(35) + f"{sample_error:.14e}" + "\n")
            means_vs_temperature.append(sample_mean)
            errors_vs_temperature.append(sample_error)
        output_file.close()
    return np.array([means_vs_temperature, errors_vs_temperature])


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise Exception("InterfaceError: At most one positional argument permitted.  None are required but you may "
                        "provide number_of_system_sizes, which must be an integer greater than 0 and less than 6 "
                        "(default value is 5).")
    if len(sys.argv) == 2:
        print("One positional argument provided.  This must be number_of_system_sizes - which must be an integer "
              "greater than 0 and less than 6 (default value is 5).")
        chosen_number_of_system_sizes = int(sys.argv[1])
        if chosen_number_of_system_sizes < 1 or chosen_number_of_system_sizes > 4:
            raise Exception("InterfaceError: chosen_number_of_system_sizes must be an integer greater than 0 and less "
                            "than 6 (default value is 5).")
        main(chosen_number_of_system_sizes)
    else:
        print("No positional arguments provided.  None are required but you may provide chosen_number_of_system_sizes, "
              "which must be an integer greater than 0 and less than 6 (default value is 5).")
        main()
