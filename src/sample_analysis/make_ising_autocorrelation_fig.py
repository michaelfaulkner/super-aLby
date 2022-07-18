from markov_chain_diagnostics import get_autocorrelation
import importlib
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sample_getter
import sys

from itertools import cycle, islice

# Add the directory that contains the module plotting_functions to sys.path
this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")


def main(number_of_system_sizes=3):
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    lattice_lengths = [2 ** (index + 2) for index in range(number_of_system_sizes)]
    # lattice_lengths = [4 + index * 4 for index in range(number_of_system_sizes)]
    config_file_4x4_wolff = ["config_files/sampling_algos/ising_figures/4x4_wolff.ini"]
    config_file_4x4_metrop = ["config_files/sampling_algos/ising_figures/4x4_metropolis.ini"]
    (wolff_mediator, _, samplers, sample_directories_4x4_wolff, temperatures, number_of_equilibration_iterations,
     number_of_observations, _, _) = helper_methods.get_basic_config_data(config_file_4x4_wolff)
    metrop_mediator = helper_methods.get_basic_config_data(config_file_4x4_metrop)[0]
    output_directory = sample_directories_4x4_wolff[0].replace("/4x4_wolff", "")
    sample_directories_wolff = [f"{output_directory}/{length}x{length}_wolff" for length in lattice_lengths]
    sample_directories_metrop = [f"{output_directory}/{length}x{length}_metropolis" for length in lattice_lengths]
    transition_temperature = 2.0 / math.log(1 + 2 ** 0.5)
    temperatures = np.array(temperatures)
    temperature_near_critical_point = np.min(np.delete(temperatures, np.where(temperatures < transition_temperature)))
    temperature_near_critical_point_index = np.where(temperatures == temperature_near_critical_point)[0][0]
    acf_temperatures_indices = [
        index for index, _ in enumerate(temperatures)
        if temperature_near_critical_point_index - 4 < index < temperature_near_critical_point_index + 4]
    reduced_temperatures = [temperature / transition_temperature for temperature in temperatures]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0))
    fig.tight_layout(w_pad=5.0)
    [axis.spines[spine].set_linewidth(3) for spine in ["top", "bottom", "left", "right"] for axis in axes]
    for axis in axes:
        axis.tick_params(which='both', direction='in', width=3)
        axis.tick_params(which='major', length=5, labelsize=18, pad=5)
        axis.tick_params(which='minor', length=4)
    axes[0].set_xlabel(r"$s$", fontsize=20, labelpad=3)
    axes[0].set_ylabel(r"$K_m(s)$ / $K_m(s = 0)$", fontsize=20, labelpad=1)
    axes[0].set_yscale('log')
    axes[1].set_xlabel(r"$\beta_{\rm c} / \beta$", fontsize=20, labelpad=3)
    axes[1].set_ylabel(r"$\tau_m^{\rm M/W}$", fontsize=20, labelpad=1)
    axes[0].set_xlim([-1.0, 51.0]), axes[0].set_ylim([0.09, 1.1])  # 0.049787068368 ~= e^(-3)
    # axes[0].text(1.63, 2.01, "(a)", fontsize=20), axes_2[1].text(1.63, 0.96, "(b)", fontsize=20)

    system_size_colors = ["black", "red", "blue", "green", "yellow", "cyan", "magenta"][:number_of_system_sizes]
    system_size_colors.reverse()
    temperature_colors = ["black", "red", "blue", "green", "yellow", "cyan", "magenta"]
    temperature_colors = list(islice(cycle(temperature_colors), len(temperatures)))

    for lattice_length_index, lattice_length in enumerate(lattice_lengths):
        _ = get_observable_autocorrelation_vs_temperature(
            "magnetic_density", metrop_mediator, output_directory, sample_directories_metrop[lattice_length_index],
            temperatures, lattice_length ** 2, number_of_equilibration_iterations)
        magnetic_norm_density_acf_vs_temp_metrop = get_observable_autocorrelation_vs_temperature(
            "magnetic_norm_density", metrop_mediator, output_directory,
            sample_directories_metrop[lattice_length_index], temperatures, lattice_length ** 2,
            number_of_equilibration_iterations)
        _ = get_observable_autocorrelation_vs_temperature(
            "potential", metrop_mediator, output_directory, sample_directories_metrop[lattice_length_index],
            temperatures, lattice_length ** 2, number_of_equilibration_iterations)
        _ = get_observable_autocorrelation_vs_temperature(
            "magnetic_density", wolff_mediator, output_directory, sample_directories_wolff[lattice_length_index],
            temperatures, lattice_length ** 2, number_of_equilibration_iterations)
        magnetic_norm_density_acf_vs_temp_wolff = get_observable_autocorrelation_vs_temperature(
            "magnetic_norm_density", wolff_mediator, output_directory,
            sample_directories_wolff[lattice_length_index], temperatures, lattice_length ** 2,
            number_of_equilibration_iterations)
        _ = get_observable_autocorrelation_vs_temperature(
            "potential", wolff_mediator, output_directory, sample_directories_wolff[lattice_length_index],
            temperatures, lattice_length ** 2, number_of_equilibration_iterations)

        magnetic_norm_density_acf_vs_temp_metrop /= magnetic_norm_density_acf_vs_temp_metrop[:, 0, None]
        magnetic_norm_density_acf_vs_temp_wolff /= magnetic_norm_density_acf_vs_temp_wolff[:, 0, None]

        if lattice_length_index == len(lattice_lengths) - 1:
            for temperature_index in acf_temperatures_indices:
                axes[0].plot(magnetic_norm_density_acf_vs_temp_metrop[temperature_index, :100],
                             marker=".", markersize=8, color=temperature_colors[temperature_index], linestyle="--",
                             label=fr"$1 / (\beta J)$ = {temperatures[temperature_index]:.02}")

        correlation_times = []
        for temperature_index, temperature in enumerate(temperatures):
            max_acf_index = min(next(index for index, value in enumerate(
                magnetic_norm_density_acf_vs_temp_metrop[temperature_index]) if value < 0.1), 2)
            if magnetic_norm_density_acf_vs_temp_metrop[temperature_index, max_acf_index] < 0.0:
                max_acf_index = 1
            correlation_times.append(
                - max_acf_index / np.log(magnetic_norm_density_acf_vs_temp_metrop[temperature_index, max_acf_index]))
            '''exponential_fit = np.polyfit(np.arange(max_acf_index),
            np.log(magnetic_norm_density_acf_vs_temp_metrop[temperature_index, :max_acf_index]), 1)
            print(exponential_fit)'''

        axes[1].plot(reduced_temperatures, correlation_times, marker=".", markersize=8,
                     color=system_size_colors[lattice_length_index], linestyle="None",
                     label=fr"$N$ = {lattice_length}x{lattice_length}")

    legends = [axes[0].legend(loc="upper right", fontsize=12), axes[1].legend(loc="upper left", fontsize=12)]
    [legend.get_frame().set_edgecolor("k") for legend in legends]
    [legend.get_frame().set_lw(3) for legend in legends]
    fig.savefig(f"{output_directory}/2d_ising_model_magnetic_fluctuation_acf.pdf", bbox_inches="tight")


def get_observable_autocorrelation_vs_temperature(observable_string, config_file_mediator, output_directory,
                                                  sample_directory, temperatures, lattice_length,
                                                  number_of_equilibration_iterations, thinning_level=None):
    try:
        return np.load(
            f"{output_directory}/{lattice_length}x{lattice_length}_zero_field_ising_model_{observable_string}_"
            f"autocorrelation_vs_temperature_{config_file_mediator.replace('_mediator', '')}_algorithm.npy")
    except IOError:
        get_sample_method = getattr(sample_getter, "get_" + observable_string)
        acf_vs_temperature = np.array([get_autocorrelation(get_sample_method(
            sample_directory, temperature, temperature_index, lattice_length ** 2, number_of_equilibration_iterations,
            thinning_level)) for temperature_index, temperature in enumerate(temperatures)])
        np.save(f"{output_directory}/{lattice_length}x{lattice_length}_zero_field_ising_model_{observable_string}_"
                f"autocorrelation_vs_temperature_{config_file_mediator.replace('_mediator', '')}_algorithm.npy",
                acf_vs_temperature)
    return acf_vs_temperature


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
