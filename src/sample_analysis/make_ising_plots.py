from markov_chain_diagnostics import get_sample_mean_and_error
from matplotlib.lines import Line2D
import importlib
import math
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
    get_onsager_specific_heat_density()

    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    lattice_lengths = [4 + index * 4 for index in range(number_of_system_sizes)]
    config_file_4x4_wolff = ["config_files/sampling_algos/ising_figures/4x4_wolff.ini"]
    (config_file_mediator, _, samplers, sample_directories_4x4_wolff, temperatures, number_of_equilibration_iterations,
     number_of_observations, _, _) = helper_methods.get_basic_config_data(config_file_4x4_wolff)
    output_directory = sample_directories_4x4_wolff[0].replace("/4x4_wolff", "")
    sample_directories_wolff = [f"{output_directory}/{length}x{length}_wolff" for length in lattice_lengths]
    sample_directories_metrop = [f"{output_directory}/{length}x{length}_metropolis" for length in lattice_lengths]
    '''if config_file_mediator == "metropolis_mediator":
        thinning_level = 10
    else:
        thinning_level = None'''
    thinning_level = None

    transition_temperature = 2.0 / math.log(1 + 2 ** 0.5)
    reduced_temperatures = [temperature / transition_temperature for temperature in temperatures]

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.0))
    figure.tight_layout(w_pad=5.0)
    [axis.spines[spine].set_linewidth(3) for spine in ["top", "bottom", "left", "right"] for axis in axes]
    for axis in axes:
        axis.tick_params(which='both', direction='in', width=3)
        axis.tick_params(which='major', length=7, labelsize=18, pad=5)
        axis.tick_params(which='minor', length=4)
    [axis.set_xlabel(r"$\beta_{\rm c} / \beta$", fontsize=20, labelpad=3) for axis in axes]
    axes[0].set_ylabel(r"$\mathbb{E} C_{\rm V}$ / $N^{1 / 2}$", fontsize=20, labelpad=1)
    axes[1].set_ylabel(r"$\mathbb{E} {|m|}$", fontsize=20, labelpad=1)
    axes[1].set_ylim([0.0, 1.05])
    axes[0].text(1.63, 28.0, "(a)", fontsize=20), axes[1].text(1.63, 0.96, "(b)", fontsize=20)
    inset_axis = plt.axes([0.33, 0.575, 0.125, 0.275])
    inset_axis.tick_params(which='both', direction='in', length=4, width=2, labelsize=12)
    inset_axis.set_xlim([0.86, 1.16])
    [inset_axis.spines[spine].set_linewidth(3) for spine in ["top", "bottom", "left", "right"]]
    inset_axis.set_xlabel(r"$\beta_{\rm c} / \beta$", fontsize=12, labelpad=3)
    inset_axis.set_ylabel(r"$\mathbb{E} C_{\rm V}$ / $N$", fontsize=12, labelpad=1)
    colors = ["black", "red", "blue", "green", "yellow", "cyan", "magenta"][:number_of_system_sizes]
    colors.reverse()

    """plot analytical solutions"""
    continuous_temperatures = np.linspace(0.9 * transition_temperature, 1.1 * transition_temperature, 100)
    onsager_specific_heat = - np.log(np.abs(continuous_temperatures - transition_temperature)) - 0.5
    inset_axis.plot(continuous_temperatures / transition_temperature, onsager_specific_heat, color="black",
                    linestyle="-", label="Onsager")
    continuous_temperatures = np.linspace(temperatures[0], temperatures[-1], 400)
    onsager_yang_mag_density = np.piecewise(
        continuous_temperatures,
        [continuous_temperatures < transition_temperature, continuous_temperatures > transition_temperature],
        [lambda temperature: (1.0 - 1.0 / np.sinh(2.0 / temperature) ** 4) ** (1 / 8), 0.0])
    axes[1].plot(continuous_temperatures / transition_temperature, onsager_yang_mag_density, color="black",
                 linestyle="-", label="Onsager-Yang")

    for lattice_length_index, lattice_length in enumerate(lattice_lengths):
        _, _ = get_observable_mean_and_error_vs_temperature(
            "magnetic_density", config_file_mediator, output_directory,
            sample_directories_wolff[lattice_length_index], temperatures, lattice_length ** 2,
            number_of_equilibration_iterations, thinning_level)
        (magnetic_norm_density_vs_temp,
         magnetic_norm_density_errors_vs_temp) = get_observable_mean_and_error_vs_temperature(
            "magnetic_norm_density", config_file_mediator, output_directory,
            sample_directories_wolff[lattice_length_index], temperatures, lattice_length ** 2,
            number_of_equilibration_iterations, thinning_level)
        _, _ = get_observable_mean_and_error_vs_temperature(
            "magnetic_susceptibility", config_file_mediator, output_directory,
            sample_directories_wolff[lattice_length_index], temperatures, lattice_length ** 2,
            number_of_equilibration_iterations, thinning_level)
        _, _ = get_observable_mean_and_error_vs_temperature(
            "magnetic_norm_susceptibility", config_file_mediator, output_directory,
            sample_directories_wolff[lattice_length_index], temperatures, lattice_length ** 2,
            number_of_equilibration_iterations, thinning_level)
        _, _ = get_observable_mean_and_error_vs_temperature(
            "potential", config_file_mediator, output_directory, sample_directories_wolff[lattice_length_index],
            temperatures, lattice_length ** 2, number_of_equilibration_iterations, thinning_level)
        (specific_heat_vs_temp, specific_heat_errors_vs_temp) = get_observable_mean_and_error_vs_temperature(
            "specific_heat", config_file_mediator, output_directory,
            sample_directories_wolff[lattice_length_index], temperatures, lattice_length ** 2,
            number_of_equilibration_iterations, thinning_level)
        axes[0].errorbar(reduced_temperatures, specific_heat_vs_temp / lattice_length,
                         specific_heat_errors_vs_temp / lattice_length, marker=".", markersize=8,
                         color=colors[lattice_length_index], linestyle="None",
                         label=fr"$N$ = {lattice_length}x{lattice_length}")
        inset_axis.errorbar(reduced_temperatures, specific_heat_vs_temp / lattice_length ** 2,
                            specific_heat_errors_vs_temp / lattice_length ** 2, marker=".", markersize=8,
                            color=colors[lattice_length_index], linestyle="None")
        axes[1].errorbar(reduced_temperatures, magnetic_norm_density_vs_temp, magnetic_norm_density_errors_vs_temp,
                         marker=".", markersize=8, color=colors[lattice_length_index], linestyle="None",
                         label=fr"$N$ = {lattice_length}x{lattice_length}")

        #if lattice_length_index == lattice_lengths[-1]:
        if lattice_length_index == 0:
            figure_trace_plot, axes_trace_plot = plt.subplots(2, 2, figsize=(12.5, 5.0))
            figure_trace_plot.tight_layout(h_pad=-1.0, w_pad=-2.0)
            subfigure_labels = ["(a)", "(b)", "(c)", "(d)"]
            [axis.text(149.0, 0.86, subfigure_labels[axis_index], fontsize=18) for axis_index, axis in
             enumerate(axes_trace_plot.flatten())]
            [axis.spines[spine].set_linewidth(3) for spine in ["top", "bottom", "left", "right"] for axis in
             axes_trace_plot.flatten()]
            for axis in axes_trace_plot.flatten():
                axis.tick_params(which='both', direction='in', width=3)
                axis.tick_params(which='major', length=7, labelsize=18, pad=5)
                axis.tick_params(which='minor', length=4)
            [axes_trace_plot[0, i].tick_params(labelbottom=False) for i in range(2)]
            axes_trace_plot[1, 0].set_xlabel(r"Metropolis observation, $t$", fontsize=18, labelpad=3)
            axes_trace_plot[1, 1].set_xlabel(r"Wolff observation, $t$", fontsize=18, labelpad=3)
            [axis.set_xlim([-5, 160]) for axis in axes_trace_plot.flatten()]
            [axes_trace_plot[i, 1].tick_params(labelleft=False) for i in range(2)]
            axes_trace_plot[0, 0].set_ylabel(r"$m\left( x(t); \beta J = 1 \right)$", fontsize=18, labelpad=3)
            axes_trace_plot[1, 0].set_ylabel(r"$m\left( x(t); \beta J = 1 \, / \, 2.4 \right)$", fontsize=18,
                                             labelpad=3)
            [axis.set_ylim([-1.15, 1.15]) for axis in axes_trace_plot.flatten()]
            plot_magnetic_density(
                axes_trace_plot[0, 0], "metropolis", output_directory, sample_directories_metrop[lattice_length_index],
                temperatures[0], 0, lattice_lengths[lattice_length_index], number_of_equilibration_iterations)
            plot_magnetic_density(
                axes_trace_plot[0, 1], "wolff", output_directory, sample_directories_wolff[lattice_length_index],
                temperatures[0], 0, lattice_lengths[lattice_length_index], number_of_equilibration_iterations)
            plot_magnetic_density(
                axes_trace_plot[1, 0], "metropolis", output_directory, sample_directories_metrop[lattice_length_index],
                temperatures[int(len(temperatures) / 2)], int(len(temperatures) / 2),
                lattice_lengths[lattice_length_index], number_of_equilibration_iterations)
            plot_magnetic_density(
                axes_trace_plot[1, 1], "wolff", output_directory, sample_directories_wolff[lattice_length_index],
                temperatures[int(len(temperatures) / 2)], int(len(temperatures) / 2),
                lattice_lengths[lattice_length_index], number_of_equilibration_iterations)
            figure_trace_plot.savefig(
                f"{output_directory}/{lattice_lengths[lattice_length_index]}x{lattice_lengths[lattice_length_index]}_"
                f"zero_field_ising_model_mag_density_vs_time_metrop_and_wolff.pdf", bbox_inches="tight")

    current_handles, _ = axes[0].get_legend_handles_labels()
    current_handles.insert(0, Line2D([0], [0], color="black", label="Onsager"))
    legends = [axes[0].legend(handles=current_handles, loc="upper left", fontsize=12),
               axes[1].legend(loc="lower left", fontsize=12)]
    [legend.get_frame().set_edgecolor("k") for legend in legends]
    [legend.get_frame().set_lw(3) for legend in legends]
    figure.savefig(f"{output_directory}/2d_ising_model_spec_heat_and_mag_norm_density_vs_temperature_"
                   f"{config_file_mediator.replace('_mediator', '')}.pdf", bbox_inches="tight")


def get_onsager_specific_heat_density(max_temperature=4.0, no_of_temperature_integration_values=1000,
                                      no_of_x_integration_values=1000):
    delta_temperature = max_temperature / no_of_temperature_integration_values
    temperatures = np.linspace(delta_temperature, max_temperature, no_of_temperature_integration_values,
                               dtype=np.float128)
    inverse_temperatures = 1.0 / temperatures
    alpha = 2.0 * np.sinh(2.0 * inverse_temperatures) / np.cosh(2.0 * inverse_temperatures) ** 2
    dalpha_dbeta = 4.0 * (
            1.0 - 2.0 * np.tanh(2.0 * inverse_temperatures) ** 2) / np.cosh(2.0 * inverse_temperatures)
    gamma_2, dgamma_2_dbeta = np.zeros(len(temperatures)), np.zeros(len(temperatures))
    x = np.linspace(0.0, math.pi / 2.0, no_of_x_integration_values, dtype=np.float128)
    for temperature_index in range(len(temperatures)):
        gamma_2_integrand = np.log(0.5 * (1.0 + np.sqrt(1.0 - alpha[temperature_index] ** 2 * np.sin(x) ** 2)))
        gamma_2[temperature_index] = np.trapz(gamma_2_integrand, x=x) / math.pi
        dgamma_2_dbeta_integrand = - alpha[temperature_index] * dalpha_dbeta[temperature_index] * np.sin(x) ** 2 / (
                1.0 - alpha[temperature_index] ** 2 * np.sin(x) ** 2 + np.sqrt(1.0 - alpha[temperature_index] ** 2 *
                                                                               np.sin(x) ** 2))
        dgamma_2_dbeta[temperature_index] = np.trapz(dgamma_2_dbeta_integrand, x=x) / math.pi
    free_energy_density = - temperatures * (np.log(2.0 * np.cosh(2.0 * inverse_temperatures)) + gamma_2)
    expected_potential_density = - 2.0 * np.tanh(2.0 * inverse_temperatures) - dgamma_2_dbeta
    specific_heat_density = np.diff(expected_potential_density) / delta_temperature
    specific_heat_temperatures = temperatures[:len(temperatures) - 1]
    '''plt.plot(temperatures, free_energy_density, color='k', linewidth=2, linestyle='-', label='free energy density')
    plt.plot(temperatures, expected_potential_density, color='r', linewidth=2, linestyle='-',
             label='expected potential density')
    plt.plot(specific_heat_temperatures, specific_heat_density, color='b', linewidth=2, linestyle='-',
             label='specific-heat density')'''
    '''transition_temperature = 2.0 / math.log(1 + 2 ** 0.5)
    specific_heat_temperatures_lower = np.delete(specific_heat_temperatures, np.argwhere(specific_heat_temperatures >
                                                                                         transition_temperature))
    specific_heat_temperatures_higher = np.delete(specific_heat_temperatures, np.argwhere(specific_heat_temperatures <
                                                                                          transition_temperature))
    specific_heat_density_lower = np.delete(specific_heat_density, np.argwhere(specific_heat_temperatures >
                                                                               transition_temperature))
    specific_heat_density_higher = np.delete(specific_heat_density, np.argwhere(specific_heat_temperatures <
                                                                                transition_temperature))
    plt.plot(specific_heat_temperatures_lower, specific_heat_density_lower, color='b', linewidth=2, linestyle='-',
             label='specific-heat density')
    plt.plot(specific_heat_temperatures_higher, specific_heat_density_higher, color='b', linewidth=2, linestyle='-')'''

    '''plt.xlabel(r"$1 / (\beta J)$", fontsize=15, labelpad=10)
    plt.tick_params(axis='both', which='major', direction='in', labelsize=14, pad=10)
    legend = plt.legend(loc='lower left', fontsize=10)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)
    plt.tight_layout()
    plt.show()'''


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
            sample = get_sample_method(sample_directory, temperature, temperature_index, number_of_particles,
                                       number_of_equilibration_iterations, thinning_level)
            acf = np.correlate(sample, sample)
            output_file.write(
                f"{temperature:.14e}".ljust(30) + f"{sample_mean:.14e}".ljust(35) + f"{sample_error:.14e}" + "\n")
            means_vs_temperature.append(sample_mean)
            errors_vs_temperature.append(sample_error)
        output_file.close()
    return np.array([means_vs_temperature, errors_vs_temperature])


def plot_magnetic_density(axis, algorithm_string, output_directory, sample_directory, temperature, temperature_index,
                          lattice_length, number_of_equilibration_iterations):
    try:
        reduced_magnetic_density_sample = np.load(
            f"{output_directory}/{lattice_length}x{lattice_length}_zero_field_ising_model_temperature_"
            f"{temperature_index:02d}_{algorithm_string}_algorithm_reduced_magnetic_density_sample.npy")
    except IOError:
        reduced_magnetic_density_sample = sample_getter.get_magnetic_density(
            sample_directory, temperature, temperature_index, lattice_length ** 2,
            number_of_equilibration_iterations)[:150].flatten()
        np.save(f"{output_directory}/{lattice_length}x{lattice_length}_zero_field_ising_model_temperature_"
                f"{temperature_index:02d}_{algorithm_string}_algorithm_reduced_magnetic_density_sample.npy",
                reduced_magnetic_density_sample)
    axis.plot(reduced_magnetic_density_sample, color="k", linewidth=1, linestyle="-")


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
