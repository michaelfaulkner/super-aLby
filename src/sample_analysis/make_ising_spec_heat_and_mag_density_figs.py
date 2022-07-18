from markov_chain_diagnostics import get_sample_mean_and_error
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


def main(number_of_system_sizes=3):
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    lattice_lengths = [2 ** (index + 2) for index in range(number_of_system_sizes)]
    # lattice_lengths = [4 + index * 4 for index in range(number_of_system_sizes)]
    config_file_4x4 = ["config_files/sampling_algos/ising_figures/4x4_wolff.ini"]
    (mediator, _, samplers, sample_directories_4x4_wolff, temperatures, number_of_equilibration_iterations,
     number_of_observations, _, _) = helper_methods.get_basic_config_data(config_file_4x4)
    output_directory = sample_directories_4x4_wolff[0].replace("/4x4_wolff", "")
    sample_directories = [f"{output_directory}/{length}x{length}_wolff" for length in lattice_lengths]
    transition_temperature = 2.0 / math.log(1 + 2 ** 0.5)
    reduced_temperatures = [temperature / transition_temperature for temperature in temperatures]

    fig_1, axis_1 = plt.subplots(1, figsize=(6.25, 4.0))
    fig_1.tight_layout()
    additional_y_axis = axis_1.twinx()  # add a twinned y axis to the right-hand subplot
    [axis_1.spines[spine].set_linewidth(3) for spine in ["top", "bottom", "left", "right"]]
    axis_1.tick_params(which='both', direction='in', width=3)
    axis_1.tick_params(which='major', length=5, labelsize=18, pad=5)
    axis_1.tick_params(which='minor', length=4)
    additional_y_axis.tick_params(which='both', direction='in', width=3, colors='red')
    additional_y_axis.tick_params(which='major', length=5, labelsize=18, pad=5)
    additional_y_axis.tick_params(which='minor', length=4)
    additional_y_axis.tick_params(axis='y', labelcolor='red')
    additional_y_axis.spines["right"].set_color("red"), additional_y_axis.spines["right"].set_linewidth(3)
    axis_1.set_xlabel(r"$\beta_{\rm c} / \beta$", fontsize=20, labelpad=3)
    axis_1.set_ylabel(r"${\rm lim}_{N \to \infty} \left[ \mathbb{E} C_{\rm V} \right.$ / $\left. N \right]$",
                      fontsize=20, labelpad=1)
    additional_y_axis.set_ylabel(r"$m_0$", fontsize=20, labelpad=1, color="red")
    axis_1.set_xlim([0.4, 1.625]), axis_1.set_ylim([-0.05, 2.2]), additional_y_axis.set_ylim([-0.025, 1.05])

    fig_2, axes_2 = plt.subplots(1, 2, figsize=(12.5, 4.0))
    fig_2.tight_layout(w_pad=5.0)
    [axis.spines[spine].set_linewidth(3) for spine in ["top", "bottom", "left", "right"] for axis in axes_2]
    for axis in axes_2:
        axis.tick_params(which='both', direction='in', width=3)
        axis.tick_params(which='major', length=5, labelsize=18, pad=5)
        axis.tick_params(which='minor', length=4)
    [axis.set_xlabel(r"$\beta_{\rm c} / \beta$", fontsize=20, labelpad=3) for axis in axes_2]
    axes_2[0].set_ylabel(r"$\mathbb{E} C_{\rm V}$ / $N$", fontsize=20, labelpad=1)
    axes_2[1].set_ylabel(r"$\mathbb{E} {|m|}$", fontsize=20, labelpad=1)
    axes_2[0].set_xlim([0.4, 1.625]), axes_2[0].set_ylim([-0.05, 2.2])
    axes_2[1].set_xlim([0.4, 1.625]), axes_2[1].set_ylim([-0.025, 1.05])
    axes_2[0].text(1.525, 2.01, "(a)", fontsize=20), axes_2[1].text(1.525, 0.96, "(b)", fontsize=20)

    system_size_colors = ["black", "red", "blue", "green", "yellow", "cyan", "magenta"][:number_of_system_sizes]
    system_size_colors.reverse()

    """plot analytical solutions"""
    (continuous_temperatures,
     onsager_specific_heat_density) = get_onsager_specific_heat_temperatures_and_density(output_directory)
    axis_1.plot(continuous_temperatures / transition_temperature, onsager_specific_heat_density, color="black",
                linestyle="-", linewidth=2.0)
    axes_2[0].plot(continuous_temperatures / transition_temperature, onsager_specific_heat_density,
                   color="black", linestyle="-", linewidth=2.0, label=r"$N \to \infty$")
    continuous_temperatures = np.linspace(temperatures[0] - 0.2, temperatures[-1] + 0.2, 800)
    onsager_yang_mag_density = np.piecewise(
        continuous_temperatures,
        [continuous_temperatures < transition_temperature, continuous_temperatures > transition_temperature],
        [lambda temperature: (1.0 - 1.0 / np.sinh(2.0 / temperature) ** 4) ** (1 / 8), 0.0])
    additional_y_axis.plot(continuous_temperatures / transition_temperature, onsager_yang_mag_density, color="red",
                           linestyle="-", linewidth=2.0)
    axes_2[1].plot(continuous_temperatures / transition_temperature, onsager_yang_mag_density, color="black",
                   linestyle="-", linewidth=2.0, label=r"$N \to \infty$")
    fig_1.savefig(f"{output_directory}/2d_ising_model_spec_heat_and_mag_norm_density_vs_temperature_analytical_"
                  f"solutions.pdf", bbox_inches="tight")

    for lattice_length_index, lattice_length in enumerate(lattice_lengths):
        _, _ = get_observable_mean_and_error_vs_temperature(
            "magnetic_density", mediator, output_directory, sample_directories[lattice_length_index], temperatures,
            lattice_length ** 2, number_of_equilibration_iterations)
        (magnetic_norm_density_vs_temp,
         magnetic_norm_density_errors_vs_temp) = get_observable_mean_and_error_vs_temperature(
            "magnetic_norm_density", mediator, output_directory, sample_directories[lattice_length_index], temperatures,
            lattice_length ** 2, number_of_equilibration_iterations)
        _, _ = get_observable_mean_and_error_vs_temperature(
            "magnetic_susceptibility", mediator, output_directory, sample_directories[lattice_length_index],
            temperatures, lattice_length ** 2, number_of_equilibration_iterations)
        _, _ = get_observable_mean_and_error_vs_temperature(
            "magnetic_norm_susceptibility", mediator, output_directory, sample_directories[lattice_length_index],
            temperatures, lattice_length ** 2, number_of_equilibration_iterations)
        _, _ = get_observable_mean_and_error_vs_temperature(
            "potential", mediator, output_directory, sample_directories[lattice_length_index], temperatures,
            lattice_length ** 2, number_of_equilibration_iterations)
        (specific_heat_vs_temp, specific_heat_errors_vs_temp) = get_observable_mean_and_error_vs_temperature(
            "specific_heat", mediator, output_directory, sample_directories[lattice_length_index], temperatures,
            lattice_length ** 2, number_of_equilibration_iterations)
        axes_2[0].errorbar(reduced_temperatures, specific_heat_vs_temp / lattice_length ** 2,
                           specific_heat_errors_vs_temp / lattice_length ** 2, marker=".", markersize=8,
                           color=system_size_colors[lattice_length_index], linestyle="None",
                           label=fr"$N$ = {lattice_length}x{lattice_length}")
        axes_2[1].errorbar(reduced_temperatures, magnetic_norm_density_vs_temp, magnetic_norm_density_errors_vs_temp,
                           marker=".", markersize=8, color=system_size_colors[lattice_length_index], linestyle="None",
                           label=fr"$N$ = {lattice_length}x{lattice_length}")

    legends = [axes_2[0].legend(loc="upper left", fontsize=12), axes_2[1].legend(loc="lower left", fontsize=12)]
    [legend.get_frame().set_edgecolor("k") for legend in legends]
    [legend.get_frame().set_lw(3) for legend in legends]
    fig_2.savefig(f"{output_directory}/2d_ising_model_spec_heat_and_mag_norm_density_vs_temperature_"
                  f"{mediator.replace('_mediator', '')}_simulations.pdf", bbox_inches="tight")


def get_onsager_specific_heat_temperatures_and_density(output_directory, no_of_temperature_integration_values=1000,
                                                       no_of_x_integration_values=1000, max_temperature=4.0):
    try:
        return np.load(f"{output_directory}/2d_ising_model_onsager_specific_heat_density_vs_temperature_"
                       f"{no_of_temperature_integration_values}_temp_values_{no_of_x_integration_values}_x_values.npy")
    except IOError:
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
        np.save(f"{output_directory}/2d_ising_model_onsager_free_energy_density_vs_temperature_"
                f"{no_of_temperature_integration_values}_temp_values_{no_of_x_integration_values}_x_values.npy",
                np.array([temperatures, free_energy_density]))
        np.save(f"{output_directory}/2d_ising_model_onsager_expected_potential_density_vs_temperature_"
                f"{no_of_temperature_integration_values}_temp_values_{no_of_x_integration_values}_x_values.npy",
                np.array([temperatures, expected_potential_density]))
        np.save(f"{output_directory}/2d_ising_model_onsager_specific_heat_density_vs_temperature_"
                f"{no_of_temperature_integration_values}_temp_values_{no_of_x_integration_values}_x_values.npy",
                np.array([specific_heat_temperatures, specific_heat_density]))
        return np.array([specific_heat_temperatures, specific_heat_density])


def get_observable_mean_and_error_vs_temperature(observable_string, config_file_mediator, output_directory,
                                                 sample_directory, temperatures, number_of_particles,
                                                 number_of_equilibration_iterations, thinning_level=None):
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
