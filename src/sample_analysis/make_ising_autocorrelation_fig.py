from markov_chain_diagnostics import get_autocorrelation, get_integrated_autocorrelation_time
import importlib
import math
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp
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
    lattice_lengths = [2 ** (index + 2) for index in range(number_of_system_sizes)]
    config_file_4x4_wolff = ["config_files/sampling_algos_ising_figs/4x4_wolff.ini"]
    config_file_4x4_metrop = ["config_files/sampling_algos_ising_figs/4x4_metropolis.ini"]
    (wolff_mediator, _, samplers, sample_directories_4x4_wolff, temperatures, number_of_equilibration_iterations,
     number_of_observations, _, _, number_of_jobs, max_number_of_cpus) = helper_methods.get_basic_config_data(
        config_file_4x4_wolff)
    metrop_mediator = helper_methods.get_basic_config_data(config_file_4x4_metrop)[0]
    output_directory = sample_directories_4x4_wolff[0].replace("/4x4_wolff", "")
    sample_directories_wolff = [f"{output_directory}/{length}x{length}_wolff" for length in lattice_lengths]
    sample_directories_metrop = [f"{output_directory}/{length}x{length}_metropolis" for length in lattice_lengths]
    transition_temperature = 2.0 / math.log(1 + 2 ** 0.5)
    temperatures = temperatures
    reduced_temperatures = [temperature / transition_temperature for temperature in temperatures]

    fig_1, axis_1 = make_empty_fig()
    fig_2, axis_2 = make_empty_fig()
    system_size_colors = ["red", "blue", "green", "magenta", "indigo", "tab:brown"][:number_of_system_sizes]
    system_size_colors.reverse()

    if number_of_jobs > 1:
        number_of_cpus = mp.cpu_count()
        pool = mp.Pool(min(number_of_cpus, max_number_of_cpus))
    else:
        pool = None

    for lattice_length_index, lattice_length in enumerate(lattice_lengths):
        _ = get_observable_autocorrelation_vs_temperature(
            "magnetic_density", metrop_mediator, output_directory, sample_directories_metrop[lattice_length_index],
            temperatures, lattice_length, number_of_equilibration_iterations, number_of_observations, number_of_jobs,
            pool)
        magnetic_norm_density_acf_vs_temp_metrop = get_observable_autocorrelation_vs_temperature(
            "magnetic_norm_density", metrop_mediator, output_directory, sample_directories_metrop[lattice_length_index],
            temperatures, lattice_length, number_of_equilibration_iterations, number_of_observations, number_of_jobs,
            pool)
        _ = get_observable_autocorrelation_vs_temperature(
            "potential", metrop_mediator, output_directory, sample_directories_metrop[lattice_length_index],
            temperatures, lattice_length, number_of_equilibration_iterations, number_of_observations, number_of_jobs,
            pool)
        _ = get_observable_autocorrelation_vs_temperature(
            "magnetic_density", wolff_mediator, output_directory, sample_directories_wolff[lattice_length_index],
            temperatures, lattice_length, number_of_equilibration_iterations, number_of_observations, number_of_jobs,
            pool)
        magnetic_norm_density_acf_vs_temp_wolff = get_observable_autocorrelation_vs_temperature(
            "magnetic_norm_density", wolff_mediator, output_directory, sample_directories_wolff[lattice_length_index],
            temperatures, lattice_length, number_of_equilibration_iterations, number_of_observations, number_of_jobs,
            pool)
        _ = get_observable_autocorrelation_vs_temperature(
            "potential", wolff_mediator, output_directory, sample_directories_wolff[lattice_length_index],
            temperatures, lattice_length, number_of_equilibration_iterations, number_of_observations, number_of_jobs,
            pool)

        magnetic_norm_density_acf_vs_temp_metrop /= magnetic_norm_density_acf_vs_temp_metrop[0, :, 0, None]
        magnetic_norm_density_acf_vs_temp_wolff /= magnetic_norm_density_acf_vs_temp_wolff[0, :, 0, None]
        metrop_integrated_autocorrelation_times = get_magnetic_norm_integrated_autocorrelation_times_vs_temperature(
            magnetic_norm_density_acf_vs_temp_metrop[0], metrop_mediator, output_directory, temperatures,
            lattice_length, number_of_observations, number_of_jobs)
        wolff_integrated_autocorrelation_times = get_magnetic_norm_integrated_autocorrelation_times_vs_temperature(
            magnetic_norm_density_acf_vs_temp_wolff[0], wolff_mediator, output_directory, temperatures,
            lattice_length, number_of_observations, number_of_jobs)
        axis_1.plot(reduced_temperatures, wolff_integrated_autocorrelation_times, marker="*", markersize=8,
                    color=system_size_colors[lattice_length_index], linestyle="--",
                    label=fr"$N$ = {lattice_length}x{lattice_length} Wolff")
        axis_1.plot(reduced_temperatures, metrop_integrated_autocorrelation_times, marker=".", markersize=8,
                    color=system_size_colors[lattice_length_index], linestyle="-",
                    label=fr"$N$ = {lattice_length}x{lattice_length} Metrop")
        axis_2.plot(reduced_temperatures, metrop_integrated_autocorrelation_times, marker=".", markersize=8,
                    color=system_size_colors[lattice_length_index], linestyle="-",
                    label=fr"$N$ = {lattice_length}x{lattice_length} Metrop")

    legends = [axis_1.legend(loc="upper left", fontsize=10), axis_2.legend(loc="upper left", fontsize=10)]
    [legend.get_frame().set_edgecolor("k") for legend in legends], [legend.get_frame().set_lw(3) for legend in legends]
    fig_1.savefig(f"{output_directory}/2d_ising_model_magnetic_norm_integrated_autocorrelation_time_"
                  f"{number_of_jobs}x{number_of_observations}_observations.pdf", bbox_inches="tight")
    fig_2.savefig(f"{output_directory}/2d_ising_model_magnetic_norm_integrated_autocorrelation_time_"
                  f"{number_of_jobs}x{number_of_observations}_observations_metrop_only.pdf", bbox_inches="tight")


def make_empty_fig():
    fig, axis = plt.subplots(1, figsize=(6.25, 4.0))
    fig.tight_layout()
    [axis.spines[spine].set_linewidth(3) for spine in ["top", "bottom", "left", "right"]]
    axis.tick_params(which='both', direction='in', width=3)
    axis.tick_params(which='major', length=5, labelsize=18, pad=5)
    axis.tick_params(which='minor', length=4)
    axis.set_xlabel(r"$\beta_{\rm c} / \beta$", fontsize=20, labelpad=3)
    axis.set_ylabel(r"$\tau_{|m|}$", fontsize=20, labelpad=1)
    # axis.set_ylim([-5.0, 400.0])  # 0.049787068368 ~= e^(-3)
    return fig, axis


def get_observable_autocorrelation_vs_temperature(observable_string, mediator, output_directory,
                                                  sample_directory, temperatures, lattice_length,
                                                  number_of_equilibration_iterations, number_of_observations,
                                                  number_of_jobs, pool, thinning_level=None):
    try:
        return np.load(f"{output_directory}/{lattice_length}x{lattice_length}_ising_model_{observable_string}_"
                       f"autocorrelation_vs_temperature_{mediator.replace('_mediator', '')}_algorithm_{number_of_jobs}x"
                       f"{number_of_observations}_observations.npy")
    except IOError:
        get_sample_method = getattr(sample_getter, "get_" + observable_string)
        acfs, acf_errors = [], []
        for temperature_index, temperature in enumerate(temperatures):
            acf_vs_job = pool.starmap(get_autocorrelation, [[get_sample_method(
                f"{sample_directory}/job_{job_number:02d}", temperature, temperature_index, lattice_length ** 2,
                number_of_equilibration_iterations, thinning_level)] for job_number in range(number_of_jobs)])
            acf, acf_error = np.mean(acf_vs_job, axis=0), np.std(acf_vs_job, axis=0) / number_of_jobs ** 0.5
            acfs.append(acf), acf_errors.append(acf_error)
        acf_vs_temperature = np.array([acfs, acf_errors])
        np.save(f"{output_directory}/{lattice_length}x{lattice_length}_ising_model_{observable_string}_autocorrelation_"
                f"vs_temperature_{mediator.replace('_mediator', '')}_algorithm_{number_of_jobs}x"
                f"{number_of_observations}_observations.npy", acf_vs_temperature)
    return acf_vs_temperature


def get_magnetic_norm_integrated_autocorrelation_times_vs_temperature(autocorrelation_function, mediator,
                                                                      output_directory, temperatures, lattice_length,
                                                                      number_of_observations, number_of_jobs):
    try:
        with open(f"{output_directory}/{lattice_length}x{lattice_length}_ising_model_magnetic_norm_integrated_"
                  f"autocorrelation_times_vs_temperature_{mediator.replace('_mediator', '')}_algorithm_"
                  f"{number_of_jobs}x{number_of_observations}_observations.tsv", "r") as output_file:
            output_file_sans_header = np.array([np.fromstring(line, dtype=float, sep='\t') for line in output_file
                                                if not line.startswith('#')]).transpose()
            integrated_autocorrelation_times_vs_temperature = output_file_sans_header[1]
    except IOError:
        integrated_autocorrelation_times_vs_temperature = [get_integrated_autocorrelation_time(
            autocorrelation_function[temperature_index]) for temperature_index, _ in enumerate(temperatures)]
        output_file = open(f"{output_directory}/{lattice_length}x{lattice_length}_ising_model_magnetic_norm_integrated_"
                           f"autocorrelation_times_vs_temperature_{mediator.replace('_mediator', '')}_algorithm_"
                           f"{number_of_jobs}x{number_of_observations}_observations.tsv", "w")
        output_file.write("# temperature".ljust(30) + "magnetic-norm integrated autocorrelation time".ljust(35) + "\n")
        for temperature_index, temperature in enumerate(temperatures):
            output_file.write(f"{temperature:.14e}".ljust(30) +
                              f"{integrated_autocorrelation_times_vs_temperature[temperature_index]:.14e}".ljust(35) +
                              "\n")
        output_file.close()
    return np.array(integrated_autocorrelation_times_vs_temperature)


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
