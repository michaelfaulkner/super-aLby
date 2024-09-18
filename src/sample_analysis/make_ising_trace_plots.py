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


def main(sampling_algos_paper=True, number_of_system_sizes=5):
    """For 'Sampling algorithms in statistical physics...', set sampling_algos_paper=True;
        for 'Emergent electrostatics in...' paper, set sampling_algos_paper=False."""
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    lattice_lengths = [2 ** (index + 2) for index in range(number_of_system_sizes)]
    if sampling_algos_paper:
        config_file_4x4_cluster = ["config_files/sampling_algos_ising_figs/4x4_wolff.ini"]
    else:
        config_file_4x4_cluster = ["config_files/emergent_electrostatics_ising_figs/4x4_swendsen_wang.ini"]
    config_file_4x4_metrop = ["config_files/sampling_algos_ising_figs/4x4_metropolis.ini"]
    (cluster_mediator, _, samplers, sample_directories_4x4_cluster, temperatures, number_of_equilibration_iterations,
     number_of_observations, _, _, number_of_jobs, max_number_of_cpus) = helper_methods.get_basic_config_data(
        config_file_4x4_cluster)
    metrop_mediator = helper_methods.get_basic_config_data(config_file_4x4_metrop)[0]
    if sampling_algos_paper:
        output_directory = sample_directories_4x4_cluster[0].replace("/4x4_wolff", "")
        sample_directories_cluster = [f"{output_directory}/{length}x{length}_wolff/job_00" for length in
                                      lattice_lengths]
        sample_directories_metrop = [f"{output_directory}/{length}x{length}_metropolis/job_00" for length in
                                     lattice_lengths]
    else:
        output_directory = sample_directories_4x4_cluster[0].replace("/4x4_swendsen_wang", "")
        sample_directories_cluster = [f"{output_directory}/{length}x{length}_swendsen_wang" for length in
                                      lattice_lengths]
        sample_directories_metrop = [f"{output_directory}/{length}x{length}_metropolis" for length in lattice_lengths]

    transition_temperature = 2.0 / math.log(1 + 2 ** 0.5)
    temperatures = np.array(temperatures)
    if sampling_algos_paper:
        temperature_near_critical_point = np.min(np.delete(temperatures,
                                                           np.where(temperatures < transition_temperature)))
        temperature_near_critical_point_index = np.where(temperatures == temperature_near_critical_point)[0][0]
    else:
        temperature_near_critical_point = temperatures[1]
        temperature_near_critical_point_index = 1

    for lattice_length_index, lattice_length in enumerate(lattice_lengths):
        for simple_axis_labels in [True, False]:
            make_low_temp_vs_transition_figs(
                cluster_mediator, metrop_mediator, output_directory, sample_directories_cluster,
                sample_directories_metrop, lattice_length, lattice_length_index, number_of_equilibration_iterations,
                temperatures, temperature_near_critical_point, temperature_near_critical_point_index,
                sampling_algos_paper, simple_axis_labels)
            make_low_vs_high_temp_figs(
                cluster_mediator, metrop_mediator, output_directory, sample_directories_cluster,
                sample_directories_metrop, lattice_length, lattice_length_index, number_of_equilibration_iterations,
                temperatures, sampling_algos_paper, simple_axis_labels)
            make_all_temps_figs(cluster_mediator, metrop_mediator, output_directory, sample_directories_cluster,
                                sample_directories_metrop, lattice_length, lattice_length_index,
                                number_of_equilibration_iterations, temperatures, temperature_near_critical_point,
                                temperature_near_critical_point_index, sampling_algos_paper, simple_axis_labels)


def make_low_temp_vs_transition_figs(cluster_mediator, metrop_mediator, output_directory, sample_directories_cluster,
                                     sample_directories_metrop, lattice_length, lattice_length_index,
                                     number_of_equilibration_iterations, temperatures, temperature_near_critical_point,
                                     temperature_near_critical_point_index, sampling_algos_paper, simple_axis_labels):
    if simple_axis_labels:
        fig, axes = make_empty_two_temperature_figure(r"$m(t; \beta J = 3 \, / \, 7)$", sampling_algos_paper,
                                                      simple_axis_labels)
    else:
        fig, axes = make_empty_two_temperature_figure(r"$m(x(t); \beta J = 3 \, / \, 7)$", sampling_algos_paper,
                                                      simple_axis_labels)
    plot_magnetic_density_vs_time(
        axes[0, 0], metrop_mediator, output_directory, sample_directories_metrop[lattice_length_index],
        temperatures[0], 0, lattice_length, number_of_equilibration_iterations)
    plot_magnetic_density_vs_time(
        axes[0, 1], cluster_mediator, output_directory, sample_directories_cluster[lattice_length_index],
        temperatures[0], 0, lattice_length, number_of_equilibration_iterations)
    plot_magnetic_density_vs_time(
        axes[1, 0], metrop_mediator, output_directory, sample_directories_metrop[lattice_length_index],
        temperature_near_critical_point, temperature_near_critical_point_index, lattice_length,
        number_of_equilibration_iterations)
    plot_magnetic_density_vs_time(
        axes[1, 1], cluster_mediator, output_directory, sample_directories_cluster[lattice_length_index],
        temperature_near_critical_point, temperature_near_critical_point_index, lattice_length,
        number_of_equilibration_iterations)
    if sampling_algos_paper:
        if simple_axis_labels:
            fig.savefig(f"{output_directory}/{lattice_length}x{lattice_length}_ising_model_magnetic_density_vs_time_"
                        f"metropolis_and_wolff_low_temperature_vs_transition_simple_axis_labels.pdf",
                        bbox_inches="tight")
        else:
            fig.savefig(f"{output_directory}/{lattice_length}x{lattice_length}_ising_model_magnetic_density_vs_time_"
                        f"metropolis_and_wolff_low_temperature_vs_transition.pdf", bbox_inches="tight")
    else:
        if simple_axis_labels:
            fig.savefig(f"{output_directory}/{lattice_length}x{lattice_length}_ising_model_magnetic_density_vs_time_"
                        f"metropolis_and_swendsen_wang_low_temperature_vs_transition_simple_axis_labels.pdf",
                        bbox_inches="tight")
        else:
            fig.savefig(f"{output_directory}/{lattice_length}x{lattice_length}_ising_model_magnetic_density_vs_time_"
                        f"metropolis_and_swendsen_wang_low_temperature_vs_transition.pdf", bbox_inches="tight")
    plt.close(fig)


def make_low_vs_high_temp_figs(cluster_mediator, metrop_mediator, output_directory, sample_directories_cluster,
                               sample_directories_metrop, lattice_length, lattice_length_index,
                               number_of_equilibration_iterations, temperatures, sampling_algos_paper,
                               simple_axis_labels):
    if sampling_algos_paper:
        if simple_axis_labels:
            fig, axes = make_empty_two_temperature_figure(r"$m(t; \beta J = 15 \, / \, 41)$", sampling_algos_paper,
                                                          simple_axis_labels)
        else:
            fig, axes = make_empty_two_temperature_figure(r"$m(x(t); \beta J = 15 \, / \, 41)$", sampling_algos_paper,
                                                          simple_axis_labels)
    else:
        if simple_axis_labels:
            fig, axes = make_empty_two_temperature_figure(r"$m(t; \beta J = 3 \, / \, 11)$", sampling_algos_paper,
                                                          simple_axis_labels)
        else:
            fig, axes = make_empty_two_temperature_figure(r"$m(x(t); \beta J = 3 \, / \, 11)$", sampling_algos_paper,
                                                          simple_axis_labels)
    plot_magnetic_density_vs_time(
        axes[0, 0], metrop_mediator, output_directory, sample_directories_metrop[lattice_length_index],
        temperatures[0], 0, lattice_length, number_of_equilibration_iterations)
    plot_magnetic_density_vs_time(
        axes[0, 1], cluster_mediator, output_directory, sample_directories_cluster[lattice_length_index],
        temperatures[0], 0, lattice_length, number_of_equilibration_iterations)
    if sampling_algos_paper:
        plot_magnetic_density_vs_time(
            axes[1, 0], metrop_mediator, output_directory, sample_directories_metrop[lattice_length_index],
            temperatures[26], 26, lattice_length, number_of_equilibration_iterations)
        plot_magnetic_density_vs_time(
            axes[1, 1], cluster_mediator, output_directory, sample_directories_cluster[lattice_length_index],
            temperatures[26], 26, lattice_length, number_of_equilibration_iterations)
        if simple_axis_labels:
            fig.savefig(f"{output_directory}/{lattice_length}x{lattice_length}_ising_model_magnetic_density_vs_time_"
                        f"metropolis_and_wolff_low_vs_high_temperature_simple_axis_labels.pdf", bbox_inches="tight")
        else:
            fig.savefig(f"{output_directory}/{lattice_length}x{lattice_length}_ising_model_magnetic_density_vs_time_"
                        f"metropolis_and_wolff_low_vs_high_temperature.pdf", bbox_inches="tight")
    else:
        plot_magnetic_density_vs_time(
            axes[1, 0], metrop_mediator, output_directory, sample_directories_metrop[lattice_length_index],
            temperatures[2], 2, lattice_length, number_of_equilibration_iterations)
        plot_magnetic_density_vs_time(
            axes[1, 1], cluster_mediator, output_directory, sample_directories_cluster[lattice_length_index],
            temperatures[2], 2, lattice_length, number_of_equilibration_iterations)
        if simple_axis_labels:
            fig.savefig(f"{output_directory}/{lattice_length}x{lattice_length}_ising_model_magnetic_density_vs_time_"
                        f"metropolis_and_swendsen_wang_low_vs_high_temperature_simple_axis_labels.pdf",
                        bbox_inches="tight")
        else:
            fig.savefig(f"{output_directory}/{lattice_length}x{lattice_length}_ising_model_magnetic_density_vs_time_"
                        f"metropolis_and_swendsen_wang_low_vs_high_temperature.pdf", bbox_inches="tight")
    plt.close(fig)


def make_all_temps_figs(cluster_mediator, metrop_mediator, output_directory, sample_directories_cluster,
                        sample_directories_metrop, lattice_length, lattice_length_index,
                        number_of_equilibration_iterations, temperatures, temperature_near_critical_point,
                        temperature_near_critical_point_index, sampling_algos_paper, simple_axis_labels):
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 7.5))
    fig.tight_layout(h_pad=-1.0, w_pad=-2.0)
    subfigure_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    [axis.text(149.0, 0.86, subfigure_labels[axis_index], fontsize=18) for axis_index, axis in
     enumerate(axes.flatten())]
    [axis.spines[spine].set_linewidth(3) for spine in ["top", "bottom", "left", "right"] for axis in axes.flatten()]
    for axis in axes.flatten():
        axis.tick_params(which='both', direction='in', width=3)
        axis.tick_params(which='major', length=5, labelsize=18, pad=5)
        axis.tick_params(which='minor', length=4)
    [axes[i, j].tick_params(labelbottom=False) for i in range(2) for j in range(2)]
    axes[2, 0].set_xlabel(r"$t / \Delta t_{\rm Metrop}$", fontsize=18, labelpad=3)
    if sampling_algos_paper:
        axes[2, 1].set_xlabel(r"$t / \Delta t_{\rm Wolff}$", fontsize=18, labelpad=3)
    else:
        axes[2, 1].set_xlabel(r"$t / \Delta t_{{\rm Swendsen}\!-\!{\rm Wang}}$", fontsize=18, labelpad=3)
    [axis.set_xlim([-5, 160]) for axis in axes.flatten()]
    [axes[i, 1].tick_params(labelleft=False) for i in range(3)]
    if simple_axis_labels:
        axes[0, 0].set_ylabel(r"$m\left( t; \beta J = 1 \right)$", fontsize=17.5, labelpad=3)
        axes[1, 0].set_ylabel(r"$m(t; \beta J = 3 \, / \, 7)$", fontsize=17.5, labelpad=3)
        axes[2, 0].set_ylabel(r"$m(t; \beta J = 15 \, / \, 41)$", fontsize=17.5, labelpad=3)
    else:
        axes[0, 0].set_ylabel(r"$m\left( x(t); \beta J = 1 \right)$", fontsize=17.5, labelpad=3)
        axes[1, 0].set_ylabel(r"$m(x(t); \beta J = 3 \, / \, 7)$", fontsize=17.5, labelpad=3)
        axes[2, 0].set_ylabel(r"$m(x(t); \beta J = 15 \, / \, 41)$", fontsize=17.5, labelpad=3)
    [axis.set_ylim([-1.15, 1.15]) for axis in axes.flatten()]
    plot_magnetic_density_vs_time(
        axes[0, 0], metrop_mediator, output_directory, sample_directories_metrop[lattice_length_index],
        temperatures[0], 0, lattice_length, number_of_equilibration_iterations)
    plot_magnetic_density_vs_time(
        axes[0, 1], cluster_mediator, output_directory, sample_directories_cluster[lattice_length_index],
        temperatures[0], 0, lattice_length, number_of_equilibration_iterations)
    plot_magnetic_density_vs_time(
        axes[1, 0], metrop_mediator, output_directory, sample_directories_metrop[lattice_length_index],
        temperature_near_critical_point, temperature_near_critical_point_index, lattice_length,
        number_of_equilibration_iterations)
    plot_magnetic_density_vs_time(
        axes[1, 1], cluster_mediator, output_directory, sample_directories_cluster[lattice_length_index],
        temperature_near_critical_point, temperature_near_critical_point_index, lattice_length,
        number_of_equilibration_iterations)
    if sampling_algos_paper:
        plot_magnetic_density_vs_time(
            axes[2, 0], metrop_mediator, output_directory, sample_directories_metrop[lattice_length_index],
            temperatures[26], 26, lattice_length, number_of_equilibration_iterations)
        plot_magnetic_density_vs_time(
            axes[2, 1], cluster_mediator, output_directory, sample_directories_cluster[lattice_length_index],
            temperatures[26], 26, lattice_length, number_of_equilibration_iterations)
        if simple_axis_labels:
            fig.savefig(f"{output_directory}/{lattice_length}x{lattice_length}_ising_model_magnetic_density_vs_time_"
                        f"metropolis_and_wolff_all_temperatures_simple_axis_labels.pdf", bbox_inches="tight")
        else:
            fig.savefig(f"{output_directory}/{lattice_length}x{lattice_length}_ising_model_magnetic_density_vs_time_"
                        f"metropolis_and_wolff_all_temperatures.pdf", bbox_inches="tight")
    else:
        plot_magnetic_density_vs_time(
            axes[2, 0], metrop_mediator, output_directory, sample_directories_metrop[lattice_length_index],
            temperatures[2], 2, lattice_length, number_of_equilibration_iterations)
        plot_magnetic_density_vs_time(
            axes[2, 1], cluster_mediator, output_directory, sample_directories_cluster[lattice_length_index],
            temperatures[2], 2, lattice_length, number_of_equilibration_iterations)
        if simple_axis_labels:
            fig.savefig(f"{output_directory}/{lattice_length}x{lattice_length}_ising_model_magnetic_density_vs_time_"
                        f"metropolis_and_swendsen_wang_all_temperatures_simple_axis_labels.pdf", bbox_inches="tight")
        else:
            fig.savefig(f"{output_directory}/{lattice_length}x{lattice_length}_ising_model_magnetic_density_vs_time_"
                        f"metropolis_and_swendsen_wang_all_temperatures.pdf", bbox_inches="tight")
    plt.close(fig)


def make_empty_two_temperature_figure(higher_temperature_y_axis_label, sampling_algos_paper, simple_axis_labels):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 5.0))
    fig.tight_layout(h_pad=-1.0, w_pad=-2.0)
    subfigure_labels = ["(a)", "(b)", "(c)", "(d)"]
    [axis.text(149.0, 0.86, subfigure_labels[axis_index], fontsize=18) for axis_index, axis in
     enumerate(axes.flatten())]
    [axis.spines[spine].set_linewidth(3) for spine in ["top", "bottom", "left", "right"] for axis in axes.flatten()]
    for axis in axes.flatten():
        axis.tick_params(which='both', direction='in', width=3)
        axis.tick_params(which='major', length=5, labelsize=18, pad=5)
        axis.tick_params(which='minor', length=4)
    [axes[0, i].tick_params(labelbottom=False) for i in range(2)]
    if sampling_algos_paper:
        axes[1, 0].set_xlabel(r"$t / \Delta t_{\rm Metrop}$", fontsize=18, labelpad=3)
        axes[1, 1].set_xlabel(r"$t / \Delta t_{\rm Wolff}$", fontsize=18, labelpad=3)
    else:
        axes[1, 0].set_xlabel(r"$t / \Delta t_{\rm Metrop}^{\rm Ising}$", fontsize=18, labelpad=3)
        axes[1, 1].set_xlabel(r"$t / \Delta t_{{\rm Swendsen}\!-\!{\rm Wang}}^{\rm Ising}$", fontsize=18, labelpad=3)
    [axis.set_xlim([-5, 160]) for axis in axes.flatten()]
    [axes[i, 1].tick_params(labelleft=False) for i in range(2)]
    if simple_axis_labels:
        axes[0, 0].set_ylabel(r"$m\left( t; \beta J = 1 \right)$", fontsize=18, labelpad=3)
    else:
        axes[0, 0].set_ylabel(r"$m\left( x(t); \beta J = 1 \right)$", fontsize=18, labelpad=3)
    axes[1, 0].set_ylabel(higher_temperature_y_axis_label, fontsize=18, labelpad=3)
    [axis.set_ylim([-1.15, 1.15]) for axis in axes.flatten()]
    return fig, axes


def plot_magnetic_density_vs_time(axis, mediator, output_directory, sample_directory, temperature,
                                  temperature_index, lattice_length, number_of_equilibration_iterations):
    try:
        reduced_magnetic_density_sample = np.load(
            f"{output_directory}/{lattice_length}x{lattice_length}_ising_model_temperature_{temperature_index:02d}_"
            f"reduced_magnetic_density_sample_{mediator.replace('_mediator', '')}_algorithm.npy")
    except IOError:
        reduced_magnetic_density_sample = sample_getter.get_magnetic_density(
            sample_directory, temperature, temperature_index, lattice_length ** 2,
            number_of_equilibration_iterations)[:150].flatten()
        np.save(f"{output_directory}/{lattice_length}x{lattice_length}_ising_model_temperature_{temperature_index:02d}_"
                f"reduced_magnetic_density_sample_{mediator.replace('_mediator', '')}_algorithm.npy",
                reduced_magnetic_density_sample)
    axis.plot(reduced_magnetic_density_sample, color="k", linewidth=1, linestyle="-")


if __name__ == "__main__":
    if len(sys.argv) > 3:
        raise Exception("InterfaceError: At most two positional arguments permitted.  None are required but you may "
                        "provide sampling_algos_paper (see docstring of main() - default value is True) and "
                        "number_of_system_sizes (which must be an integer greater than 0 and less than 6 - default "
                        "value is 5).")
    if len(sys.argv) == 3:
        print("Two positional arguments provided.  These must be sampling_algos_paper (see docstring of main() - "
              "default value is True) and number_of_system_sizes (which must be an integer greater than 0 and less "
              "than 6 - default value is 5).")
        chosen_number_of_system_sizes = int(sys.argv[2])
        if chosen_number_of_system_sizes < 1 or chosen_number_of_system_sizes > 5:
            raise Exception("InterfaceError: chosen_number_of_system_sizes must be an integer greater than 0 and less "
                            "than 6 (default value is 5).")
        if sys.argv[1] == "True":
            main(True, chosen_number_of_system_sizes)
        elif sys.argv[1] == "False":
            main(False, chosen_number_of_system_sizes)
        else:
            Exception("InterfaceError: The provided value of sampling_algos_paper is neither True nor False (see "
                      "docstring of main()).")
    elif len(sys.argv) == 2:
        print("One positional argument provided.  This must be sampling_algos_paper (see docstring of main() - default "
              "value is True).  In addition, you may provide number_of_system_sizes - which must be an integer greater "
              "than 0 and less than 6 (default value is 5).")
        if sys.argv[1] == "True":
            main(True)
        elif sys.argv[1] == "False":
            main(False)
        else:
            Exception("InterfaceError: The provided value of sampling_algos_paper is neither True nor False (see "
                      "docstring of main()).")
    else:
        print("No positional arguments provided.  None are required but you may provide sampling_algos_paper (see "
              "docstring of main() - default value is True) and chosen_number_of_system_sizes (which must be an "
              "integer greater than 0 and less than 6 - default value is 5).")
        main()
