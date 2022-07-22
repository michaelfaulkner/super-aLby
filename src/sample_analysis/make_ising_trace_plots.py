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
    max_lattice_length = 2 ** (number_of_system_sizes + 1)
    config_file_wolff = [
        f"config_files/sampling_algos/ising_figures/{max_lattice_length}x{max_lattice_length}_wolff.ini"]
    config_file_metrop = [
        f"config_files/sampling_algos/ising_figures/{max_lattice_length}x{max_lattice_length}_metropolis.ini"]
    (wolff_mediator, _, samplers, sample_directories_wolff, temperatures, number_of_equilibration_iterations,
     number_of_observations, _, _, _, _) = helper_methods.get_basic_config_data(config_file_wolff)
    metrop_mediator = helper_methods.get_basic_config_data(config_file_metrop)[0]
    output_directory = sample_directories_wolff[0].replace(f"/{max_lattice_length}x{max_lattice_length}_wolff", "")
    sample_directory_wolff = f"{sample_directories_wolff[0]}/job_00"
    sample_directory_metrop = f"{output_directory}/{max_lattice_length}x{max_lattice_length}_metropolis/job_00"

    transition_temperature = 2.0 / math.log(1 + 2 ** 0.5)
    temperatures = np.array(temperatures)
    temperature_near_critical_point = np.min(np.delete(temperatures, np.where(temperatures < transition_temperature)))
    temperature_near_critical_point_index = np.where(temperatures == temperature_near_critical_point)[0][0]

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
    axes[1, 0].set_xlabel(r"Metropolis observation, $t$", fontsize=18, labelpad=3)
    axes[1, 1].set_xlabel(r"Wolff observation, $t$", fontsize=18, labelpad=3)
    [axis.set_xlim([-5, 160]) for axis in axes.flatten()]
    [axes[i, 1].tick_params(labelleft=False) for i in range(2)]
    axes[0, 0].set_ylabel(r"$m\left( x(t); \beta J = 1 \right)$", fontsize=18, labelpad=3)
    axes[1, 0].set_ylabel(
        rf"$m\left( x(t); \beta J = 1 \, / \, {temperature_near_critical_point:.1f} \right)$", fontsize=18, labelpad=3)
    [axis.set_ylim([-1.15, 1.15]) for axis in axes.flatten()]

    plot_magnetic_density_vs_time(
        axes[0, 0], metrop_mediator, output_directory, sample_directory_metrop, temperatures[0], 0, max_lattice_length,
        number_of_equilibration_iterations)
    plot_magnetic_density_vs_time(
        axes[0, 1], wolff_mediator, output_directory, sample_directory_wolff, temperatures[0], 0, max_lattice_length,
        number_of_equilibration_iterations)
    plot_magnetic_density_vs_time(
        axes[1, 0], metrop_mediator, output_directory, sample_directory_metrop, temperature_near_critical_point,
        temperature_near_critical_point_index, max_lattice_length, number_of_equilibration_iterations)
    plot_magnetic_density_vs_time(
        axes[1, 1], wolff_mediator, output_directory, sample_directory_wolff, temperature_near_critical_point,
        temperature_near_critical_point_index, max_lattice_length, number_of_equilibration_iterations)
    fig.savefig(f"{output_directory}/{max_lattice_length}x{max_lattice_length}_ising_model_magnetic_density_vs_time_"
                f"metropolis_and_wolff.pdf", bbox_inches="tight")


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
    if len(sys.argv) > 2:
        raise Exception("InterfaceError: At most one positional argument permitted.  None are required but you may "
                        "provide number_of_system_sizes, which must be an integer greater than 0 and less than 6 "
                        "(default value is 5).")
    if len(sys.argv) == 2:
        print("One positional argument provided.  This must be number_of_system_sizes - which must be an integer "
              "greater than 0 and less than 6 (default value is 5).")
        chosen_number_of_system_sizes = int(sys.argv[1])
        if chosen_number_of_system_sizes < 1 or chosen_number_of_system_sizes > 4:
            raise Exception(
                "InterfaceError: chosen_number_of_system_sizes must be an integer greater than 0 and less "
                "than 6 (default value is 5).")
        main(chosen_number_of_system_sizes)
    else:
        print("No positional arguments provided.  None are required but you may provide chosen_number_of_system_sizes, "
              "which must be an integer greater than 0 and less than 6 (default value is 5).")
        main()
