from configparser import NoSectionError
import importlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add the directory that contains the module plotting_functions to sys.path
this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)

factory = importlib.import_module("base.factory")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")
run_module = importlib.import_module("run")
markov_chain_diagnostics = importlib.import_module("output.markov_chain_diagnostics")


def main(argv):
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    config = parsing.read_config(parsing.parse_options(argv).config_file)

    try:
        sampler = factory.build_from_config(config, strings.to_camel_case(config.get("LeapfrogMediator", "sampler")),
                                            "sampler")
        number_of_equilibration_iterations = parsing.get_value(config, "LeapfrogMediator",
                                                               "number_of_equilibration_iterations")
        if config.get("LeapfrogMediator", "potential") == 'exponential_power_potential' and config.get(
                "ExponentialPowerPotential", "power") == "4.0":
            reference_sample = np.loadtxt('output/other_convergence_tests/fourth_exponential_power_variance_4_reference'
                                          '_sample.csv', dtype=float, delimiter=',')
        elif config.get("LeapfrogMediator", "kinetic_energy") == 't_distribution_kinetic_energy':
            reference_sample = np.loadtxt('output/other_convergence_tests/gaussian_variance_1_reference_sample.csv',
                                          dtype=float, delimiter=',')
        else:
            reference_sample = np.loadtxt('output/other_convergence_tests/gaussian_variance_4_reference_sample.csv',
                                          dtype=float, delimiter=',')
    except NoSectionError:
        try:
            sampler = factory.build_from_config(
                config, strings.to_camel_case(config.get("ToroidalLeapfrogMediator", "sampler")), "sampler")
            number_of_equilibration_iterations = parsing.get_value(config, "ToroidalLeapfrogMediator",
                                                                   "number_of_equilibration_iterations")
            if config.get("ToroidalLeapfrogMediator", "potential") == 'coulomb_potential':
                reference_sample = np.loadtxt('output/srmc_in_soft_matter/two_unit_charge_coulomb_particles_unit_cube_'
                                              'beta_2_reference_sample.csv', dtype=float, delimiter=',')
            else:
                reference_sample = np.loadtxt('output/srmc_in_soft_matter/two_lennard_jones_particles_5x5x5_cube_beta_'
                                              '2_reference_sample.csv', dtype=float, delimiter=',')
        except NoSectionError:
            sampler = factory.build_from_config(
                config, strings.to_camel_case(config.get("LazyToroidalLeapfrogMediator", "sampler")), "sampler")
            number_of_equilibration_iterations = parsing.get_value(config, "LazyToroidalLeapfrogMediator",
                                                                   "number_of_equilibration_iterations")
            if config.get("LazyToroidalLeapfrogMediator", "potential") == 'coulomb_potential':
                reference_sample = np.loadtxt('output/srmc_in_soft_matter/two_unit_charge_coulomb_particles_unit_cube_'
                                              'beta_2_reference_sample.csv', dtype=float, delimiter=',')
            else:
                reference_sample = np.loadtxt('output/srmc_in_soft_matter/two_lennard_jones_particles_5x5x5_cube_beta_'
                                              '2_reference_sample.csv', dtype=float, delimiter=',')

    reference_cdf = get_cumulative_distribution(reference_sample)
    sample = sampler.get_sample()
    if type(sample[0]) != np.float64:
        sample = sample[:, 0]
    sample_size = len(sample) - number_of_equilibration_iterations - 1
    effective_sample_size = markov_chain_diagnostics.get_effective_sample_size(sample)
    print(f"Effective sample size = {effective_sample_size} (from a total sample size of {sample_size}).")
    sample_cdf = get_cumulative_distribution(sample[number_of_equilibration_iterations + 1:])

    plt.plot(reference_cdf[0], reference_cdf[1], color='r', linewidth=3, linestyle='-', label='reference data')
    plt.plot(sample_cdf[0], sample_cdf[1], color='k', linewidth=2, linestyle='-', label='super-rel-mc data')

    plt.xlabel(r"$x$", fontsize=15, labelpad=10)
    plt.ylabel(r"$ \mathbb{P} \left( X < x \right)$", fontsize=15, labelpad=10)
    plt.tick_params(axis='both', which='major', labelsize=14, pad=10)
    legend = plt.legend(loc='lower right', fontsize=10)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)
    plt.tight_layout()
    plt.show()


def get_cumulative_distribution(input_sample):
    bin_values = np.arange(1, len(input_sample) + 1) / float(len(input_sample))
    ordered_input_sample = np.sort(input_sample)
    return ordered_input_sample, bin_values


if __name__ == '__main__':
    main(sys.argv[1:])
