from configparser import NoSectionError, NoOptionError
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
    # todo remove gaussian reference data now that we use np.random.normal() in lion 57?
    # todo rename other ref data files (and convert to npy?) eg, do not use variance in 4th exponential power file name
    # todo move ref data files to a reference_data directory and move sample analysis scripts to sample_analysis
    #  (separate from output directory)
    """argv is the location of the config file"""
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    config = parsing.read_config(parsing.parse_options(argv).config_file)
    possible_mediators = ["EuclideanLeapfrogMediator", "ToroidalLeapfrogMediator", "LazyToroidalLeapfrogMediator",
                          "MetropolisMediator"]
    sampler, number_of_equilibration_iterations, config_file_mediator = None, None, None
    for possible_mediator in possible_mediators:
        try:
            sampler = factory.build_from_config(config, strings.to_camel_case(config.get(possible_mediator, "sampler")),
                                                "sampler")
            number_of_equilibration_iterations = parsing.get_value(config, possible_mediator,
                                                                   "number_of_equilibration_iterations")
            config_file_mediator = possible_mediator
            break
        except NoSectionError:
            continue
    if sampler is None:
        raise IOError("Mediator not one of EuclideanLeapfrogMediator, ToroidalLeapfrogMediator, "
                      "LazyToroidalLeapfrogMediator or MetropolisMediator.")

    """n.b., potentials may include additional prefactors (to beta and potential_prefactor (latter defined below) in 
        their definitions, e.g., the definitions of ExponentialPowerPotential and GaussianPotential include additional 
        prefactors of 1/power and 1/2, respectively - potential_prefactor defines the relative weight of the potential 
        in question when included in the sum of a more complex model"""
    beta = parsing.get_value(config, "ModelSettings", "beta")
    potential = config.get(config_file_mediator, "potential")
    try:
        potential_prefactor = parsing.get_value(config, strings.to_camel_case(potential), "prefactor")
    except (NoOptionError, RuntimeError) as _:
        potential_prefactor = 1.0
    combined_potential_prefactor = beta * potential_prefactor

    if potential == "gaussian_potential" or (potential == "exponential_power_potential" and config.get(
                "ExponentialPowerPotential", "power") == "2.0"):
        reference_sample = np.random.normal(0.0, combined_potential_prefactor ** (- 0.5), size=10000)
    elif potential == "exponential_power_potential" and config.get("ExponentialPowerPotential", "power") == "4.0":
        if combined_potential_prefactor == 0.25:
            reference_sample = np.loadtxt('output/other_convergence_tests/fourth_exponential_power_variance_4_reference'
                                          '_sample.csv', dtype=float, delimiter=',')
        else:
            raise ValueError("ExponentialPowerPotential reference data only available for models for which the product "
                             "of beta (set in the ModelSettings section) and ExponentialPowerPotential.prefactor is "
                             "equal to 0.25 (n.b., the potential's definition includes an additional prefactor of 1 / "
                             "power).")
    elif potential == "coulomb_potential":
        if (config.get("ModelSettings", "number_of_particles") == "2" and combined_potential_prefactor == 2.0 and
                config.get("ModelSettings", "size_of_particle_space") == "[1.0, 1.0, 1.0]"):
            reference_sample = np.loadtxt('output/srmc_in_soft_matter/two_unit_charge_coulomb_particles_unit_cube_'
                                          'beta_2_separation_reference_sample.csv', dtype=float, delimiter=',')
        else:
            raise ValueError("CoulombPotential reference data only available for models for which the product of beta "
                             "and CoulombPotential.prefactor is equal to 2.0, number_of_particles equals 2 and "
                             "size_of_particle_space equals [1.0, 1.0, 1.0] (n.b., beta, number_of_particles and "
                             "size_of_particle_space are set in the ModelSettings section).")
    elif "lennard_jones" in potential:
        try:
            well_depth = parsing.get_value(config, strings.to_camel_case(potential), "well_depth")
        except (NoOptionError, RuntimeError) as _:
            well_depth = 1.0
        if potential == "lennard_jones_potential_without_cutoff":
            if (config.get("ModelSettings", "number_of_particles") == "2" and combined_potential_prefactor == 2.0 and
                    well_depth == 0.25 and config.get("ModelSettings", "size_of_particle_space") == "[5.0, 5.0, 5.0]"):
                reference_sample = np.loadtxt('output/srmc_in_soft_matter/two_lennard_jones_particles_5x5x5_cube_well_'
                                              'depth_one_quarter_beta_2_separation_reference_sample.csv', dtype=float,
                                              delimiter=',')
            else:
                raise ValueError("LennardJonesPotentialWithoutCutoff reference data only available for models for "
                                 "which the product of beta and LennardJonesPotentialWithoutCutoff.prefactor is equal "
                                 "to 2.0, LennardJonesPotentialWithoutCutoff.well_depth is equal to 0.25, "
                                 "number_of_particles equals 2, size_of_particle_space equals [5.0, 5.0, 5.0] (n.b., "
                                 "beta, number_of_particles and size_of_particle_space are set in the ModelSettings "
                                 "section).")
        elif (potential == "lennard_jones_potential_with_linked_lists" or
              potential == "lennard_jones_potential_without_linked_lists"):
            try:
                cutoff_length = parsing.get_value(config, strings.to_camel_case(potential), "cutoff_length")
            except (NoOptionError, RuntimeError) as _:
                cutoff_length = 2.5
            if (config.get("ModelSettings", "number_of_particles") == "8" and combined_potential_prefactor == 1.0 and
                    well_depth == 1.0 and cutoff_length == 3.0 and
                    config.get("ModelSettings", "size_of_particle_space") == "[8.0, 8.0, 8.0]"):
                # todo rename the following file to include the cutoff value (cut_off_length = 3.0)
                reference_sample = np.loadtxt('output/srmc_in_soft_matter/eight_lennard_jones_particles_8x8x8_cube_'
                                              'cutoff_3_beta_1_separation_reference_sample.csv', dtype=float,
                                              delimiter=',')
            else:
                raise ValueError("LennardJonesPotentialWithLinkedLists / LennardJonesPotentialWithoutLinkedLists "
                                 "reference data only available for models for which the product of beta and "
                                 "LennardJonesPotentialWithoutCutoff.prefactor is equal to 1.0, "
                                 "LennardJonesPotentialWithoutCutoff.well_depth is equal to 1.0, "
                                 "LennardJonesPotentialWithoutCutoff.cut_off_length is equal to 3.0, "
                                 "number_of_particles equals 8, size_of_particle_space equals [8.0, 8.0, 8.0] (n.b., "
                                 "beta, number_of_particles and size_of_particle_space are set in the ModelSettings "
                                 "section).")
    else:
        raise ValueError("Reference data not provided for this potential.")

    reference_cdf = get_cumulative_distribution(reference_sample)
    sample = sampler.get_sample()[number_of_equilibration_iterations + 1:].flatten()
    effective_sample_size = markov_chain_diagnostics.get_effective_sample_size(sample)
    print(f"Effective sample size = {effective_sample_size} (from a total sample size of {len(sample)}).")
    sample_cdf = get_cumulative_distribution(sample)

    plt.plot(reference_cdf[0], reference_cdf[1], color='r', linewidth=3, linestyle='-', label='reference data')
    plt.plot(sample_cdf[0], sample_cdf[1], color='k', linewidth=2, linestyle='-', label='super-aLby data')

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
