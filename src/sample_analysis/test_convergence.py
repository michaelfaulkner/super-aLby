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
other_methods = importlib.import_module("other_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")
run_module = importlib.import_module("run")
markov_chain_diagnostics = importlib.import_module("sample_analysis.markov_chain_diagnostics")


def main(argv):
    """argv is the location of the config file"""
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    config = parsing.read_config(parsing.parse_options(argv).config_file)
    possible_mediators = ["EuclideanLeapfrogMediator", "ToroidalLeapfrogMediator", "LazyToroidalLeapfrogMediator",
                          "MetropolisMediator", "WolffMediator"]
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

    potential = config.get(config_file_mediator, "potential")
    try:
        potential_prefactor = parsing.get_value(config, strings.to_camel_case(potential), "prefactor")
    except (NoOptionError, RuntimeError) as _:
        potential_prefactor = 1.0  # set as default value

    temperatures = other_methods.get_temperatures(
        parsing.get_value(config, config_file_mediator, "minimum_temperature"),
        parsing.get_value(config, config_file_mediator, "maximum_temperature"),
        parsing.get_value(config, config_file_mediator, "number_of_temperature_values"))
    if potential == "ising_potential":
        if not (len(temperatures) == 2 and temperatures[0] == 1.2 and temperatures[1] == 3.0):
            raise ValueError("IsingPotential reference data only available for models for which two sampling "
                             "temperatures are given with values 1.2 and 3.0.")
    elif len(temperatures) > 1:
        raise RuntimeWarning(f"The value of number_of_temperature_values in {config_file_mediator} is greater than 1.  "
                             f"Convergence is therefore tested only for the minimum temperature value.")
    beta = 1.0 / temperatures[0]
    """n.b., potentials may include additional prefactors (to beta and potential_prefactor (latter defined below) in 
        their definitions, e.g., the definitions of ExponentialPowerPotential and GaussianPotential include additional 
        prefactors of 1/power and 1/2, respectively - potential_prefactor defines the relative weight of the potential 
        in question when included in the sum of a more complex model"""
    combined_potential_prefactor = beta * potential_prefactor

    if potential == "gaussian_potential" or (potential == "exponential_power_potential" and config.get(
            "ExponentialPowerPotential", "power") == "2.0"):
        reference_sample = np.random.normal(0.0, combined_potential_prefactor ** (- 0.5), size=10000)
    elif potential == "exponential_power_potential" and config.get("ExponentialPowerPotential", "power") == "4.0":
        if combined_potential_prefactor == 0.25:
            reference_sample = np.load('permanent_data/reference_data/fourth_exponential_power_potential_beta_one_'
                                       'quarter_reference_sample.npy')
        else:
            raise ValueError("ExponentialPowerPotential reference data only available for models for which the product "
                             "of beta (the reciprocal sampling temperature) and ExponentialPowerPotential.prefactor is "
                             "equal to 0.25 (n.b., the potential's definition includes an additional prefactor of 1 / "
                             "power).")
    elif potential == "coulomb_potential":
        if (config.get("ModelSettings", "number_of_particles") == "2" and combined_potential_prefactor == 2.0 and
                config.get("ModelSettings", "size_of_particle_space") == "[1.0, 1.0, 1.0]"):
            reference_sample = np.load('permanent_data/reference_data/two_unit_charge_coulomb_particles_beta_2_unit_'
                                       'cube_separation_reference_sample.npy')
        else:
            raise ValueError("CoulombPotential reference data only available for models for which the product of beta "
                             "(the reciprocal sampling temperature) and CoulombPotential.prefactor is equal to 2.0, "
                             "number_of_particles equals 2 and size_of_particle_space equals [1.0, 1.0, 1.0] (n.b., "
                             "number_of_particles and size_of_particle_space are set in the ModelSettings section).")
    elif "lennard_jones" in potential:
        try:
            well_depth = parsing.get_value(config, strings.to_camel_case(potential), "well_depth")
        except (NoOptionError, RuntimeError) as _:
            well_depth = 1.0  # set as default value
        try:
            characteristic_length = parsing.get_value(config, strings.to_camel_case(potential), "characteristic_length")
        except (NoOptionError, RuntimeError) as _:
            characteristic_length = 1.0  # set as default value
        if potential == "lennard_jones_potential_without_cutoff":
            if (config.get("ModelSettings", "number_of_particles") == "2" and combined_potential_prefactor == 2.0 and
                    well_depth == 0.25 and characteristic_length == 1.0 and (
                            config.get("ModelSettings", "size_of_particle_space") == "[5.0, 5.0, 5.0]" or
                            config.get("ModelSettings", "size_of_particle_space") == "[2.0, 2.0, 2.0]" or
                            config.get("ModelSettings", "size_of_particle_space") == "[1.0, 1.0, 1.0]")):
                if config.get("ModelSettings", "size_of_particle_space") == "[5.0, 5.0, 5.0]":
                    reference_sample = np.load(
                        'permanent_data/reference_data/two_lennard_jones_particles_without_cutoff_well_depth_one_'
                        'quarter_char_length_1_beta_2_5x5x5_cube_separation_reference_sample.npy')
                elif config.get("ModelSettings", "size_of_particle_space") == "[2.0, 2.0, 2.0]":
                    reference_sample = np.load(
                        'permanent_data/reference_data/two_lennard_jones_particles_without_cutoff_well_depth_one_'
                        'quarter_char_length_1_beta_2_2x2x2_cube_separation_reference_sample.npy')
                else:
                    reference_sample = np.load(
                        'permanent_data/reference_data/two_lennard_jones_particles_without_cutoff_well_depth_one_'
                        'quarter_char_length_1_beta_2_1x1x1_cube_separation_reference_sample.npy', )
            else:
                raise ValueError("LennardJonesPotentialWithoutCutoff reference data only available for models for "
                                 "which the product of beta (the reciprocal sampling temperature) and "
                                 "LennardJonesPotentialWithoutCutoff.prefactor is equal to 2.0, "
                                 "LennardJonesPotentialWithoutCutoff.well_depth is equal to 0.25, "
                                 "LennardJonesPotentialWithoutCutoff.characteristic_length is equal to 1.0, "
                                 "number_of_particles equals 2, size_of_particle_space equals [5.0, 5.0, 5.0], "
                                 "[2.0, 2.0, 2.0] or [1.0, 1.0, 1.0] (n.b., number_of_particles and "
                                 "size_of_particle_space are set in the ModelSettings section).")
        elif (potential == "lennard_jones_potential_with_linked_lists" or
              potential == "lennard_jones_potential_without_linked_lists"):
            try:
                cutoff_length = parsing.get_value(config, strings.to_camel_case(potential), "cutoff_length")
            except (NoOptionError, RuntimeError) as _:
                cutoff_length = 2.5  # set as default value
            if (config.get("ModelSettings", "number_of_particles") == "8" and combined_potential_prefactor == 1.0 and
                    well_depth == 1.0 and characteristic_length == 1.0 and cutoff_length == 3.0 and
                    config.get("ModelSettings", "size_of_particle_space") == "[8.0, 8.0, 8.0]"):
                reference_sample = np.load('permanent_data/reference_data/eight_lennard_jones_particles_cutoff_3_well_'
                                           'depth_1_char_length_1_beta_1_8x8x8_cube_separation_reference_sample.npy')
            else:
                raise ValueError("LennardJonesPotentialWithLinkedLists / LennardJonesPotentialWithoutLinkedLists "
                                 "reference data only available for models for which the product of beta (the "
                                 "reciprocal sampling temperature) and LennardJonesPotentialWithoutCutoff.prefactor is "
                                 "equal to 1.0, LennardJonesPotentialWithoutCutoff.well_depth is equal to 1.0, "
                                 "LennardJonesPotentialWithoutCutoff.cut_off_length is equal to 3.0, "
                                 "number_of_particles equals 8, size_of_particle_space equals [8.0, 8.0, 8.0] (n.b., "
                                 "number_of_particles and size_of_particle_space are set in the ModelSettings "
                                 "section).")
    elif potential == "ising_potential":
        try:
            lattice_dimensionality = parsing.get_value(config, strings.to_camel_case(potential),
                                                       "lattice_dimensionality")
        except (NoOptionError, RuntimeError) as _:
            lattice_dimensionality = 2  # set as default value
        try:
            exchange_constant = parsing.get_value(config, strings.to_camel_case(potential), "exchange_constant")
        except (NoOptionError, RuntimeError) as _:
            exchange_constant = 1.0  # set as default value
        number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")
        if (number_of_particles == 16 and lattice_dimensionality == 2 and
                potential_prefactor * exchange_constant == 1.0 and
                config.get("ModelSettings", "size_of_particle_space") == "2"):
            if config_file_mediator == "MetropolisMediator":
                thinning_level = 10
            else:
                thinning_level = 1
            # [number_of_equilibration_iterations + 1::thinning_level] in next line removes equilibration obs then thins
            samples = [
                sampler.get_sample(temperature_index)[number_of_equilibration_iterations + 1::thinning_level].flatten()
                for temperature_index in range(len(temperatures))]
            if config.get(possible_mediator, "sampler") == "standard_mean_position_sampler":
                # sampler is the magnetic density and we're interested in the expected absolute magnetic density (in
                # order to distinguish the high- and low-T phases when using the symm-restoring Wolff algorithm) so...
                samples = [np.abs(sample) for sample in samples]
            sample_means_and_errors = np.array([[np.mean(sample), np.std(sample) / len(sample) ** 0.5]
                                                for sample in samples])
            squared_deviation_samples = [(sample - np.mean(sample)) ** 2 for sample in samples]
            sample_variances_and_errors = np.array([[np.mean(sample), np.std(sample) / len(sample) ** 0.5]
                                                    for sample in squared_deviation_samples])
            expected_magnetic_norm_density_reference_values = ["0.997", "0.601"]
            expected_magnetic_norm_susc_per_particle_reference_values = ["0.00566", "0.476"]
            expected_potential_per_particle_reference_values = ["-1.99", "-1.01"]
            expected_spec_heat_per_particle_reference_values = ["0.0663", "0.603"]
            for index, temperature in enumerate(temperatures):
                print("---------------------------------")
                print(f"Temperature = {temperature:.4f}")
                if config.get(possible_mediator, "sampler") == "standard_mean_position_sampler":
                    # expected magnetic-norm density is E(|m|) où m = sum_i s_i / N
                    print(f"Sample estimate of expected magnetic-norm density = "
                          f"{sample_means_and_errors[index][0]:.3g} +- {sample_means_and_errors[index][1]:.3g} "
                          f"(ref. value = {expected_magnetic_norm_density_reference_values[index]})")
                    # w/M = Nm the magnetisation, expected magnetic susc is dE(M)/dh = beta N^2 Var(m), but to compare
                    # Wolff and Metropolis simulations, we estimate the expected magnetic-norm susc beta N^2 Var(|m|)
                    print(f"Sample estimate of expected magnetic-norm susc. per particle = "
                          f"{number_of_particles * sample_variances_and_errors[index][0] / temperature:.3g} +- "
                          f"{number_of_particles * sample_variances_and_errors[index][1] / temperature:.3g} (ref. "
                          f"value = {expected_magnetic_norm_susc_per_particle_reference_values[index]})")
                elif config.get(possible_mediator, "sampler") == "potential_sampler":
                    print(f"Sample estimate of expected potential per particle = "
                          f"{sample_means_and_errors[index][0] / number_of_particles} +- "
                          f"{sample_means_and_errors[index][1] / number_of_particles} (ref. value = "
                          f"{expected_potential_per_particle_reference_values[index]})")
                    # expected specific heat is dE(U)/dT = beta^2 Var(U) (a dimensionless quantity)
                    print(f"Sample estimate of expected specific heat per particle = "
                          f"{sample_variances_and_errors[index][0] / number_of_particles / temperature ** 2} +- "
                          f"{sample_variances_and_errors[index][1] / number_of_particles / temperature ** 2} (ref. "
                          f"value = {expected_spec_heat_per_particle_reference_values[index]})")
                print("---------------------------------")
        else:
            raise ValueError("IsingPotential reference data only available for 16-particle models on a two-dimensional "
                             "lattice for which size_of_particle_space equals 2 and the product of "
                             "IsingPotential.prefactor and IsingPotential.exchange_constant is equal to 1.0 (n.b., "
                             "number_of_particles and size_of_particle_space are set in the ModelSettings section).")
    else:
        raise ValueError("Reference data not provided for this potential.")

    if potential != "ising_potential":
        reference_cdf = get_cumulative_distribution(reference_sample)
        sample = sampler.get_sample(temperature_index=0)[number_of_equilibration_iterations + 1:].flatten()
        effective_sample_size = markov_chain_diagnostics.get_effective_sample_size(sample)
        print(f"Effective sample size = {effective_sample_size} (from a total sample size of {len(sample)}).")
        sample_cdf = get_cumulative_distribution(sample)

        plt.plot(reference_cdf[0], reference_cdf[1], color='r', linewidth=3, linestyle='-', label='reference data')
        plt.plot(sample_cdf[0], sample_cdf[1], color='k', linewidth=2, linestyle='-', label='super-aLby data')

        plt.xlabel(r"$x$", fontsize=15, labelpad=10)
        plt.ylabel(r"$ F_n \left( X < x \right)$", fontsize=15, labelpad=10)
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
