import importlib
import markov_chain_diagnostics
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

metropolis_mediator_module = importlib.import_module("mediator.metropolis_mediator")
coulomb_potential_module = importlib.import_module("potential.coulomb_potential")
exponential_power_potential_module = importlib.import_module("potential.exponential_power_potential")
gaussian_potential_module = importlib.import_module("potential.gaussian_potential")
ising_potential_module = importlib.import_module("potential.ising_potential")
lennard_jones_potential_without_cutoff_module = importlib.import_module(
    "potential.lennard_jones_potential_without_cutoff")
lennard_jones_potential_with_linked_lists_module = importlib.import_module(
    "potential.lennard_jones_potential_with_linked_lists")
lennard_jones_potential_without_linked_lists_module = importlib.import_module(
    "potential.lennard_jones_potential_without_linked_lists")
particle_separation_sampler_module = importlib.import_module("sampler.particle_separation_sampler")
potential_sampler_module = importlib.import_module("sampler.potential_sampler")
standard_mean_position_sampler_module = importlib.import_module("sampler.standard_mean_position_sampler")
standard_position_sampler_module = importlib.import_module("sampler.standard_position_sampler")


def main(argv):
    """argv is the location of the config file"""
    lennard_jones_potentials = [
        lennard_jones_potential_without_cutoff_module.LennardJonesPotentialWithoutCutoff,
        lennard_jones_potential_with_linked_lists_module.LennardJonesPotentialWithLinkedLists,
        lennard_jones_potential_without_linked_lists_module.LennardJonesPotentialWithoutLinkedLists]
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    config = parsing.read_config(parsing.parse_options(argv).config_file)
    mediator = factory.build_from_config(config, strings.to_camel_case(config.get("Run", "mediator")), "mediator")
    number_of_equilibration_iterations = mediator._number_of_equilibration_iterations
    temperatures, samplers, potential = mediator._temperatures, mediator._samplers, mediator._potential
    potential_prefactor = potential._prefactor

    if isinstance(potential, ising_potential_module.IsingPotential):
        if not (len(temperatures) == 2 and temperatures[0] == 1.2 and temperatures[1] == 3.0):
            raise ValueError("IsingPotential reference data only available for models for which two sampling "
                             "temperatures are given with values 1.2 and 3.0.")
    elif len(temperatures) > 1:
        raise RuntimeWarning(f"The value of number_of_temperature_values in the Mediator section is greater than 1.  "
                             f"Convergence is therefore tested only for the minimum temperature value.")
    beta = 1.0 / temperatures[0]
    """n.b., potentials may include additional prefactors (to beta and potential_prefactor (latter defined below) in 
        their definitions, e.g., the definitions of ExponentialPowerPotential and GaussianPotential include additional 
        prefactors of 1/power and 1/2, respectively - potential_prefactor defines the relative weight of the potential 
        in question when included in the sum of a more complex model"""
    combined_potential_prefactor = beta * potential_prefactor

    if isinstance(potential, ising_potential_module.IsingPotential):
        if not (len(samplers) <= 2 and
                [isinstance(sampler, potential_sampler_module.PotentialSampler) or
                 isinstance(sampler, standard_mean_position_sampler_module.StandardMeanPositionSampler)
                 for sampler in samplers]):
            raise ValueError("IsingPotential reference data only available for PotentialSampler and "
                             "StandardMeanPositionSampler.  Please give only these values for samplers in the Mediator "
                             "section.")
    elif isinstance(potential, coulomb_potential_module.CoulombPotential) or any(
            [isinstance(potential, lennard_jones_potential) for lennard_jones_potential in lennard_jones_potentials]):
        if not (len(samplers) == 1 and
                isinstance(samplers[0], particle_separation_sampler_module.ParticleSeparationSampler)):
            raise ValueError("Coulomb and Lennard-Jones reference data only available for ParticleSeparationSampler.  "
                             "Please give only this value for samplers in the Mediator section.")
    elif (isinstance(potential, gaussian_potential_module.GaussianPotential) or
          isinstance(potential, exponential_power_potential_module.ExponentialPowerPotential)):
        if not (len(samplers) == 1 and
                isinstance(samplers[0], standard_position_sampler_module.StandardPositionSampler)):
            raise ValueError("Gaussian and exponential-power reference data only available for StandardPositionSampler."
                             "  Please give only this value for samplers in the Mediator section.")
    else:
        raise ValueError("Reference data not provided for this potential.")

    if isinstance(potential, gaussian_potential_module.GaussianPotential) or (
            isinstance(potential, exponential_power_potential_module.ExponentialPowerPotential) and
            potential._power == 2.0):
        reference_sample = np.random.normal(0.0, combined_potential_prefactor ** (- 0.5), size=10000)
    elif (isinstance(potential, exponential_power_potential_module.ExponentialPowerPotential) and
          potential._power == 4.0):
        if combined_potential_prefactor == 0.25:
            reference_sample = np.load('permanent_data/reference_data/fourth_exponential_power_potential_beta_one_'
                                       'quarter_reference_sample.npy')
        else:
            raise ValueError("ExponentialPowerPotential reference data only available for "
                             "ExponentialPowerPotential._power = 2 or 4.  For the latter, reference data is only "
                             "available for models for which the product of beta (the reciprocal sampling temperature) "
                             "and ExponentialPowerPotential._prefactor is equal to 0.25 (n.b., the potential's "
                             "definition includes an additional prefactor of 1 / power).")
    elif isinstance(potential, coulomb_potential_module.CoulombPotential):
        if (config.get("ModelSettings", "number_of_particles") == "2" and combined_potential_prefactor == 2.0 and
                config.get("ModelSettings", "size_of_particle_space") == "[1.0, 1.0, 1.0]"):
            reference_sample = np.load('permanent_data/reference_data/two_unit_charge_coulomb_particles_beta_2_unit_'
                                       'cube_separation_reference_sample.npy')
        else:
            raise ValueError("CoulombPotential reference data only available for models for which the product of beta "
                             "(the reciprocal sampling temperature) and CoulombPotential.prefactor is equal to 2.0, "
                             "number_of_particles equals 2 and size_of_particle_space equals [1.0, 1.0, 1.0] (n.b., "
                             "number_of_particles and size_of_particle_space are set in the ModelSettings section).")
    elif any([isinstance(potential, lennard_jones_potential) for lennard_jones_potential in lennard_jones_potentials]):
        well_depth = potential._well_depth
        characteristic_length = potential._characteristic_length
        if isinstance(potential, lennard_jones_potential_without_cutoff_module.LennardJonesPotentialWithoutCutoff):
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
        else:
            cutoff_length = potential._cutoff_length
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
    elif isinstance(potential, ising_potential_module.IsingPotential):
        lattice_dimensionality = potential._lattice_dimensionality
        potential_constant = potential.potential_constant
        number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")
        if (number_of_particles == 16 and lattice_dimensionality == 2 and potential_constant == - 1.0 and
                config.get("ModelSettings", "size_of_particle_space") == "2"):
            if isinstance(mediator, metropolis_mediator_module.MetropolisMediator):
                thinning_level = 10
            else:
                thinning_level = 1
            # [number_of_equilibration_iterations + 1::thinning_level] in next line removes equilibration obs then thins
            samples = [[
                sampler.get_sample(temperature_index)[number_of_equilibration_iterations + 1::thinning_level].flatten()
                for temperature_index in range(len(temperatures))] for sampler in samplers]
            sampler_labels = ["sampler_string" for _ in samplers]
            for sampler_index, sampler in enumerate(samplers):
                if isinstance(sampler, standard_mean_position_sampler_module.StandardMeanPositionSampler):
                    sampler_labels[sampler_index] = "standard_mean_position_sampler"
                    # this is the magnetic density and we're interested in the expected absolute magnetic density (to
                    # distinguish the high- and low-T phases when using the symm-restoring Wolff algorithm) so...
                    samples[sampler_index] = [np.abs(sample) for sample in samples[sampler_index]]
                elif isinstance(sampler, potential_sampler_module.PotentialSampler):
                    sampler_labels[sampler_index] = "potential_sampler"
            sample_means_and_errors = [np.array([markov_chain_diagnostics.get_sample_mean_and_error(sample)
                                                for sample in samples[sampler_index]])
                                       for sampler_index in range(len(samples))]
            squared_deviation_samples = [[(sample - np.mean(sample)) ** 2 for sample in samples[sampler_index]]
                                         for sampler_index in range(len(samples))]
            sample_variances_and_errors = [np.array([markov_chain_diagnostics.get_sample_mean_and_error(sample)
                                                     for sample in squared_deviation_samples[sampler_index]])
                                           for sampler_index in range(len(squared_deviation_samples))]

            expected_magnetic_norm_density_reference_values = ["0.997", "0.601"]
            expected_magnetic_norm_susc_per_particle_reference_values = ["0.00566", "0.476"]
            expected_potential_per_particle_reference_values = ["-1.99", "-1.01"]
            expected_spec_heat_per_particle_reference_values = ["0.0663", "0.603"]
            for temperature_index, temperature in enumerate(temperatures):
                print("---------------------------------")
                print(f"Temperature = {temperature:.4f}")
                for sampler_index, sampler in enumerate(samplers):
                    if sampler_labels[sampler_index] == "standard_mean_position_sampler":
                        # expected magnetic-norm density is E[|m|] où m = sum_i s_i / N
                        print(f"Sample estimate of expected magnetic-norm density = "
                              f"{sample_means_and_errors[sampler_index][temperature_index][0]:.3g} +- "
                              f"{sample_means_and_errors[sampler_index][temperature_index][1]:.3g} "
                              f"(ref. value = {expected_magnetic_norm_density_reference_values[temperature_index]})")
                        # w/M = Nm the magnetisation, the expected magnetic susc is \nabla_h E[M] = beta N^2 Var[m],
                        # but to compare Wolff and Metropolis, we estimate beta N Var[|m|] (the expected magnetic-norm
                        # susc (per particle))
                        sample_variances_and_errors[sampler_index][temperature_index] *= (number_of_particles /
                                                                                          temperature)
                        print(f"Sample estimate of expected magnetic-norm susc. per particle = "
                              f"{sample_variances_and_errors[sampler_index][temperature_index][0]:.3g} +- "
                              f"{sample_variances_and_errors[sampler_index][temperature_index][1]:.3g} (ref. value = "
                              f"{expected_magnetic_norm_susc_per_particle_reference_values[temperature_index]})")
                    elif sampler_labels[sampler_index] == "potential_sampler":
                        print(f"Sample estimate of expected potential per particle = "
                              f"{sample_means_and_errors[sampler_index][temperature_index][0] / number_of_particles} "
                              f"+- {sample_means_and_errors[sampler_index][temperature_index][1] / number_of_particles}"
                              f" (ref. value = {expected_potential_per_particle_reference_values[temperature_index]})")
                        # expected specific heat is \partial_T E[U] = beta^2 Var[U] (a dimensionless quantity) -- we
                        # estimate beta^2 Var[U] / N (the expected specific heat per particle)
                        sample_variances_and_errors[sampler_index][temperature_index] /= (number_of_particles *
                                                                                          temperature ** 2)
                        print(f"Sample estimate of expected specific heat per particle = "
                              f"{sample_variances_and_errors[sampler_index][temperature_index][0]} +- "
                              f"{sample_variances_and_errors[sampler_index][temperature_index][1]} (ref. value = "
                              f"{expected_spec_heat_per_particle_reference_values[temperature_index]})")
                print("---------------------------------")
        else:
            raise ValueError("IsingPotential reference data only available for 16-particle models on a two-dimensional "
                             "lattice for which size_of_particle_space equals 2 and the product of "
                             "IsingPotential.prefactor and IsingPotential.exchange_constant is equal to 1.0 (n.b., "
                             "number_of_particles and size_of_particle_space are set in the ModelSettings section).")

    if not isinstance(potential, ising_potential_module.IsingPotential):
        reference_cdf = markov_chain_diagnostics.get_cumulative_distribution(reference_sample)
        sample = samplers[0].get_sample(temperature_index=0)[number_of_equilibration_iterations + 1:].flatten()
        effective_sample_size = markov_chain_diagnostics.get_effective_sample_size(sample)
        print(f"Effective sample size = {effective_sample_size} (from a total sample size of {len(sample)}).")
        sample_cdf = markov_chain_diagnostics.get_cumulative_distribution(sample)

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


if __name__ == '__main__':
    main(sys.argv[1:])
