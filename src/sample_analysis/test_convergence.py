from configparser import NoOptionError
from markov_chain_diagnostics import get_cumulative_distribution, get_sample_mean_and_error, get_iact
import importlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sample_getter
import sys

matplotlib.use('Agg')
# Add the directory that contains the module plotting_functions to sys.path
this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")


def main(config_file_string):
    """config_file_string is the location of the config file"""
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    """nb, argument of parsing.parse_options() must be of type Sequence[str]"""
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, temperature, number_of_equilibration_iterations,
     _, number_of_particles, size_of_particle_space) = helper_methods.get_basic_config_data(config_file_string)
    if not isinstance(size_of_particle_space, list):
        size_of_particle_space = [size_of_particle_space]
    temperatures, sample_directories = None, None
    if potential == "ising_potential":
        sh_file_string = f"{os.path.splitext(config_file_string)[0]}.sh"
        sweep_variable = helper_methods.read_variable_from_sh_file(sh_file_string, "CONFIG_VARIABLE")
        if sweep_variable != "temperature":
            raise ValueError("IsingPotential reference data only available for temperature sweep. "
                             f"CONFIG_VARIABLE must be set to temperature. Provided: {sweep_variable}")
        temperatures = helper_methods.get_temps_from_bash_file(sh_file_string)
        if not (len(temperatures) == 2 and temperatures[0] == 1.2 and temperatures[1] == 3.0):
            raise ValueError("IsingPotential reference data only available for models for which two sampling "
                             "temperatures are given with values 1.2 and 3.")
        sample_directories = [(f"{helper_methods.get_basic_config_data(config_file_string)[4]}/temperature_"
                               f"{temperature_index:02d}/job_00") for temperature_index in range(len(temperatures))]

    try:
        potential_prefactor = parsing.get_value(config, strings.to_camel_case(potential), "prefactor")
    except (NoOptionError, RuntimeError) as _:
        potential_prefactor = 1.0  # set as default value
    combined_potential_prefactor = potential_prefactor / temperature
    """n.b., potentials may include additional prefactors (to beta (1 / temperature) and potential_prefactor in their 
        definitions, e.g. the definitions of ExponentialPowerPotential and GaussianPotential include additional 
        prefactors of 1/power and 1/2, respectively - potential_prefactor defines the relative weight of the potential 
        in question when included in the sum of a more complex model (though multi-sub-potential functionality has not 
        yet been integrated into super-aLby)"""
    if potential == "ising_potential":
        if not (len(samplers) <= 2 and all([sampler == "potential_sampler" or
                                            sampler == "standard_mean_position_sampler" for sampler in samplers])):
            raise ValueError("IsingPotential reference data only available for PotentialSampler and "
                             "StandardMeanPositionSampler.  Please give only these values for samplers in the Mediator "
                             "section.")
    elif potential == "coulomb_potential" or "lennard_jones" in potential:
        if not (len(samplers) == 1 and samplers[0] == "particle_separation_sampler"):
            raise ValueError("Coulomb and Lennard-Jones reference data only available for ParticleSeparationSampler.  "
                             "Please give only this value for samplers in the Mediator section.")
    elif potential == "gaussian_potential" or potential == "exponential_power_potential":
        if not (len(samplers) == 1 and samplers[0] == "standard_position_sampler"):
            raise ValueError("Gaussian and exponential-power reference data only available for StandardPositionSampler."
                             "  Please give only this value for samplers in the Mediator section.")
    elif potential == "xy_potential":
        if not (len(samplers) == 1 and samplers[0] == "xy_magnetisation_norm_sampler"):
            raise ValueError("XY model reference data only available for XyMagnetisationNormSampler."
                             "  Please give only this value for samplers in the Mediator section.")
    elif "harmonic_chain_potential" in potential:
        if not (len(samplers) == 1 and samplers[0] == "structure_factor_sampler"):
            raise ValueError("Harmonic chain model reference data only available for StructureFactorSampler."
                             "  Please give only this value for samplers in the Mediator section.")
    elif potential == "quantum_harmonic_oscillator_potential":
        if not (len(samplers) == 1 and samplers[0] == "mean_squared_position_sampler"):
            raise ValueError("1D quantum harmonic oscillator model reference data only available for "
                             "MeanSquaredPositionSampler. Please give only this value for samplers in the Mediator "
                             "section.")
    elif potential == "hard_disk_potential" and len(size_of_particle_space) == 1:
        if not (len(samplers) == 1 and samplers[0] == "structure_factor_sampler"):
            raise ValueError("1D hard-disk model reference data only available for StructureFactorSampler. "
                             "Please give only this value for samplers in the Mediator section.")
    elif potential == "hard_disk_potential" and len(size_of_particle_space) == 2:
        if not (len(samplers) == 1 and samplers[0] == "pressure_sampler"):
            raise ValueError("2D hard-disk model reference data only available for PressureSampler. "
                             "Please give only this value for samplers in the Mediator section.")
    elif potential == "quantum_hard_disk_potential":
        if not (config_file_mediator == "event_chain_mediator" or config_file_mediator == "metropolis_mediator"):
            raise ValueError("Reference data for the quantum hard-disk model is only available for EventChainMediator "
                             "and MetropolisMediator.  Please choose only one of these mediators in the Run section.")
        if not (len(samplers) == 1 and samplers[0] == "standard_position_sampler"):
            raise ValueError("Reference data for the quantum hard-disk model is only available for PositionSampler.  "
                             "Please give only this value for samplers in the Mediator section.")
    else:
        raise ValueError("Reference data not provided for this potential.")

    exponential_power_exponent = None
    if potential == "exponential_power_potential":
        try:
            exponential_power_exponent = parsing.get_value(config, strings.to_camel_case(potential), "power")
        except (NoOptionError, RuntimeError) as _:
            exponential_power_exponent = 2.0  # set as default value
    if potential == "gaussian_potential" or (potential == "exponential_power_potential" and
                                             exponential_power_exponent == 2.0):
        reference_sample = np.random.normal(0.0, combined_potential_prefactor ** (- 0.5), size=10000)
    elif potential == "exponential_power_potential" and exponential_power_exponent == 4.0:
        if combined_potential_prefactor == 0.25:
            reference_sample = np.load('permanent_data/reference_data/fourth_exponential_power_potential_beta_one_'
                                       'quarter_reference_sample.npy')
        else:
            raise ValueError("ExponentialPowerPotential reference data only available for "
                             "ExponentialPowerPotential._power = 2 or 4.  For the latter, reference data is only "
                             "available for models for which the product of beta (the reciprocal sampling temperature) "
                             "and ExponentialPowerPotential._prefactor is equal to 0.25 (n.b., the potential's "
                             "definition includes an additional prefactor of 1 / power).")
    elif potential == "coulomb_potential":
        if (number_of_particles == 2 and combined_potential_prefactor == 2.0 and
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
        if "lennard_jones_potential_without_cutoff" in potential:
            if (number_of_particles == 2 and combined_potential_prefactor == 2.0 and well_depth == 0.25 and
                    characteristic_length == 1.0 and (
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
            try:
                cutoff_length = parsing.get_value(config, strings.to_camel_case(potential), "cutoff_length")
            except (NoOptionError, RuntimeError) as _:
                cutoff_length = 2.5  # set as default value
            if (number_of_particles == 8 and combined_potential_prefactor == 1.0 and well_depth == 1.0 and
                    characteristic_length == 1.0 and cutoff_length == 3.0 and
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
        if (number_of_particles == 16 and lattice_dimensionality == 2 and
                potential_prefactor * exchange_constant == 1.0 and
                config.get("ModelSettings", "size_of_particle_space") == "2"):
            if config_file_mediator == "metropolis_mediator":
                thinning_level = 10
            else:
                thinning_level = 1
            expected_magnetic_norm_density_reference_values = ["0.997", "0.601"]
            expected_magnetic_norm_susc_per_particle_reference_values = ["0.00566", "0.476"]
            expected_potential_per_particle_reference_values = ["-1.99", "-1.01"]
            expected_spec_heat_per_particle_reference_values = ["0.0663", "0.603"]
            for temperature_index, temperature in enumerate(temperatures):
                print(temperature_index)
                print("---------------------------------")
                print(f"Temperature = {temperature:.4f}")
                for sample_index, sampler in enumerate(samplers):
                    if sampler == "standard_mean_position_sampler":
                        # expected magnetic-norm density is E[|m|] où m = sum_i s_i / N
                        magnetic_norm_density_mean_and_error = get_sample_mean_and_error(
                            sample_getter.get_magnetic_norm_density(sample_directories[temperature_index],
                                                                    temperature,0, number_of_particles,
                                                                    number_of_equilibration_iterations, thinning_level))
                        print(f"Sample estimate of expected magnetic-norm density = "
                              f"{magnetic_norm_density_mean_and_error[0]:.3g} +- "
                              f"{magnetic_norm_density_mean_and_error[1]:.3g} (ref. value = "
                              f"{expected_magnetic_norm_density_reference_values[temperature_index]})")
                        # w/M = Nm the magnetisation, the expected magnetic susc is \nabla_h E[M] = beta N^2 Var[m],
                        # but to compare Swendsen-Wang/Wolff and Metropolis, we estimate beta N Var[|m|] (the expected
                        # magnetic-norm susc (per particle))
                        magnetic_norm_susceptibility_mean_and_error = get_sample_mean_and_error(
                            sample_getter.get_magnetic_norm_susceptibility(sample_directories[temperature_index],
                                                                           temperature,0,
                                                                           number_of_particles,
                                                                           number_of_equilibration_iterations,
                                                                           thinning_level))
                        print(f"Sample estimate of expected magnetic-norm susc. per particle = "
                              f"{magnetic_norm_susceptibility_mean_and_error[0]:.3g} +- "
                              f"{magnetic_norm_susceptibility_mean_and_error[1]:.3g} (ref. value = "
                              f"{expected_magnetic_norm_susc_per_particle_reference_values[temperature_index]})")
                    elif sampler == "potential_sampler":
                        potential_mean_and_error = get_sample_mean_and_error(sample_getter.get_potential(
                            sample_directories[temperature_index], temperature, 0, number_of_particles,
                            number_of_equilibration_iterations, thinning_level))
                        print(f"Sample estimate of expected potential per particle = "
                              f"{potential_mean_and_error[0] / number_of_particles} +- "
                              f"{potential_mean_and_error[1] / number_of_particles} (ref. value = "
                              f"{expected_potential_per_particle_reference_values[temperature_index]})")
                        # expected specific heat is \partial_T E[U] = beta^2 Var[U] (a dimensionless quantity) -- we
                        # estimate beta^2 Var[U] / N (the expected specific heat per particle)
                        specific_heat_mean_and_error = get_sample_mean_and_error(sample_getter.get_specific_heat(
                            sample_directories[temperature_index], temperature, 0, number_of_particles,
                            number_of_equilibration_iterations, thinning_level))
                        print(f"Sample estimate of expected specific heat per particle = "
                              f"{specific_heat_mean_and_error[0] / number_of_particles} +- "
                              f"{specific_heat_mean_and_error[1] / number_of_particles} (ref. value = "
                              f"{expected_spec_heat_per_particle_reference_values[temperature_index]})")
                print("---------------------------------")
        else:
            raise ValueError("IsingPotential reference data only available for 16-particle models on a two-dimensional "
                             "lattice for which size_of_particle_space equals 2 and the product of "
                             "IsingPotential.prefactor and IsingPotential.exchange_constant is equal to 1.0 (n.b., "
                             "number_of_particles and size_of_particle_space are set in the ModelSettings section).")
    elif "xy_potential" in potential:
        reference_sample = np.load(
            "permanent_data/reference_data/xy_8x8_sites_temp_0_point_8_magnetisation_norm_reference_sample.npy")
    elif "harmonic_chain_potential" in potential:
        reference_sample = np.load(
            "permanent_data/reference_data/eight_harmonic_chain_particles_temp_1_L_16.npy").flatten()
    elif "quantum_harmonic_oscillator_potential" in potential:
        try:
            anharmonicity = parsing.get_value(config, strings.to_camel_case(potential), "anharmonicity")
            # anharmonic potential
            reference_sample = np.load("permanent_data/reference_data/quantum_anharmonic_oscillator_m05_dt1_lambda1_"
                                       "wsquared-15_Nt50_Nq1.npy").flatten()
        except:
            # else we have a harmonic potential with anharmonicity = 0
            reference_sample = np.load("permanent_data/reference_data/quantum_harmonic_oscillator_m08_dt15_Nt30_Nq1_"
                                   "reference_sample.npy").flatten()
    elif "hard_disk_potential" in potential and "quantum_hard_disk_potential" not in potential:
        if len(size_of_particle_space) == 1:
            reference_sample = np.load("permanent_data/reference_data/nine_1d_hard_disks_particles_packing_"
                                       "fraction_point_5_reference_sample.npy")
        elif len(size_of_particle_space) == 2:
            reference_sample = np.load("permanent_data/reference_data/"
                                       "nine_2d_hard_disks_particles_packing_fraction_point_688_reference_sample.npy")
    elif "quantum_hard_disk_potential" in potential and "event_chain_mediator" in config_file_mediator:
        reference_sample = np.load("permanent_data/reference_data/ten_quantum_hard_disks_two_timeslices_"
                                   "packing_fraction_point_97_event_chain_reference_sample.npy")
    elif "quantum_hard_disk_potential" in potential and "metropolis_mediator" in config_file_mediator:
        reference_sample = np.load("permanent_data/reference_data/ten_quantum_hard_disks_two_timeslices_"
                                   "packing_fraction_point_9_metropolis_reference_sample.npy")

    if "ising_potential" not in potential:
        reference_cdf = get_cumulative_distribution(reference_sample)
        if "coulomb" in potential or "lennard_jones" in potential:
            sample = sample_getter.get_particle_separations(sample_directory, temperature, 0,
                                                            number_of_particles,
                                                            number_of_equilibration_iterations).flatten()
        elif "xy" in potential:
            sample = sample_getter.get_xy_magnetisation_norm(sample_directory, temperature, 0,
                                                             number_of_particles,
                                                             number_of_equilibration_iterations).flatten()
        elif "harmonic_chain" in potential:
            sample = sample_getter.get_structure_factor(sample_directory, temperature, 0, number_of_particles,
                                                        number_of_equilibration_iterations).flatten()
        elif "quantum_harmonic_oscillator_potential" in potential:
            sample = sample_getter.get_mean_squared_positions(sample_directory, temperature, 0,
                                                              number_of_particles, number_of_equilibration_iterations
                                                              ).flatten()
        elif "hard_disk_potential" in potential and "quantum_hard_disk_potential" not in potential:
            if len(size_of_particle_space) == 1:
                sample = sample_getter.get_structure_factor(sample_directory, temperature,0,
                                                            number_of_particles,
                                                            number_of_equilibration_iterations).flatten()
            elif len(size_of_particle_space) == 2:
                sample = sample_getter.get_pressure(sample_directory, temperature,0, number_of_particles,
                                                    number_of_equilibration_iterations).flatten()
        else:
            sample = sample_getter.get_positions(sample_directory, temperature, 0, number_of_particles,
                                                 number_of_equilibration_iterations).flatten()

        sample_cdf = get_cumulative_distribution(sample)
        #iact = get_iact(sample)
        #ref_iact = get_iact(reference_sample)


        plt.plot(reference_cdf[0], reference_cdf[1], color='r', linewidth=3, linestyle='-',
                 label=f'reference data')#\n IACT: {ref_iact:.3f}')
        plt.plot(sample_cdf[0], sample_cdf[1], color='k', linewidth=2, linestyle='-',
                 label=f'super-aLby data')#\n IACT: {iact:.3f}')
        plt.xlabel(r"$x$", fontsize=15, labelpad=10)
        plt.ylabel(r"$ F_n \left( X < x \right)$", fontsize=15, labelpad=10)
        plt.tick_params(axis='both', which='major', labelsize=14, pad=10)
        legend = plt.legend(loc='lower right', fontsize=10)
        legend.get_frame().set_edgecolor('k')
        legend.get_frame().set_lw(1.5)
        plt.tight_layout()
        #plt.show()
        plt.savefig("convergence_cpp.png")


if __name__ == '__main__':
    main(sys.argv[1])