"""Executable script which runs the super-relativistic-mc application based on an input configuration file. This script
    and the base package are both taken from the JeLLyFysh application, on which two of the 2-2authors worked."""
import platform
import sys
import time
from typing import Sequence
from base import factory
from base.strings import to_camel_case
from base.uuid import get_uuid
from config_file_parsing_and_logging import get_algorithm_config, parse_options, set_up_logging, read_config
from version import version
import integrator.leapfrog_integrator
import markov_chain
import numpy as np


def print_start_message():
    """Print the start message which includes the copyright."""
    print(
        "super-relativistic-mc (version {0}) - a Python application for super-relativistic Monte Carlo".format(version))
    print("Copyright (C) 2020 The Super-relativistic Monte Carlo organization")


def main(argv: Sequence[str]) -> None:
    """
    Use the argument strings to run the super-relativistic-mc application.

    First the argument strings are parsed. The configuration file specified in the argument strings is parsed. Based on
    the configuration file, the algorithm is built from the relevant instantiated classes.

    Parameters
    ----------
    argv : Sequence[str]
        The argument strings.
    """
    args = parse_options(argv)
    logger = set_up_logging(args)

    logger.info("Run identification hash: {0}".format(get_uuid()))
    logger.info("Underlying platform (determined via platform.platform(aliased=True): {0}"
                .format(platform.platform(aliased=True)))

    logger.info("Setting up the run based on the configuration file {0}.".format(args.config_file))
    config = read_config(args.config_file)
    print(config)
    algorithm_config = get_algorithm_config(config)
    print(algorithm_config)
    kinetic_energy_instance = factory.build_from_config(
        config, to_camel_case(config.get("Hamiltonian", "kinetic_energy")), "kinetic_energy")
    potential_instance = factory.build_from_config(
        config, to_camel_case(config.get("Hamiltonian", "potential")), "potential")
    integrator_instance = integrator.leapfrog_integrator.LeapfrogIntegrator(kinetic_energy_instance, potential_instance)
    markov_chain_instance = markov_chain.MarkovChain(integrator_instance, kinetic_energy_instance, potential_instance,
                                                     *get_algorithm_config(config))
    support_variable = np.zeros(config.get("Hamiltonian", "dimension_of_target_distribution"))

    used_sections = factory.used_sections
    for section in config.sections():
        if section not in used_sections and section != "Algorithm" and section != "Hamiltonian":
            logger.warning("The section {0} in the .ini file has not been used!".format(section))

    logger.info("Running the super-relativistic Monte Carlo simulation.")
    start_time = time.time()
    (momentum_sample, support_variable_sample, adapted_step_size, acceptance_rate,
     number_of_numerical_divergences_during_equilibration,
     number_of_numerical_divergences_during_equilibrated_process) = markov_chain_instance.run(support_variable)
    end_time = time.time()

    logger.info("Running the post_run method.")
    # mediator.post_run()
    logger.info("Runtime of the simulation: --- %s seconds ---" % (end_time - start_time))


if __name__ == '__main__':
    print_start_message()
    main(sys.argv[1:])
