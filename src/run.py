"""Executable script which runs the super-aLby application based on an input configuration file. This script and most
    of the base package are taken from the JeLLyFysh application, which one of the super-aLby authors co-wrote."""
from base import factory
from base.exceptions import ConfigurationError
from base.logging import set_up_logging, print_and_log
from base.parsing import parse_options, read_config
from base.strings import to_camel_case
from base.uuid import get_uuid
from version import version
from typing import Sequence
import platform
import sys
import time


def main(argv: Sequence[str]) -> None:
    """
    Use the argument strings to run the super-aLby application.

    First the argument strings are parsed. The configuration file specified in the argument strings is parsed. Based on
    the configuration file, the algorithm (mediator) is built from the relevant instantiated classes.

    Parameters
    ----------
    argv : Sequence[str]
        The argument strings.
    """
    args = parse_options(argv)
    logger = set_up_logging(args)

    logger.info(f"Run identification hash: {get_uuid()}")
    logger.info(f"Underlying platform (determined via platform.platform(aliased=True): "
                f"{platform.platform(aliased=True)}")

    print_and_log(logger, f"Setting up the run based on the configuration file {args.config_file}.")
    config = read_config(args.config_file)
    mediator = factory.build_from_config(config, to_camel_case(config.get("Run", "mediator")), "mediator")

    used_sections = factory.used_sections
    for section in config.sections():
        if section not in used_sections and section not in ["Run", "ModelSettings"]:
            logger.warning("The section {0} in the configuration file has not been used!".format(section))

    if config.get("Run", "mediator") == "lazy_toroidal_leapfrog_mediator":
        if config.get("LazyToroidalLeapfrogMediator", "potential") == "lennard_jones_potential_with_linked_lists":
            raise ConfigurationError(f"When using LennardJonesPotentialWithLinkedLists, give a value of "
                                     f"lazy_toroidal_leapfrog_mediator for mediator in the [Run] section of the "
                                     f"configuration file.")

    print_and_log(logger, "Running the Monte Carlo simulation.")
    start_time = time.time()
    mediator.generate_sample()
    end_time = time.time()
    print("-----------------------------------------------------------------------------------------")
    print_and_log(logger,
                  f"Total runtime of the simulation (of all temperature values) = {end_time - start_time} seconds.")
    print("-----------------------------------------------------------------------------------------")


def print_start_message():
    """Print the start message which includes the copyright."""
    print(f"super-aLby (version {version}) - a Python application for super-relativistic, Hamiltonian and Metropolis "
          f"Monte Carlo")
    print("Copyright (C) 2022 The super-aLby organisation")


if __name__ == '__main__':
    print_start_message()
    main(sys.argv[1:])
