"""Executable script which runs the super-aLby application based on an input configuration file. This script and most
    of the base package are taken from the JeLLyFysh application, which one of the super-aLby authors co-wrote."""
from base import factory
from base.exceptions import ConfigurationError
from base.logging import set_up_logging, print_and_log
from base.parsing import parse_options, read_config
from base.strings import to_camel_case
from base.uuid import get_uuid
from helper_methods import get_basic_config_data
from version import version
from typing import Sequence
import os
import platform
import sys
import time


def print_start_message():
    """Print the start message which includes the copyright."""
    print(f"super-aLby (version {version}) - a Python application for various Monte Carlo sampling algorithms in "
          f"statistical physics and Bayesian computation")
    print("Copyright (C) 2025 The super-aLby organisation")


def main(argv: Sequence[str]) -> None:
    """
    Use the argument strings to run the super-aLby application.

    The location of the configuration file is retrieved from the argument strings.  The algorithm (mediator) is then
    built from the relevant instantiated classes, as defined in the configuration file.  The simulation is then run.

    Parameters
    ----------
    argv : Sequence[str]
        The argument strings.
    """
    config_file_location = argv[0]
    args = parse_options([config_file_location])
    logger = set_up_logging(args)
    logger.info(f"Run identification hash: {get_uuid()}")
    logger.info(f"Underlying platform (determined via platform.platform(aliased=True): "
                f"{platform.platform(aliased=True)}")
    
    config = read_config(args.config_file)
    mediator = factory.build_from_config(config, to_camel_case(config.get("Run", "mediator")), "mediator")
    output_directory = get_basic_config_data(config_file_location)[4][0]
    restart_flag = os.path.isfile(os.path.join(os.getcwd(), output_directory, "checkpoint_index.txt"))

    if restart_flag:
        print_and_log(logger,f"Restarting the simulation (based on the configuration file {args.config_file}) "
                      f"from checkpoint {mediator.get_checkpoint_index() - 1} using the checkpoint configuration "
                      f"saved at {os.path.join(os.getcwd(), output_directory, 'configuration_at_checkpoint.npy')}")
    else:
        print_and_log(logger,f"Setting up the Markov process based on the configuration file {args.config_file}.")

    used_sections = factory.used_sections
    for section in config.sections():
        if section not in used_sections and section not in ["Run", "ModelSettings"]:
            logger.warning("The section {0} in the configuration file has not been used!".format(section))
    if config.get("Run", "mediator") == "lazy_toroidal_leapfrog_mediator":
        if config.get("LazyToroidalLeapfrogMediator", "potential") == "lennard_jones_potential_with_linked_lists":
            raise ConfigurationError(f"When using LennardJonesPotentialWithLinkedLists, give a value of "
                                     f"lazy_toroidal_leapfrog_mediator for mediator in the [Run] section of the "
                                     f"configuration file.")
    print_and_log(logger, "Starting the Markov process.")

    print("-----------------------------------------------------------------------------------------")
    start_time = time.time()
    mediator.generate_sample(restart_flag)
    end_time = time.time()
    print("-----------------------------------------------------------------------------------------")
    print_and_log(logger,f"Total runtime of the simulation = {end_time - start_time} seconds.")
    print("-----------------------------------------------------------------------------------------")


def get_ordinal(integer):
    """Returns a string that states the ordinal of the integer provided."""
    return str(integer) + {1: "st", 2: "nd", 3: "rd"}.get(4 if 10 <= integer % 100 < 20 else integer % 10, "th")


if __name__ == '__main__':
    print_start_message()
    main(sys.argv[1:])
