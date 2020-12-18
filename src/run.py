"""Executable script which runs the super-aLby application based on an input configuration file. This script and most of
    the base package are taken from the JeLLyFysh application, on which two of the super-aLby authors worked."""
import platform
import sys
import time
from typing import Sequence
from base import factory
from base.strings import to_camel_case
from base.uuid import get_uuid
from base.parsing import get_algorithm_settings, parse_options, read_config
from base.logging import set_up_logging, print_and_log
from version import version
import mediator


def main(argv: Sequence[str]) -> None:
    """
    Use the argument strings to run the super-aLby application.

    First the argument strings are parsed. The configuration file specified in the argument strings is parsed. Based on
    the configuration file, the algorithm is built from the relevant instantiated classes.

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
    integrator = factory.build_from_config(config, to_camel_case(config.get("Mediator", "integrator")), "integrator")
    kinetic_energy = factory.build_from_config(config, to_camel_case(config.get("Mediator", "kinetic_energy")),
                                               "kinetic_energy")
    potential = factory.build_from_config(config, to_camel_case(config.get("Mediator", "potential")), "potential")
    sampler = factory.build_from_config(config, to_camel_case(config.get("Mediator", "sampler")), "sampler")
    algorithm = mediator.Mediator(integrator, kinetic_energy, potential, sampler, *get_algorithm_settings(config))

    used_sections = factory.used_sections
    for section in config.sections():
        if section not in used_sections and section not in ["Mediator", "ModelSettings",  "AlgorithmSettings"]:
            logger.warning("The section {0} in the .ini file has not been used!".format(section))

    print_and_log(logger, "Running the super-relativistic Monte Carlo simulation.")
    start_time = time.time()
    sample = algorithm.get_sample()
    end_time = time.time()

    print_and_log(logger, "Running the post-run methods.")
    sampler.output_sample(sample)
    print_and_log(logger, f"Runtime of the simulation: --- {end_time - start_time} seconds ---")


def print_start_message():
    """Print the start message which includes the copyright."""
    print(f"super-aLby (version {version}) - a Python application for super-relativistic Monte Carlo")
    print("Copyright (C) 2021 The super-aLby organisation")


if __name__ == '__main__':
    print_start_message()
    main(sys.argv[1:])
