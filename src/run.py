"""Executable script which runs the super-aLby application based on an input configuration file. This script and most
    of the base package are taken from the JeLLyFysh application, which one of the super-aLby authors co-wrote."""
from base import factory
from base.exceptions import ConfigurationError
from base.logging import set_up_logging, print_and_log
from base.parsing import get_value, parse_options, read_config
from base.strings import to_camel_case
from base.uuid import get_uuid
from version import version
from typing import Sequence, Union
import fileinput
import multiprocessing as mp
import os
import platform
import sys
import time


def print_start_message():
    """Print the start message which includes the copyright."""
    print(f"super-aLby (version {version}) - a Python application for super-relativistic, Hamiltonian, Metropolis, "
          f"Swendsen-Wang and Wolff Monte Carlo in statistical physics and Bayesian computation")
    print("Copyright (C) 2022 The super-aLby organisation")


def main(argv: Sequence[str]) -> None:
    """
    Use the argument strings to run the super-aLby application.

    First, the location of the configuration file is retrieved from the argument strings.  number_of_jobs (the number
    of parallel jobs) and max_number_of_cpus are then read from the configuration file.  If number_of_jobs is one, the
    location of the configuration file is then parsed to run_single_simulation(); otherwise, the location of the
    configuration file is then used to spawn multiple configuration files (each corresponding to one of the parallel
    jobs), each of which are then parsed to run_single_simulation() via mp.starmap().

    In run_single_simulation(), the algorithm (mediator) is built from the relevant instantiated classes, as defined
    in the configuration file.

    Parameters
    ----------
    argv : Sequence[str]
        The argument strings.
    """
    base_config_file_location = argv[0]
    number_of_jobs = get_value(read_config(base_config_file_location), "Run", "number_of_jobs")
    max_number_of_cpus = get_value(read_config(base_config_file_location), "Run", "max_number_of_cpus")
    if number_of_jobs < 1:
        raise ConfigurationError("For the value of number_of_jobs in Run, give an integer not less than one.")
    elif number_of_jobs == 1:
        print("Running a single Markov process.")
        run_single_simulation(base_config_file_location, None, number_of_jobs)
    else:
        number_of_available_cpus = mp.cpu_count()
        if number_of_available_cpus > max_number_of_cpus:
            number_of_cpus = max_number_of_cpus
        else:
            number_of_cpus = number_of_available_cpus
        if number_of_jobs < number_of_cpus:
            print(f"Running {number_of_jobs} Markov processes in parallel on {number_of_jobs} CPUs, where",
                  f"{number_of_available_cpus} CPUs are available.")
            pool = mp.Pool(number_of_jobs)
        else:
            print(f"Running {number_of_jobs} Markov processes in parallel on {number_of_cpus} CPUs, where "
                  f"{number_of_available_cpus} CPUs are available.")
            pool = mp.Pool(number_of_cpus)
        """create directory in which to store temporary copies of the parent config file"""
        os.system(f"mkdir -p {base_config_file_location.replace('.ini', '')}")
        config_file_copies = [base_config_file_location.replace(".ini", f"/job_{job_number:02d}.ini")
                              for job_number in range(number_of_jobs)]
        for job_number, config_file_copy in enumerate(config_file_copies):
            """create temporary copies of parent config file"""
            os.system(f"cp {base_config_file_location} {config_file_copy}")
            for line in fileinput.input(config_file_copy, inplace=True):
                if "output_directory" in line:
                    print(line.replace(' ', '').replace('=', ' = ').replace('\n', '') + f"/job_{job_number:02d}")
                else:
                    print(line, end="")
        pool.starmap(run_single_simulation, [(config_file_copy, job_number, number_of_jobs)
                                             for job_number, config_file_copy in enumerate(config_file_copies)])
        pool.close()
        """delete temporary copies of parent config file"""
        os.system(f"rm -r {base_config_file_location.replace('.ini', '')}")


def run_single_simulation(config_file_location: str, job_number: Union[int, None], number_of_jobs: int) -> None:
    """
    Run a single job / simulation of the super-aLby application.

    The configuration file at config_file_location is parsed. The algorithm (mediator) is then built from the relevant
    instantiated classes, as defined in the configuration file.

    Parameters
    ----------
    config_file_location : str
        A string defining the location of the configuration file.
    job_number: int or None
        The label corresponding to the current parallel job.  If None, number_of_jobs is one.
    number_of_jobs: int
        The total number of parallel jobs.
    """
    args = parse_options([config_file_location])
    logger = set_up_logging(args)

    logger.info(f"Run identification hash: {get_uuid()}")
    logger.info(f"Underlying platform (determined via platform.platform(aliased=True): "
                f"{platform.platform(aliased=True)}")

    if job_number is None:
        print_and_log(logger, f"Setting up the single run based on the configuration file {args.config_file}.")
    else:
        print_and_log(logger, f"Setting up the {get_ordinal(job_number + 1)} of {number_of_jobs} runs based on the "
                              f"configuration file {args.config_file}.")
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

    if job_number is None:
        print_and_log(logger, "Running the single Monte Carlo simulation.")
    else:
        print_and_log(logger, f"Running the {get_ordinal(job_number + 1)} of {number_of_jobs} Monte Carlo simulations.")
    start_time = time.time()
    mediator.generate_sample()
    end_time = time.time()
    print("-----------------------------------------------------------------------------------------")
    if job_number is None:
        print_and_log(logger,
                      f"Total runtime of the simulation (of all temperature values) = {end_time - start_time} seconds.")
    else:
        print_and_log(logger, f"Total runtime of the {get_ordinal(job_number + 1)} of {number_of_jobs} simulations (of "
                              f"all temperature values) = {end_time - start_time} seconds.")
    print("-----------------------------------------------------------------------------------------")


def get_ordinal(integer):
    """Returns a string that states the ordinal of the integer provided."""
    return str(integer) + {1: "st", 2: "nd", 3: "rd"}.get(4 if 10 <= integer % 100 < 20 else integer % 10, "th")


if __name__ == '__main__':
    print_start_message()
    main(sys.argv[1:])
