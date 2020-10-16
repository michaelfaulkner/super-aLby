import logging
from argparse import ArgumentParser, Namespace
import configparser
import json
from configparser import ConfigParser
from typing import Sequence
from version import version


def read_config(config_file: str) -> ConfigParser:
    """
    Parse the configuration file.

    Parameters
    ----------
    config_file : str
        The filename of the configuration file.

    Returns
    -------
    configparser.ConfigParser
        The parsed configuration file.

    Raises
    ------
    RuntimeError
        If the configuration file does not exist.
    """
    config = ConfigParser()
    if not config.read(config_file):
        raise RuntimeError("Given configuration file does not exist.")
    return config


def get_algorithm_config(config):
    return (get_value(config, "Settings", "number_of_equilibration_iterations"),
            get_value(config, "Settings", "number_of_observations"),
            get_value(config, "Settings", "initial_step_size"),
            get_value(config, "Settings", "max_number_of_integration_steps"),
            get_value(config, "Settings", "randomise_number_of_integration_steps"),
            get_value(config, "Settings", "randomise_initial_momenta"),
            get_value(config, "Settings", "randomise_initial_position"),
            get_value(config, "Settings", "step_size_adaptor_is_on"),
            get_value(config, "Settings", "use_metropolis_accept_reject"))


def get_value(config_file, section_name, option_name):
    try:
        return json.loads(config_file.get(section_name, option_name))  # Get lists of integers and floats
    except configparser.NoSectionError:
        raise RuntimeError("Obligatory section '%s' not given in .ini file" % section_name)
    except configparser.NoOptionError:
        raise RuntimeError(
            "Obligatory option '%s' in section '%s' not given in .ini file." % (option_name, section_name))
    except ValueError:
        pass
    try:
        return config_file.getboolean(section_name, option_name)
    except ValueError:
        pass
    value = config_file.get(section_name, option_name)
    if value == "None":
        return None
    return value


def add_general_parser_arguments(parser: ArgumentParser) -> None:
    """
    Add parser arguments to the command line argument parser.

    This method adds the following possible arguments:
    1. --version, -V: Print the version of the application and exit.
    2. --verbose, -v: Increase verbosity of logging messages. Multiple -v options increase the verbosity. The maximum
    is 2.
    3. --logfile LOGFILE, -l LOGFILE: Specify the logging file.
    Per default, also the following argument is added:
    4. --help, -h: Show the help message and exit.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser.
    """
    parser.add_argument(
        "-V", "--version", action="version", version="super-relativistic-mc application version " + version)
    parser.add_argument("-v", "--verbose", action="count",
                        help="increase verbosity of logging messages "
                             "(multiple -v options increase the verbosity, the maximum is 2)")
    parser.add_argument("-l", "--logfile", action="store", help="specify the logging file")


def parse_options(args: Sequence[str]) -> Namespace:
    """
    Convert argument strings to objects and assign them as attributes of the argparse namespace. Return the populated
    namespace.

    The argument strings can for example be sys.argv[1:] in order to parse the command line arguments. The argument
    parser parses the arguments specified in the add_general_parser_arguments function. This function also adds the
    configuration file as an obligatory positional argument.

    Parameters
    ----------
    args : Sequence[str]
        The argument strings.

    Returns
    -------
    argparse.Namespace
        The populated argparse namespace.
    """
    parser = ArgumentParser()
    parser.add_argument("config_file", help="specify the path to config.ini file")
    add_general_parser_arguments(parser)
    return parser.parse_args(args)


def set_up_logging(parsed_arguments: Namespace) -> logging.Logger:
    """
    Set up the logging based on the populated argparse namespace.

    The level of the root logger is set based on the number of -v arguments parsed. For zero -v arguments, it is set
    to logging.WARNING, for one to logging.INFO and for two to logging.DEBUG. If a log file was specified in the
    parsed arguments, the logging information is written into that file.

    Parameters
    ----------
    parsed_arguments : argparse.Namespace
        The populated argparse namespace.

    Returns
    -------
    logging.Logger
        The initialized root logger.
    """
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)

    if parsed_arguments.verbose is None:
        logger_level = logging.WARNING
    elif parsed_arguments.verbose == 1:
        logger_level = logging.INFO
    else:
        logger_level = logging.DEBUG

    if parsed_arguments.logfile is not None:
        handler = logging.FileHandler(parsed_arguments.logfile)
    else:
        handler = logging.StreamHandler()
    logger.setLevel(logger_level)
    handler.setLevel(logger_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Don't collect information about where calls were made from (slows down PyPy)
    logging._srcfile = None
    return logger
