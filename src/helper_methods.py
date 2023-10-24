"""Helper methods used in the main package and/or some sample analysis script(s)."""
from base.exceptions import ConfigurationError
from configparser import NoSectionError
import importlib
import os
import sys

# Add the directory that contains the module plotting_functions to sys.path
this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")


def get_temperatures(minimum_temperature, maximum_temperature, number_of_temperature_increments):
    """Creates a list of the temperatures over which super-aLby iterates"""
    if minimum_temperature < 0.0:
        raise ValueError("Give a value not less than 0.0 as minimum_temperature in helper_methods.get_temperatures().")
    if maximum_temperature < 0.0:
        raise ValueError("Give a value not less than 0.0 as maximum_temperature in helper_methods.get_temperatures().")
    if maximum_temperature < minimum_temperature:
        raise ValueError("Give values of minimum_temperature and maximum_temperature in "
                         "helper_methods.get_temperatures() such that the value of maximum_temperature is not less "
                         "than the value of minimum_temperature..")
    if number_of_temperature_increments < 0:
        raise ValueError("Give a value not less than 0 as number_of_temperature_increments in "
                         "helper_methods.get_temperatures().")
    if number_of_temperature_increments == 0 and minimum_temperature != maximum_temperature:
        raise ValueError("As the value of number_of_temperature_increments is equal to 0, give equal values of "
                         "minimum_temperature and maximum_temperature in helper_methods.get_temperatures().")
    if number_of_temperature_increments == 0:
        return [minimum_temperature]
    temperature_increment = (maximum_temperature - minimum_temperature) / number_of_temperature_increments
    return [minimum_temperature + temperature_increment * temperature_index
            for temperature_index in range(number_of_temperature_increments + 1)]


def get_basic_config_data(config_file_string):
    if type(config_file_string) == str:
        """nb, argument of parsing.parse_options() must be of type Sequence[str]"""
        config_file_string = [config_file_string]
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    possible_mediators = ["EuclideanLeapfrogMediator", "ToroidalLeapfrogMediator", "LazyToroidalLeapfrogMediator",
                          "MetropolisMediator", "SwendsenWangMediator", "WolffMediator"]
    (config_file_mediator, potential, samplers, temperatures, number_of_equilibration_iterations,
     number_of_observations) = (None, None, None, None, None, None)
    for possible_mediator in possible_mediators:
        try:
            potential = config.get(possible_mediator, "potential")
            samplers = config.get(possible_mediator, "samplers").replace(" ", "").split(",")
            temperatures = get_temperatures(parsing.get_value(config, possible_mediator, "minimum_temperature"),
                                            parsing.get_value(config, possible_mediator, "maximum_temperature"),
                                            parsing.get_value(config, possible_mediator,
                                                              "number_of_temperature_increments"))
            number_of_equilibration_iterations = parsing.get_value(config, possible_mediator,
                                                                   "number_of_equilibration_iterations")
            number_of_observations = parsing.get_value(config, possible_mediator, "number_of_observations")
            config_file_mediator = strings.to_snake_case(possible_mediator)
            break
        except NoSectionError:
            continue
    if potential is None:
        raise ConfigurationError("Mediator not one of EuclideanLeapfrogMediator, ToroidalLeapfrogMediator, "
                                 "LazyToroidalLeapfrogMediator, MetropolisMediator, SwendsenWangMediator or "
                                 "WolffMediator.")
    sample_directories = [config.get(strings.to_camel_case(sampler), "output_directory") for sampler in samplers]
    return (config_file_mediator, potential, samplers, sample_directories, temperatures,
            number_of_equilibration_iterations, number_of_observations,
            parsing.get_value(config, "ModelSettings", "number_of_particles"), 
            parsing.get_value(config, "ModelSettings", "size_of_particle_space"),
            parsing.get_value(config, "Run", "number_of_jobs"),
            parsing.get_value(config, "Run", "max_number_of_cpus"))
