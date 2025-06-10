"""Helper methods used in the main package and/or some sample analysis script(s)."""
import importlib
import math
import numpy as np
import os
import sys
from base.exceptions import ConfigurationError
from configparser import NoSectionError

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
    if type(config_file_string) is str:
        """nb, argument of parsing.parse_options() must be of type Sequence[str]"""
        config_file_string = [config_file_string]
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    possible_mediators = ["UnboundedLeapfrogMediator", "ToroidalLeapfrogMediator", "LazyToroidalLeapfrogMediator",
                          "MetropolisMediator", "SwendsenWangMediator", "WolffMediator", "EventChainMediator"]
    (config_file_mediator, potential, samplers, temperatures, number_of_equilibration_iterations,
     number_of_observations, size_of_particle_space) = (None, None, None, None, None, None, None)
    for possible_mediator in possible_mediators:
        try:
            potential = config.get(possible_mediator, "potential")
            if "hard_disk_potential" in str(potential) and "quantum_hard_disk_potential" not in str(potential):
                number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")
                packing_fraction = parsing.get_value(config, "HardDiskPotential", "packing_fraction")
                disk_radius = parsing.get_value(config, "HardDiskPotential", "disk_radius")
                linear_system_size = math.sqrt(number_of_particles * math.pi / packing_fraction) * disk_radius
                size_of_particle_space = [linear_system_size, linear_system_size]
            else:
                size_of_particle_space = parsing.get_value(config, "ModelSettings", "size_of_particle_space")
            if ("quantum_hard_disk_potential" in str(potential) or
                    "quantum_harmonic_oscillator_potential" in str(potential)):
                try:
                    number_of_quantum_particles = parsing.get_value(config, "ModelSettings",
                                                                    "number_of_quantum_particles")
                    number_of_timeslices = parsing.get_value(config, "ModelSettings", "number_of_timeslices")
                except:
                    raise ConfigurationError(
                        "Do not give a value for number_of_particles for worldline Monte Carlo.  Instead set "
                        "number_of_quantum_particles and number_of_timeslices.  number_of_particles is then calculated "
                        "via number_of_particles = number_of_quantum_particles * number_of_timeslices.")
                number_of_particles = number_of_quantum_particles * number_of_timeslices
            else:
                number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles")
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
        raise ConfigurationError("Mediator not one of UnboundedLeapfrogMediator, ToroidalLeapfrogMediator, "
                                 "LazyToroidalLeapfrogMediator, MetropolisMediator, SwendsenWangMediator, "
                                 "WolffMediator or EventChainMediator.")
    sample_directories = [config.get(strings.to_camel_case(sampler), "output_directory") for sampler in samplers]
    return (config_file_mediator, potential, samplers, sample_directories, temperatures,
            number_of_equilibration_iterations, number_of_observations, number_of_particles, size_of_particle_space,
            parsing.get_value(config, "Run", "number_of_jobs"), parsing.get_value(config, "Run", "max_number_of_cpus"))


def get_neighbours(lattice_site_index, lattice_length):
    """Returns a list of the four neighbours (on the 2D lattice) of lattice_site_index"""
    return [get_east_neighbour(lattice_site_index, lattice_length),
            get_north_neighbour(lattice_site_index, lattice_length),
            get_west_neighbour(lattice_site_index, lattice_length),
            get_south_neighbour(lattice_site_index, lattice_length)]


def get_east_neighbour(lattice_site_index, lattice_length):
    """Returns the eastwards neighbour (on the 2D lattice) of lattice_site_index"""
    return lattice_site_index + (
            lattice_site_index + 1) % lattice_length - lattice_site_index % lattice_length


def get_north_neighbour(lattice_site_index, lattice_length):
    """Returns the northwards neighbour (on the 2D lattice) of lattice_site_index"""
    return lattice_site_index + lattice_length * (
            (int(lattice_site_index / lattice_length) + 1) % lattice_length -
            (int(lattice_site_index / lattice_length)) % lattice_length)


def get_west_neighbour(lattice_site_index, lattice_length):
    """Returns the westwards neighbour (on the 2D lattice) of lattice_site_index"""
    return lattice_site_index + (lattice_site_index - 1 + lattice_length) % lattice_length - (
            lattice_site_index + lattice_length) % lattice_length


def get_south_neighbour(lattice_site_index, lattice_length):
    """Returns the southwards neighbour (on the 2D lattice) of lattice_site_index"""
    return lattice_site_index + lattice_length * (
            (int(lattice_site_index / lattice_length) + lattice_length - 1) % lattice_length -
            (int(lattice_site_index / lattice_length) + lattice_length) % lattice_length)


def get_initial_positions_of_smooth_potential(potential_class):
    """
    Returns the initial positions array for a smooth potential function.

    Parameters
    ----------
    potential_class : class instance
        The potential class.

    Returns
    -------
    numpy.ndarray
        A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
        is a float and represents one Cartesian component of the position of a single particle, e.g., two particles
        (confined to one-dimensional space) at positions 0.0 and 1.0 is represented by [[0.0] [1.0]]; three
        particles (confined to two-dimensional space) at positions (0.0, 1.0), (2.0, 3.0) and (- 1.0, - 2.0) is
        represented by [[0.0 1.0] [2.0 3.0] [-1.0 -2.0]].
    """
    """NB, we import from model_settings within this function to avoid circular imports."""
    from model_settings import dimensionality_of_particle_space, number_of_particles, \
        range_of_initial_particle_positions
    if dimensionality_of_particle_space == 1:
        if not (range_of_initial_particle_positions is None or type(range_of_initial_particle_positions) is float or
                (type(range_of_initial_particle_positions) is list and
                 len(range_of_initial_particle_positions) == 2 and
                 [type(bound) is float for bound in range_of_initial_particle_positions])):
            raise ConfigurationError(
                f"Give either None (indicating that the initial position is drawn from the real line), a float "
                f"(representing a precise initial position for each particle) or a list of two floats "
                f"(representing the bounds of the interval from which each initial particle position is randomly "
                f"chosen) for the value of range_of_initial_particle_positions in the ModelSettings section when "
                f"using {potential_class} (or any child class of ContinuousPotential) with a "
                f"one-dimensional particle space.")
        if range_of_initial_particle_positions is None:
            return np.array([np.atleast_1d(np.random.normal()) for _ in range(number_of_particles)])
        elif type(range_of_initial_particle_positions) is float:
            return np.array(
                [np.atleast_1d(range_of_initial_particle_positions) for _ in range(number_of_particles)])
        else:
            return np.array([np.atleast_1d(np.random.uniform(*range_of_initial_particle_positions))
                             for _ in range(number_of_particles)])
    else:
        if not (type(range_of_initial_particle_positions) is list and
                len(range_of_initial_particle_positions) == dimensionality_of_particle_space and
                ([component is None for component in range_of_initial_particle_positions] or
                 [type(component) is float for component in range_of_initial_particle_positions] or
                 [type(component) is list and len(component) == 2 and type(bound) is float
                  for component in range_of_initial_particle_positions for bound in component])):
            raise ConfigurationError(
                f"Give a list of length dimensionality_of_particle_space for range_of_initial_particle_positions "
                f"in the ModelSettings section when using {potential_class} (or any child class of "
                f"ContinuousPotential) with a particle space of dimension greater than one.  Each element of the "
                f"list corresponds to a Cartesian component of each particle position and must be either None "
                f"(indicating that the initial Cartesian component is drawn from the real line), a float "
                f"(representing a precise initial Cartesian component) or a list of two floats (representing the "
                f"bounds of the interval from which the initial Cartesian component is randomly chosen).")
        if range_of_initial_particle_positions[0] is None:
            return np.array([np.random.normal(size=dimensionality_of_particle_space)
                             for _ in range(number_of_particles)])
        elif type(range_of_initial_particle_positions[0]) is float:
            return np.array([range_of_initial_particle_positions for _ in range(number_of_particles)])
        else:
            return np.array([[np.random.uniform(*axis_range) for axis_range in range_of_initial_particle_positions]
                             for _ in range(number_of_particles)])

def get_east_neighbour_worldline(lattice_site_index, number_of_timeslices, number_of_quantum_particles):
    """Returns the eastwards timeslice neighbour of lattice_site_index in the quantum hard disks model"""

    return int((lattice_site_index + number_of_quantum_particles + 1.0e-12) %
                (number_of_timeslices * number_of_quantum_particles))

def get_west_neighbour_worldline(lattice_site_index, number_of_timeslices, number_of_quantum_particles):
    """Returns the eastwards timeslice neighbour of lattice_site_index in the quantum hard disks model"""

    return int((lattice_site_index - number_of_quantum_particles + 1.0e-12) % 
               (number_of_timeslices * number_of_quantum_particles))