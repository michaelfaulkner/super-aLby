from base.exceptions import ConfigurationError
from base.parsing import parse_options, read_config, get_value
import importlib
import numpy as np
import sys
helper_methods = importlib.import_module("helper_methods")


args = parse_options(sys.argv[1:])
(_, potential, factor_field, _, _, _, _, _, number_of_particles, size_of_particle_space
) = helper_methods.get_basic_config_data(args.config_file)
number_of_particle_pairs = int(number_of_particles * (number_of_particles - 1) / 2)
if size_of_particle_space is None or type(size_of_particle_space) is float or type(size_of_particle_space) is int:
    dimensionality_of_particle_space = 1
else:
    dimensionality_of_particle_space = len(size_of_particle_space)
system_volume = 1.0
if type(size_of_particle_space) is list:
    for component in size_of_particle_space:
        system_volume *= component
elif type(size_of_particle_space) is float:
    system_volume *= size_of_particle_space
else:
    system_volume = None
dimensionality_of_momenta_array = (number_of_particles, dimensionality_of_particle_space)
number_of_momenta_components = number_of_particles * dimensionality_of_particle_space
if factor_field is not None and "xy_factor_field" in factor_field and "xy_potential" not in potential:
    raise ConfigurationError(f"XyFactorField can only be combined with XyPotential. Selected: {potential}")

with open(args.config_file) as config_file:
    config_file_as_str = config_file.read()
    config = read_config(args.config_file)
    if "HardDiskPotential" in config_file_as_str:
        if ("size_of_particle_space" in config_file_as_str or
                "range_of_initial_particle_positions" in config_file_as_str):
            raise ConfigurationError(
                f"When using HardDiskPotential, do not include size_of_particle_space or "
                f"range_of_initial_particle_positions in the ModelSettings section of the configuration file. "
                f"size_of_particle_space is set by packing_fraction and number_of_particles; the initial disk "
                f"positions must be ordered to avoid disk overlaps.")
    else:
        range_of_initial_particle_positions = get_value(config, "ModelSettings", "range_of_initial_particle_positions")
        if dimensionality_of_particle_space == 1 and size_of_particle_space is not None:
            if type(range_of_initial_particle_positions) is float or type(range_of_initial_particle_positions) is int \
                or range_of_initial_particle_positions is None:
                conditions = abs(range_of_initial_particle_positions) <= size_of_particle_space / 2
            else:
                conditions = (range_of_initial_particle_positions[0] >= - size_of_particle_space / 2 and
                              range_of_initial_particle_positions[1] <= size_of_particle_space / 2)
        elif dimensionality_of_particle_space > 1 and size_of_particle_space[0] is not None:
            if (type(range_of_initial_particle_positions[0]) is float or
                    type(range_of_initial_particle_positions[0]) is int):
                conditions = [abs(range_of_initial_particle_positions[i]) <= size_of_particle_space[i] / 2
                              for i in range(len(size_of_particle_space))]
            else:
                conditions = [range_of_initial_particle_positions[i][0] >= - size_of_particle_space[i] / 2 and
                              range_of_initial_particle_positions[i][1] <= size_of_particle_space[i] / 2
                              for i in range(len(size_of_particle_space))]
            for condition in np.atleast_1d(conditions):
                if not condition:
                    raise ConfigurationError(
                        "The absolute value of any float or integer given within range_of_initial_particle_positions "
                        "must be less than half the size_of_particle_space.")
    if "QuantumHardDiskPotential" in config_file_as_str or "QuantumHarmonicOscillatorPotential" in config_file_as_str:
        number_of_quantum_particles = get_value(config, "ModelSettings", "number_of_quantum_particles")
        number_of_timeslices = get_value(config, "ModelSettings", "number_of_timeslices")
    else:
        number_of_quantum_particles = None
        number_of_timeslices = None

size_of_particle_space = np.atleast_1d(size_of_particle_space)
size_of_particle_space.flags.writeable = False
