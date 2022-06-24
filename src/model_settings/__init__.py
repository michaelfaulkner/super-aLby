from base.exceptions import ConfigurationError
from base.parsing import parse_options, read_config, get_value
import numpy as np
import sys


args = parse_options(sys.argv[1:])
config = read_config(args.config_file)
beta = get_value(config, "ModelSettings", "beta")
number_of_particles = get_value(config, "ModelSettings", "number_of_particles")
size_of_particle_space = get_value(config, "ModelSettings", "size_of_particle_space")
range_of_initial_particle_positions = get_value(config, "ModelSettings", "range_of_initial_particle_positions")
# todo log the above values?

one_over_beta = 1.0 / beta
one_over_root_beta = beta ** (-0.5)

number_of_particle_pairs = int(number_of_particles * (number_of_particles - 1) / 2)
if size_of_particle_space is None or type(size_of_particle_space) == float or type(size_of_particle_space) == int:
    dimensionality_of_particle_space = 1
else:
    dimensionality_of_particle_space = len(size_of_particle_space)
dimensionality_of_momenta_array = (number_of_particles, dimensionality_of_particle_space)
number_of_momenta_components = number_of_particles * dimensionality_of_particle_space

if dimensionality_of_particle_space == 1 and size_of_particle_space is not None:
    if type(range_of_initial_particle_positions) == float or type(range_of_initial_particle_positions) == int:
        conditions = abs(range_of_initial_particle_positions) <= size_of_particle_space / 2
    else:
        conditions = (range_of_initial_particle_positions[0] >= - size_of_particle_space / 2 and
                      range_of_initial_particle_positions[1] <= size_of_particle_space / 2)
elif dimensionality_of_particle_space > 1 and size_of_particle_space[0] is not None:
    if type(range_of_initial_particle_positions[0]) == float or type(range_of_initial_particle_positions[0]) == int:
        conditions = [abs(range_of_initial_particle_positions[i]) <= size_of_particle_space[i] / 2
                      for i in range(len(size_of_particle_space))]
    else:
        conditions = [range_of_initial_particle_positions[i][0] >= - size_of_particle_space[i] / 2 and
                      range_of_initial_particle_positions[i][1] <= size_of_particle_space[i] / 2
                      for i in range(len(size_of_particle_space))]
    for condition in np.atleast_1d(conditions):
        if not condition:
            raise ConfigurationError(
                "The absolute value of any float or integer given within range_of_initial_particle_positions must be "
                "less than half the size_of_particle_space.")

if (dimensionality_of_particle_space == 1 and type(size_of_particle_space) == float or
        (dimensionality_of_particle_space > 1 and type(size_of_particle_space) == list and
         type(size_of_particle_space[0]) == float)):
    size_of_particle_space_over_two = 0.5 * np.atleast_1d(size_of_particle_space)
elif (dimensionality_of_particle_space == 1 and type(size_of_particle_space) == int or
        (dimensionality_of_particle_space > 1 and type(size_of_particle_space) == list and
         type(size_of_particle_space[0]) == int)):
    size_of_particle_space_over_two = np.int(0.5 * np.atleast_1d(size_of_particle_space))
else:
    size_of_particle_space_over_two = np.atleast_1d(size_of_particle_space)
size_of_particle_space_over_two.flags.writeable = False
size_of_particle_space = np.atleast_1d(size_of_particle_space)
size_of_particle_space.flags.writeable = False
