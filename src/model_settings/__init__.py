from base.exceptions import ConfigurationError
from base.parsing import parse_options, read_config, get_value
import numpy as np
import sys


args = parse_options(sys.argv[1:])
config = read_config(args.config_file)
beta = get_value(config, "ModelSettings", "beta")  # todo integrate beta into code
number_of_particles = get_value(config, "ModelSettings", "number_of_particles")
size_of_particle_space = get_value(config, "ModelSettings", "size_of_particle_space")
range_of_initial_particle_positions = get_value(config, "ModelSettings", "range_of_initial_particle_positions")
# todo log the above values?

one_over_beta = 1.0 / beta
one_over_root_beta = beta ** (-0.5)

number_of_particle_pairs = int(number_of_particles * (number_of_particles - 1) / 2)

if size_of_particle_space is None or type(size_of_particle_space) == float:
    dimensionality_of_particle_space = 1
else:
    dimensionality_of_particle_space = len(size_of_particle_space)

if dimensionality_of_particle_space == 1:
    dimensionality_of_momenta_array = number_of_particles
else:
    dimensionality_of_momenta_array = (number_of_particles, dimensionality_of_particle_space)

conditions_1 = (dimensionality_of_particle_space == 1 and
                (type(range_of_initial_particle_positions) == float or
                 (type(range_of_initial_particle_positions) == list and
                  len(range_of_initial_particle_positions) == 2 and
                  [type(bound) == float for bound in range_of_initial_particle_positions])))
conditions_2 = (1 < dimensionality_of_particle_space == len(range_of_initial_particle_positions) and
                type(range_of_initial_particle_positions) == list and
                ([type(component) == float for component in range_of_initial_particle_positions] or
                [type(component) == list and len(component) == 2 and type(bound) == float
                 for component in range_of_initial_particle_positions for bound in component]))
# todo fix the bug in the following commented-out code: it currently throws a ConfigurationError when
#  coulomb_soft_matter_potential is correctly configured
"""for condition_1 in np.atleast_1d(conditions_1):
    for condition_2 in np.atleast_1d(conditions_2):
        if not (condition_1 or condition_2):
            raise ConfigurationError("For each component of range_of_initial_particle_positions, give a list of two float "
                             "values (the lower and upper bounds of the range) for any model or a float when fixing "
                             "each initial particle positions in Bayesian models.")"""
if size_of_particle_space is not None:
    if dimensionality_of_particle_space == 1:
        if type(range_of_initial_particle_positions) == float:
            conditions_3 = abs(range_of_initial_particle_positions) <= size_of_particle_space / 2
        else:
            conditions_3 = (range_of_initial_particle_positions[0] >= - size_of_particle_space / 2 and
                            range_of_initial_particle_positions[1] <= size_of_particle_space / 2)
    elif type(range_of_initial_particle_positions[0]) == float:
        conditions_3 = [abs(range_of_initial_particle_positions[i]) <= size_of_particle_space[i] / 2
                        for i in range(len(size_of_particle_space))]
    else:
        conditions_3 = [range_of_initial_particle_positions[i][0] >= - size_of_particle_space[i] / 2 and
                        range_of_initial_particle_positions[i][1] <= size_of_particle_space[i] / 2
                        for i in range(len(size_of_particle_space))]
    for condition_3 in np.atleast_1d(conditions_3):
        if not condition_3:
            raise ConfigurationError("The absolute value of all floats given within range_of_initial_particle_positions"
                                     " must be less than half the size_of_particle_space.")

if size_of_particle_space is not None:
    size_of_particle_space_over_two = 0.5 * np.atleast_1d(size_of_particle_space)
    size_of_particle_space_over_two.flags.writeable = False
size_of_particle_space = np.atleast_1d(size_of_particle_space)
size_of_particle_space.flags.writeable = False
