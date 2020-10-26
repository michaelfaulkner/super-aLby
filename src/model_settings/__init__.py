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
if size_of_particle_space is None or type(size_of_particle_space) == float:
    dimensionality_of_particle_space = 1
else:
    dimensionality_of_particle_space = len(size_of_particle_space)
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
for condition_1 in np.atleast_1d(conditions_1):
    for condition_2 in np.atleast_1d(conditions_2):
        if not (condition_1 or condition_2):
            raise ValueError("For each component of range_of_initial_particle_positions, give a list of two float "
                             "values (the lower and upper bounds of the range) for any model or a float when fixing "
                             "each initial particle position in Bayesian models.")
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
            raise ValueError("The absolute value of all floats given within range_of_initial_particle_positions must be"
                             " less than half the size_of_particle_space.")
