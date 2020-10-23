from base.parsing import parse_options, read_config, get_value
import sys


args = parse_options(sys.argv[1:])
config = read_config(args.config_file)
beta = get_value(config, "ModelSettings", "beta")
number_of_particles = get_value(config, "ModelSettings", "number_of_particles")
size_of_particle_space = get_value(config, "ModelSettings", "size_of_particle_space")
if size_of_particle_space is None or type(size_of_particle_space) == float:
    dimensionality_of_particle_space = 1
else:
    dimensionality_of_particle_space = len(size_of_particle_space)
