from base.parsing import parse_options, read_config, get_value
import sys


args = parse_options(sys.argv[1:])
config = read_config(args.config_file)
beta = get_value(config, "ModelSettings", "beta")
number_of_particles = get_value(config, "ModelSettings", "number_of_particles")
dimensionality_of_particle_space = get_value(config, "ModelSettings", "dimensionality_of_particle_space")
size_of_particle_space = get_value(config, "ModelSettings", "size_of_particle_space")
