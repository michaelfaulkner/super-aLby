from markov_chain_diagnostics import get_iact_and_acf
import importlib
import numpy as np
import os
import sys


this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")


def main(config_file):
    config_file_string = config_file
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directories, temperatures,
     number_of_equilibration_iterations, number_of_observations, number_of_particles,
     _, _, _) = helper_methods.get_basic_config_data(config_file_string)
    sample_directory = sample_directories[0]
    sample = np.load(sample_directory)
    print(sample)


if __name__ == '__main__':
    main(sys.argv[1])
