import importlib
import os
import sys


this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")


def main(config_folder, min, max, min_timestep, max_timestep=3.0):
    min = int(min)
    max = int(max)
    min_timestep = float(min_timestep)
    max_timestep = float(max_timestep)

    for iact_index in range(min, max):

        for index, folder in enumerate(os.listdir(config_folder)):
            config_file_string = os.path.join(
                config_folder, folder, f"{iact_index}.ini")
            print(config_file_string)
            config = parsing.read_config(
                parsing.parse_options([config_file_string]).config_file)
            (config_file_mediator, potential, _, samplers, sample_directory, temperature,
             number_of_equilibration_iterations, number_of_observations, number_of_particles,
             size_of_particle_space) = helper_methods.get_basic_config_data(config_file_string)

            
            timestep = parsing.get_value(
                config, strings.to_camel_case(potential), "timestep")
            
            temperature_index = 0
            thinning_level = None
            if timestep >= min_timestep and timestep <= max_timestep:
         
                old_filepath = os.path.join(sample_directory, "run_index.txt")

                new_filepath = os.path.join(sample_directory, "checkpoint_index.txt")

                os.rename(old_filepath, new_filepath)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
