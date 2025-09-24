import os
import sys
import copy
import configparser
import numpy as np
import fnmatch


def spawn_identical_configs(config_file_location, num_jobs):
    """
    Generates num_jobs identical configuration files.

    Parameters
    ----------
    config_file_location : str
        A string defining the location of the configuration file.
    num_jobs : int
        Number of identical configuration files to generate.
    """
    if not os.path.exists(config_file_location):
        raise ValueError(f"Template config file {config_file_location} does not exist")

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_file_location)

    config_file_directory = os.path.splitext(config_file_location)[0]
    config_file_basename = os.path.basename(config_file_directory)
    os.makedirs(config_file_directory, exist_ok=True)

    samplers = [section for section in config.sections() if fnmatch.fnmatch(section, "*Sampler")]
    for job_index in range(num_jobs):
        job_config = copy.deepcopy(config)
        job_config_file_path = os.path.join(config_file_directory, f"job_{job_index:02d}.ini")
        for sampler_name in samplers:
            output_directory = config.get(sampler_name, "output_directory")
            job_output_directory = os.path.join(output_directory, config_file_basename, f"job_{job_index:02d}")
            job_config.set(sampler_name, "output_directory", job_output_directory)
        with open(job_config_file_path, 'w') as f:
            job_config.write(f)

    print(f"Created {num_jobs} config file(s) in {config_file_directory}.")


def main(config_file_location, num_jobs, sweep_start, sweep_end, num_increments, config_header, config_variable,
         increment_type='linear'):
    if not os.path.exists(config_file_location):
        raise ValueError(f"Template config file {config_file_location} does not exist")

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_file_location)

    config_file_directory = os.path.splitext(config_file_location)[0]
    config_file_basename = os.path.basename(config_file_location)
    os.makedirs(config_file_directory, exist_ok=True)

    sweep_increment = (sweep_end - sweep_start) / num_increments
    sweep_values = ([sweep_start * (sweep_end / sweep_start) ** (i / num_increments) for i in range(num_increments + 1)]
                    if increment_type == 'log' else [sweep_start + sweep_increment * i for
                                                     i in range(num_increments + 1)])

    for index, increment in enumerate(sweep_values):
        increment_config = copy.deepcopy(config)
        increment_config.set(config_header, config_variable, str(increment))
        increment_config_file_path = os.path.join(config_file_directory,
                                                  f'{config_file_basename.split(".")[0]}_{index:02d}.ini')
        with open(increment_config_file_path, 'w') as f:
            increment_config.write(f)
        spawn_identical_configs(increment_config_file_path, num_jobs)
        if os.path.exists(increment_config_file_path):
            os.remove(increment_config_file_path)

    print(f"Created {num_increments}x{num_jobs} config file(s) in {config_file_directory}.")


if __name__ == "__main__":
    if len(sys.argv) < 8:
        raise ValueError("Not enough arguments provided to spawn_configs.py. "
                         "Usage: python spawn_configs.py <template_ini> <num_jobs> <start> <end> "
                         "<num_increments> <config_header> <config_variable> [<increment_type>]")
    config_file_loc = sys.argv[1]
    n_jobs = int(sys.argv[2])
    start = float(sys.argv[3])
    end = float(sys.argv[4])
    n_increments = int(sys.argv[5])
    header = sys.argv[6]
    variable = sys.argv[7]
    if len(sys.argv) == 8:
        main(config_file_loc, n_jobs, start, end, n_increments, header, variable)
    else:
        type_increment = sys.argv[8]
        main(config_file_loc, n_jobs, start, end, n_increments, header, variable, type_increment)
