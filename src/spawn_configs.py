import os
import sys
import copy
import configparser
import fnmatch


def spawn_identical_configs(config_file_location, number_of_jobs, config):
    """
    Generates n = number_of_jobs identical configuration files.
    """
    config_file_directory = os.path.splitext(config_file_location)[0]
    config_file_basename = os.path.basename(config_file_directory)
    os.makedirs(config_file_directory, exist_ok=True)

    samplers = [section for section in config.sections() if fnmatch.fnmatch(section, "*Sampler")]
    for job_index in range(number_of_jobs):
        job_config = copy.deepcopy(config)
        job_config_file_path = os.path.join(config_file_directory, f"job_{job_index:02d}.ini")
        for sampler_name in samplers:
            output_directory = config.get(sampler_name, "output_directory")
            job_output_directory = os.path.join(output_directory, config_file_basename, f"job_{job_index:02d}")
            job_config.set(sampler_name, "output_directory", job_output_directory)
        with open(job_config_file_path, 'w') as f:
            job_config.write(f)

    print(f"Created {number_of_jobs} config file(s) in {config_file_directory}.")


def main(config_file_location, number_of_jobs, sweep_start, sweep_end, number_of_increments, config_header,
         config_variable, increment_type='linear'):
    """
    Generates number_of_increments x number_of_jobs configuration files.  Each increment features a different value
    of config_variable.  The value of each increment is determined by sweep_start, sweep_end, and number_of_increments.
    """
    if not os.path.exists(config_file_location):
        raise ValueError(f"Template config file {config_file_location} does not exist")

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_file_location)

    config_file_directory = os.path.splitext(config_file_location)[0]
    os.makedirs(config_file_directory, exist_ok=True)

    sweep_increment = (sweep_end - sweep_start) / number_of_increments
    sweep_values = ([sweep_start * (sweep_end / sweep_start) ** (i / number_of_increments) for i in
                     range(number_of_increments + 1)] if increment_type == 'log'
                    else [sweep_start + sweep_increment * i for i in range(number_of_increments + 1)])

    for index, increment in enumerate(sweep_values):
        increment_config = copy.deepcopy(config)
        increment_config.set(config_header, config_variable, str(increment))
        increment_config_file_path = os.path.join(config_file_directory, f"{config_variable}_{index:02d}.ini")
        with open(increment_config_file_path, 'w') as f:
            increment_config.write(f)
        spawn_identical_configs(increment_config_file_path, number_of_jobs, increment_config)
        if os.path.exists(increment_config_file_path):
            os.remove(increment_config_file_path)

    print(f"Created {number_of_increments}x{number_of_jobs} config file(s) in {config_file_directory}.")


if __name__ == "__main__":
    if len(sys.argv) < 8:
        raise ValueError("Not enough arguments provided to spawn_configs.py.  Correct usage: python spawn_configs.py "
                         "<template_ini> <num_jobs> <start> <end> <num_increments> <config_header> <config_variable> "
                         "[<increment_type>]")
    elif len(sys.argv) == 8:
        main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6],
             sys.argv[7])
    else:
        main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6],
             sys.argv[7], sys.argv[8])
