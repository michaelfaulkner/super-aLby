import numpy as np
import os
import importlib
import matplotlib
import matplotlib.pyplot as plt
import os
import sample_getter
import sys


this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")
matplotlib.rcParams['mathtext.fontset'] = 'cm'

def spatial_correlation_function(positions, length, number_of_particles):
    """
    returns spatial correlation function for the positions sample. 
    C(r) = <x_{0} x_{r}> - <x_{0}><x_{r}>
    
    Parameters
    -----------
    positons : np array
        The array containing the position data sample
    length: float
        The distance between the first and second indices of the correlation function.
    """
    corr_func = np.zeros(number_of_particles)
    for particle_index in range(number_of_particles):
        corr_func[particle_index] = np.mean(positions[:, particle_index] * positions[:, (particle_index + length)%number_of_particles]) - \
            np.mean(positions[:, particle_index]) * np.mean(positions[:, (particle_index + length)%number_of_particles])

    return np.mean(corr_func)


def main(config_folder, min_length, max_length, N_repeats, output_directory):

    min_length = int(min_length)
    max_length = int(max_length)
    N_repeats = int(N_repeats)

    timesteps = [3.0, 2.0, 1.0, 0.95, 0.9, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1, 0.075, 0.05, 0.025, 0.015]
    timestep_strs = ["3", "2", "1", "095", "09", "075", "05", "04", "03", "02", "01", "0075", "005", "0025", "0015"]

    correlation_length = np.zeros(len(timesteps))

    for t_index, timestep in enumerate(timesteps):
        timestep_str = str(timestep)
        timestep_str = timestep_strs[t_index]
        #####
        # iteration for a single timstep

        lengths = np.arange(min_length, max_length, step = 5)
        spatial_correlations = np.zeros((len(lengths), N_repeats))
        spatial_correlations_pm_1 = np.zeros((len(lengths), N_repeats))


        for n in range(N_repeats):
            config_file_string = os.path.join(
                            config_folder, timestep_str, f"{n}.ini")
    
            config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
            (config_file_mediator, potential, _, samplers, sample_directory, temperature, number_of_equilibration_iterations,
            _, number_of_particles, size_of_particle_space) = helper_methods.get_basic_config_data(config_file_string)
            
    
            thinning_level = None
            timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
            print(n)
            position_sample = sample_getter.get_positions(sample_directory, temperature, 0, number_of_particles,
                                                    number_of_equilibration_iterations, thinning_level=thinning_level)
            for index, length in enumerate(lengths):
                spatial_correlations[index, n]= spatial_correlation_function(position_sample, length, number_of_particles)
                if 2.0 <= length <= 39:
                    spatial_correlations_pm_1[index, n]= spatial_correlation_function(position_sample, length-1, number_of_particles) / \
                                                    spatial_correlation_function(position_sample, length+1, number_of_particles)
            
            spatial_correlations_pm_1 = 0.5 * np.log(spatial_correlations_pm_1)

        spatial_correlations_err = np.std(spatial_correlations, axis = 1)
        spatial_correlations = np.mean(spatial_correlations, axis = 1)

        spatial_correlations_pm_1_err = np.std(spatial_correlations_pm_1, axis = 1)
        spatial_correlations_pm_1 = np.mean(spatial_correlations_pm_1, axis = 1)

        correlation_length[t_index] = timestep / np.mean(spatial_correlations_pm_1) 


        output_array = np.zeros((len(spatial_correlations), 4))
        output_array[:, 0] = spatial_correlations
        output_array[:, 1] = spatial_correlations_err
        output_array[:, 2] = spatial_correlations_pm_1
        output_array[:, 3] = lengths

        np.save(f"{output_directory}/correlation_func_data_{timestep_str}.npy", output_array)
    





if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])