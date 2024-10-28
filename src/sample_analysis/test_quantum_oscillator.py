import numpy as np
import os

# read in the k values from k_values.txt
current_directory = os.path.dirname(__file__)
k_values_filepath = os.path.join(os.path.split(current_directory)[0], "k_values.txt")
f = open(k_values_filepath, "r")
print(f.read()) 
# for line in file, remove index, copy value to array
f.close() 
# calculate <x^2> from MC users guide paper for each k
def get_analytical_x2(k, m, timestep, number_of_time_elements):
    dimensionless_omega = np.sqrt(k/m) * timestep
    dimensionless_m = m * timestep
    dimensionless_omega_squared = dimensionless_omega**2
    
    auxillary = 1 + dimensionless_omega_squared / 2 - dimensionless_omega * np.sqrt(1 + dimensionless_omega_squared / 4)

    return (1 / (2 * dimensionless_m * dimensionless_omega * np.sqrt(1 + 0.25 * dimensionless_omega_squared)) *
            (1 + auxillary**number_of_time_elements) / (1 - auxillary**number_of_time_elements))

# read in the x^2 values from the corresponding files
# which are formatted ../metropolis_{k_read_in}/temperature_00_sample_of_mean_positions.npy
# take the mean of these for each simulation?

# plot the simulation values against the calculated values