import numpy as np
import os
import importlib
import matplotlib
import matplotlib.pyplot as plt
import sample_getter
import sys
from configparser import NoOptionError
from markov_chain_diagnostics import get_sample_mean_and_error

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")

# read in the k values from k_values.txt
current_directory = os.path.dirname(__file__)
k_values_filepath = os.path.join(os.path.split(current_directory)[0], "m_values.txt")
k_data = np.loadtxt(k_values_filepath, dtype='str')
k_values = k_data[:,1]


def get_analytical_x2(dimensionless_m, timestep, number_of_time_elements):
    """
    calculate <x^2> from MC users guide paper for given k
    """
    dimensionless_omega = dimensionless_m
    dimensionless_omega_squared = dimensionless_omega**2
    
    auxillary = 1 + dimensionless_omega_squared / 2 - dimensionless_omega * np.sqrt(1 + dimensionless_omega_squared / 4)

    return (1 / (2 * dimensionless_m * dimensionless_omega * np.sqrt(1 + 0.25 * dimensionless_omega_squared)) *
            (1 + auxillary**number_of_time_elements) / (1 - auxillary**number_of_time_elements))


analytical_x2 = np.zeros(len(k_values))
numerical_x2 = np.zeros(len(k_values))
numerical_x2_dimensionless = np.zeros(len(k_values))
x2_err = np.zeros(len(k_values))
k_arr = np.zeros(len(k_values))
m_arr = np.zeros(len(k_values))

for index, string in enumerate(k_values):
    config_file_string = f"src/config_files/convergence_tests/one_dim_quantum_oscillator_potential/metropolis_{string}.ini"
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
     _, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)
    
    dimensionless_mass = parsing.get_value(config, strings.to_camel_case(potential), "dimensionless_mass")
    timestep = parsing.get_value(config, strings.to_camel_case(potential), "timestep")
    number_of_particles = parsing.get_value(config, "ModelSettings", "number_of_particles") # this is janky but model_settings.number_of_particles doesn't like that I need to access several config files
    #print(mass, timestep, k)
    analytical_x2[index] = get_analytical_x2(dimensionless_mass, timestep, number_of_particles)
    
    # read in the x^2 values from the corresponding files
    # which are formatted ../metropolis_{k_read_in}/temperature_00_sample_of_mean_positions.npy
    # take the mean of these for each simulation?
    sample_directory = f"output/convergence_tests/one_dim_quantum_oscillator_potential/metropolis_{string}"
    temperature_index = 0
    try:
        x2_mean_and_error = get_sample_mean_and_error(sample_getter.get_mean_positions(sample_directory, temperatures[temperature_index],
                        temperature_index, number_of_particles, number_of_equilibration_iterations, thinning_level=None))
    except FileNotFoundError:
        print("data not produced")
   
    numerical_x2_dimensionless[index] = x2_mean_and_error[0]
    numerical_x2[index] = x2_mean_and_error[0] * timestep**2
    x2_err[index] = x2_mean_and_error[1]
    k_arr[index] = dimensionless_mass
    m_arr[index] = dimensionless_mass / timestep

np.save("src/m_arr", m_arr)
np.save("src/numerical_x2", numerical_x2)
fig, ax = plt.subplots(1,1)
ax.scatter(k_arr, analytical_x2, label="Analytical result", marker="x", color="red")
ax.errorbar(k_arr, numerical_x2_dimensionless, x2_err, label="Numerical result", color="blue", marker="x", linestyle="")
ax.set_xlabel("dimensionless m")
ax.set_ylabel("<x^2> - with dimensionless positions")
ax.legend()

# fig1, ax1 = plt.subplots(1,1)
# ax1.scatter(m_arr, numerical_x2 * timestep**2, marker="x", color="red")
# ax1.set_xlabel("m")
# ax1.set_ylabel("<x^2>")

# ax[1].scatter(k_arr, analytical_x2, label="Analytical result", marker="x", color="red")
# ax[1].errorbar(k_arr, numerical_x2, x2_err, label="Numerical result * scaling factor", color="blue", marker="x", linestyle="")
# ax[1].set_xlabel("dimentionless m")
# ax[1].set_ylabel("<x^2>")
# ax[1].legend()
plt.tight_layout()
plt.savefig("output/figs/analytical_numerical_x2.png")
plt.show()








# plot the simulation values against the calculated values