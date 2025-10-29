import numpy as np
import os
import importlib
import matplotlib
import matplotlib.pyplot as plt
import mpmath as mp
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

def main(data_file_path):

    numerical_data = np.zeros(len(os.listdir(data_file_path)))
    analytical_data = np.zeros(len(os.listdir(data_file_path)))
    T_data = np.zeros(len(os.listdir(data_file_path)))

    for index, folder in enumerate(os.listdir(data_file_path)):

        omega = 1.0
        total_time = float(folder.split("_")[1])

        data_path = os.path.join(os.path.join(data_file_path, folder), "temperature_00_checkpoint_00_sample_of_mean_squared_positions.npy")
        temperature_index = 0
        thinning_level = None
        number_of_equilibration_iterations = None
        mean_sample = np.load(data_path)[5000:]
        numerical = np.mean(mean_sample)
        print(f"sample: {numerical}")
        analytical = analytical_x_squared_open_worldlines(total_time, omega)
        print(f"analytical: {analytical}")

        numerical_data[index] = numerical
        analytical_data[index] = analytical
        T_data[index] = total_time
        

    plt.scatter(T_data, analytical_data, label="analytical")
    plt.scatter(T_data, numerical_data, label="numerical")
    plt.xlabel("T")
    plt.ylabel("<x^2>")
    plt.legend()
    plt.savefig("dirichlet.png")





def coth(x):
    if x != 0:
        print(np.cosh(x))
        print(np.sinh(x))

        return np.cosh(x)/np.sinh(x)
    else:
        raise Exception("Input to coth(x) may not be 0")

def analytical_x_squared_open_worldlines(T, omega):
    return (-1 + T * omega *mp.coth(T * omega))/(2 * T * omega**2)

if __name__ == '__main__':
    main(sys.argv[1])