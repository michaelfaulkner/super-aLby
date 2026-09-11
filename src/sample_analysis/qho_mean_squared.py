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

def main(data_path):

    sample_directory = data_path
    temperature_index = 0
    thinning_level = None
    number_of_equilibration_iterations = None
    mean_sample = np.load(data_path)

    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(np.shape(mean_sample)[0]), mean_sample, color="purple")

    plt.savefig("trace_anharmonic_ecmc_w2_50.png")


if __name__ == '__main__':
    main(sys.argv[1])