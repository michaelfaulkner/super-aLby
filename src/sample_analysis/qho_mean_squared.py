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

def main(metrop_data_path, ecmc_data_path):

    metrop = np.load(metrop_data_path)[:10000]
    ecmc = np.load(ecmc_data_path)[:10000]

    fig, ax = plt.subplots(2,1, sharey =True, sharex =True)
    ax[0].plot(np.arange(np.shape(metrop)[0]), metrop, color="#db9437", label= "Metropolis MC")
    ax[1].plot(np.arange(np.shape(ecmc)[0]), ecmc, color="#e31bb4", label = "ECMC")

    ax[1].set_xlabel("Sample index", fontsize=15)
    ax[0].set_ylabel(r"$<x^2>$", fontsize=20)
    ax[1].set_ylabel(r"$<x^2>$", fontsize=20)

    ax[0].legend(loc ="upper left")
    ax[1].legend(loc ="upper left")

    plt.savefig("trace_paper.pdf")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])