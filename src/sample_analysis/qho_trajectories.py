from configparser import NoOptionError
import importlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sample_getter
import sys
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')

# Add the directory that contains the module plotting_functions to sys.path
this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")
matplotlib.use('Agg')

def main(metropolis_positions_data, ecmc_positions_data):


    metrop_positions = np.load(metropolis_positions_data)
    ecmc_positions = np.load(ecmc_positions_data)

    sample_index_0 = 10000
    sample_index_1 = 10100

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

    ax[0].plot(np.arange(len(metrop_positions[sample_index_0,:])), metrop_positions[sample_index_0,:], color="#e91f4e")
    ax[0].plot(np.arange(len(metrop_positions[sample_index_1,:])), metrop_positions[sample_index_1,:], color="#320968", linestyle="--")
    ax[1].plot(np.arange(len(ecmc_positions[sample_index_0,:])), ecmc_positions[sample_index_0,:], color="#e91f4e")
    ax[1].plot(np.arange(len(ecmc_positions[sample_index_1,:])), ecmc_positions[sample_index_1,:], color="#320968", linestyle="--")

    ax[0].tick_params(direction="in", left="off",labelleft="off", axis='both', which='both', labelsize=10)
    ax[1].tick_params(direction="in", left="off",labelleft="off", axis='both', which='both', labelsize=10)


    for tick in ax[0].get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax[0].get_yticklabels():
        tick.set_fontweight('bold')
    for tick in ax[1].get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax[1].get_yticklabels():
        tick.set_fontweight('bold')

    ax[0].text(400, -1.2, "Metropolis MC", color='black', weight = "bold",
        bbox=dict(facecolor='none', edgecolor='grey'))
    ax[1].text(465, -1.2, "ECMC", color='black',  weight = "bold", 
        bbox=dict(facecolor='none', edgecolor='grey'))

    ax[0].set_ylabel(r"$x_j$",  fontsize = 30, weight = "bold")
    ax[1].set_ylabel(r"$x_j$",  fontsize = 30, weight = "bold")
    ax[1].set_xlabel(r"$j$",  fontsize = 25, weight = "bold")

    plt.tight_layout()

    plt.savefig("trajectories.pdf")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])