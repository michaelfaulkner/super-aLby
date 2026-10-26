import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import sample_getter
import sys
from markov_chain_diagnostics import get_cumulative_distribution
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")


def main(metrop_data, ecmc_data):

    # /home/raichkel/super-aLby-private/src/output/iact_data/metropolis/001/0/checkpoint_00_sample_of_mean_squared_positions.npy
    # /home/raichkel/super-aLby-private/src/output/iact_data/ecmc/001/0/checkpoint_00_sample_of_mean_squared_positions.npy
    metrop_mean_squared_sample = np.load(metrop_data)

    ecmc_mean_squared_sample = np.load(ecmc_data)

    fig, ax = plt.subplots(1,1, figsize=(7,5))
    sample_index_cutoff = 10000
    ax.plot(np.arange(sample_index_cutoff), metrop_mean_squared_sample[:sample_index_cutoff], color = "#e16f04ff", label ="Metropolis MC")

    ax.plot(np.arange(sample_index_cutoff), ecmc_mean_squared_sample[:sample_index_cutoff], color = "#e20acdff", label = "ECMC")
    ax.tick_params(direction="in", left="off",labelleft="off", axis='both', which='both', labelsize=18)
    ax.set_xlabel("Sample index", fontsize = 25)
    ax.set_ylabel(r"$|\mathbf{x}|^2$", fontsize = 30)
    legend = ax.legend(fontsize=18)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)
    plt.setp(ax.spines.values(), lw=2.0, color='black')
    
    plt.tight_layout()
    plt.savefig("trace.pdf")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

