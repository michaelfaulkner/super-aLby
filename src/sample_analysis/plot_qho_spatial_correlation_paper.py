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
matplotlib.use('Agg')


def main(output_directory, propertime, cutoff):
    propertime = float(propertime)
    cutoff = float(cutoff)
    timesteps = [0.2, 0.025]
    #timesteps = [ 0.015, 0.01]
    timestep_strs = ["02", "0025"]
    #timestep_strs = [ "0015", "001"]



    for t_index, timestep in enumerate(timesteps):

        timestep_str = timestep_strs[t_index]

        output_array = np.load(f"{output_directory}/correlation_func_data_{timestep_str}.npy")

        if timestep == 0.2:
            print("0.2")
            spatial_correlations_01 = output_array[:, 0]
            spatial_correlations_err_01  = output_array[:, 1]
            spatial_correlations_pm_1_01  = output_array[:, 2]
            lengths = output_array[:, 3]
            
            spatial_correlations_01  = spatial_correlations_01 [lengths<cutoff]
            spatial_correlations_err_01  = spatial_correlations_err_01 [lengths<cutoff]
    

        elif timestep == 0.025:
            print("0.025")
            spatial_correlations_0025 = output_array[:, 0]
            spatial_correlations_err_0025  = output_array[:, 1]
            spatial_correlations_pm_1_0025  = output_array[:, 2]
            
            spatial_correlations_0025  = spatial_correlations_0025[lengths<cutoff]
            spatial_correlations_err_0025  = spatial_correlations_err_0025[lengths<cutoff]
            lengths = lengths[lengths<cutoff]


    fig, ax = plt.subplots(1,2, figsize =(15,5), sharey=True, sharex=True)
    ax[0].errorbar(lengths[spatial_correlations_err_01>0], spatial_correlations_01[spatial_correlations_err_01>0],
                yerr = spatial_correlations_err_01[spatial_correlations_err_01>0], fmt="o", capsize=5, color = "#ae2132ff", label = r"$N_{\tau}=600$")

    ax[1].errorbar(lengths[spatial_correlations_err_0025>0], spatial_correlations_0025[spatial_correlations_err_0025>0],
                yerr = spatial_correlations_err_0025[spatial_correlations_err_0025>0], fmt="o", capsize=5, color = "#7508c3ff", label = r"$N_{\tau}=4800$")

    
    ax[0].set_xlabel(r"$r$", fontsize = 30, weight ="bold")
    ax[0].set_ylabel(r"$G(r)$", fontsize = 30, weight ="bold")
    ax[0].set_yscale("log")
    ax[0].set_xscale("linear")

    ax[1].set_xlabel(r"$r$", fontsize = 30, weight ="bold")
    ax[1].set_ylabel(r"$G(r)$", fontsize = 30, weight ="bold")
    ax[1].set_yscale("log")
    ax[1].set_xscale("linear")

    legend_properties = {'weight':'bold', 'size':20}
    legend = ax[0].legend(prop=legend_properties)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)

    legend = ax[1].legend(prop=legend_properties)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)

    
    ax[0].tick_params(direction="in", left="off",labelleft="off", axis='both', which='both', labelsize=17)
    ax[1].tick_params(direction="in", left="off",labelleft="off", axis='both', which='both', labelsize=17)

    for tick in ax[0].get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax[0].get_yticklabels():
        tick.set_fontweight('bold')

    for tick in ax[1].get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax[1].get_yticklabels():
        tick.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(f"correlation_func_paper.pdf")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])