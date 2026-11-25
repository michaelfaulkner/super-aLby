import numpy as np
import os
import importlib
import matplotlib
import matplotlib.pyplot as plt
import os
import sample_getter
import sys
from scipy.optimize import curve_fit

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')

def exp_fit(r, a, E):
    return np.exp(-(r-a) / E)

def algebraic_fit(r, a, b, n):
    return (a / (r-b))**n

def main(output_directory, propertime, cutoff):
    propertime = float(propertime)
    cutoff = float(cutoff)
    timesteps = [0.2, 0.025, 0.1, 0.05]
    #timesteps = [ 0.025, 0.02]
    timestep_strs = ["02", "0025","01", "005"]
    #timestep_strs = [ "0025", "001"]



    for t_index, timestep in enumerate(timesteps):

        timestep_str = timestep_strs[t_index]

        output_array = np.load(f"{output_directory}/correlation_func_data_{timestep_str}.npy")

        if timestep == 0.2:
            print("0.2")
            spatial_correlations_02 = output_array[:, 0]
            spatial_correlations_err_02  = output_array[:, 1]
            spatial_correlations_pm_1_02  = output_array[:, 2]
            lengths = output_array[:, 3]
            
            spatial_correlations_02  = spatial_correlations_02 [lengths<cutoff]
            spatial_correlations_err_02  = spatial_correlations_err_02 [lengths<cutoff]
    

        elif timestep == 0.025:
            print("0.025")
            spatial_correlations_0025 = output_array[:, 0]
            spatial_correlations_err_0025  = output_array[:, 1]
            spatial_correlations_pm_1_0025  = output_array[:, 2]
            lengths = output_array[:, 3]
            
            spatial_correlations_0025  = spatial_correlations_0025[lengths<cutoff]
            spatial_correlations_err_0025  = spatial_correlations_err_0025[lengths<cutoff]
            lengths = lengths[lengths<cutoff]

        elif timestep == 0.1:
            print("0.1")
            spatial_correlations_01 = output_array[:, 0]
            spatial_correlations_err_01  = output_array[:, 1]
            spatial_correlations_pm_1_01  = output_array[:, 2]
            lengths = output_array[:, 3]
            
            spatial_correlations_01  = spatial_correlations_01[lengths<cutoff]
            spatial_correlations_err_01  = spatial_correlations_err_01[lengths<cutoff]
            lengths = lengths[lengths<cutoff]
        
        elif timestep == 0.05:
            print("0.05")
            spatial_correlations_005 = output_array[:, 0]
            spatial_correlations_err_005  = output_array[:, 1]
            spatial_correlations_pm_1_005  = output_array[:, 2]
            lengths = output_array[:, 3]
            
            spatial_correlations_005  = spatial_correlations_005[lengths<cutoff]
            spatial_correlations_err_005  = spatial_correlations_err_005[lengths<cutoff]
            lengths = lengths[lengths<cutoff]


    popt_02, pcov_02 = curve_fit(exp_fit, lengths[lengths<75], spatial_correlations_02[lengths<75])

    fig, ax = plt.subplots(1,1, sharey=True, sharex=True)
    ax.errorbar(lengths[spatial_correlations_err_02>0], spatial_correlations_02[spatial_correlations_err_02>0],
                yerr = spatial_correlations_err_02[spatial_correlations_err_02>0], fmt="o", capsize=5, color = "#d54e5dff", label = r"$N_{\tau}=600$", zorder=2)
    ax.plot(lengths[lengths<75], exp_fit(lengths[lengths<75], *popt_02), color = "#0e0d0dff", zorder=6)
    print(f"0.2: {popt_02}")


    popt_01, pcov_01 = curve_fit(exp_fit, lengths[lengths>1], spatial_correlations_01[lengths>1])
    ax.errorbar(lengths[spatial_correlations_err_01>0], spatial_correlations_01[spatial_correlations_err_01>0],
                yerr = spatial_correlations_err_01[spatial_correlations_err_01>0], fmt="o", capsize=5, color = "#e79d3cff", label = r"$N_{\tau}=1200$", zorder=3)
    ax.plot(lengths[spatial_correlations_err_01>0], exp_fit(lengths[spatial_correlations_err_01>0], *popt_01), color = "#161617ff", zorder=6)
    print(f"0.1: {popt_01}")


    popt_005, pcov_005 = curve_fit(exp_fit, lengths[lengths>1], spatial_correlations_005[lengths>1])
    ax.errorbar(lengths[spatial_correlations_err_005>0], spatial_correlations_005[spatial_correlations_err_005>0],
                yerr = spatial_correlations_err_005[spatial_correlations_err_005>0], fmt="o", capsize=5, color = "#ee5be9ff", label = r"$N_{\tau}=2400$", zorder=4)
    ax.plot(lengths[spatial_correlations_err_005>0], exp_fit(lengths[spatial_correlations_err_005>0], *popt_005), color = "#161617ff", zorder=6)
    print(f"0.05: {popt_005}")


    popt_0025, pcov_0025 = curve_fit(exp_fit, lengths[lengths>1], spatial_correlations_0025[lengths>1])
    ax.errorbar(lengths[spatial_correlations_err_0025>0], spatial_correlations_0025[spatial_correlations_err_0025>0],
                yerr = spatial_correlations_err_0025[spatial_correlations_err_0025>0], fmt="o", capsize=5, color = "#b662f1ff", label = r"$N_{\tau}=4800$", zorder=5)
    ax.plot(lengths[spatial_correlations_err_0025>0], exp_fit(lengths[spatial_correlations_err_0025>0], *popt_0025), color = "#161617ff", zorder=6)
    print(f"0.025: {popt_0025}")


    ax.set_xlabel(r"$r$", fontsize = 30, weight ="bold")
    ax.set_ylabel(r"$G(r)$", fontsize = 30, weight ="bold")
    ax.set_yscale("log")
    ax.set_xscale("linear")




    legend_properties = {'size':20}
    legend = ax.legend(prop=legend_properties)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)

    ax.set_ylim(bottom = 10e-5)

    
    ax.tick_params(direction="in", left="off",labelleft="off", axis='both', which='both', labelsize=17)



    plt.tight_layout()
    plt.savefig(f"correlation_func_paper.pdf")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])