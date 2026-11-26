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
            lengths_02 = output_array[:, 3]
            
            spatial_correlations_02  = spatial_correlations_02 [lengths_02<cutoff]
            spatial_correlations_err_02  = spatial_correlations_err_02 [lengths_02<cutoff]
            lengths_02 = lengths_02[lengths_02<cutoff]

        elif timestep == 0.025:
            print("0.025")
            spatial_correlations_0025 = output_array[:, 0]
            spatial_correlations_err_0025  = output_array[:, 1]
            spatial_correlations_pm_1_0025  = output_array[:, 2]
            lengths_0025 = output_array[:, 3]
            
            spatial_correlations_0025  = spatial_correlations_0025[lengths_0025<cutoff]
            spatial_correlations_err_0025  = spatial_correlations_err_0025[lengths_0025<cutoff]
            lengths_0025 = lengths_0025[lengths_0025<cutoff]

        elif timestep == 0.1:
            print("0.1")
            spatial_correlations_01 = output_array[:, 0]
            spatial_correlations_err_01  = output_array[:, 1]
            spatial_correlations_pm_1_01  = output_array[:, 2]
            lengths_01 = output_array[:, 3]
            
            spatial_correlations_01  = spatial_correlations_01[lengths_01<cutoff]
            spatial_correlations_err_01  = spatial_correlations_err_01[lengths_01<cutoff]
            lengths_01 = lengths_01[lengths_01<cutoff]
        
        elif timestep == 0.05:
            print("0.05")
            spatial_correlations_005 = output_array[:, 0]
            spatial_correlations_err_005  = output_array[:, 1]
            spatial_correlations_pm_1_005  = output_array[:, 2]
            lengths_005 = output_array[:, 3]
            
            spatial_correlations_005  = spatial_correlations_005[lengths_005<cutoff]
            spatial_correlations_err_005  = spatial_correlations_err_005[lengths_005<cutoff]
            lengths_005 = lengths_005[lengths_005<cutoff]


    popt_02, pcov_02 = curve_fit(exp_fit, lengths_02[lengths_02<75], spatial_correlations_02[lengths_02<75])

    fig, ax = plt.subplots(1,1, sharey=True, sharex=True)
    ax.errorbar(lengths_02[spatial_correlations_err_02>0][lengths_02<80], spatial_correlations_02[spatial_correlations_err_02>0][lengths_02<80],
                yerr = spatial_correlations_err_02[spatial_correlations_err_02>0][lengths_02<80], fmt="o", capsize=5, color = "#d54e5dff", label = r"$N_{\tau}=600$", zorder=2)
    ax.plot(lengths_02[lengths_02<80], exp_fit(lengths_02[lengths_02<80], *popt_02), color = "#0e0d0dff", zorder=6)
    print(f"0.2: {popt_02}")


    popt_01, pcov_01 = curve_fit(exp_fit, lengths_01[lengths_01<100], spatial_correlations_01[lengths_01<100])
    ax.errorbar(lengths_01[spatial_correlations_err_01>0][lengths_01<100], spatial_correlations_01[spatial_correlations_err_01>0][lengths_01<100],
                yerr = spatial_correlations_err_01[spatial_correlations_err_01>0][lengths_01<100], fmt="o", capsize=5, color = "#e79d3cff", label = r"$N_{\tau}=1200$", zorder=3)
    ax.plot(lengths_01[lengths_01<100], exp_fit(lengths_01[lengths_01<100], *popt_01), color = "#161617ff", zorder=6)
    print(f"0.1: {popt_01}")


    popt_005, pcov_005 = curve_fit(exp_fit, lengths_005[lengths_005<180], spatial_correlations_005[lengths_005<180])
    ax.errorbar(lengths_005[spatial_correlations_err_005>0][lengths_005<200], spatial_correlations_005[spatial_correlations_err_005>0][lengths_005<200],
                yerr = spatial_correlations_err_005[spatial_correlations_err_005>0][lengths_005<200], fmt="o", capsize=5, color = "#ee5be9ff", label = r"$N_{\tau}=2400$", zorder=4)
    ax.plot(lengths_005[lengths_005<180], exp_fit(lengths_005[lengths_005<180], *popt_005), color = "#161617ff", zorder=6)
    print(f"0.05: {popt_005}")


    popt_0025, pcov_0025 = curve_fit(exp_fit, lengths_0025[lengths_0025>1], spatial_correlations_0025[lengths_0025>1])
    ax.errorbar(lengths_0025[spatial_correlations_err_0025>0], spatial_correlations_0025[spatial_correlations_err_0025>0],
                yerr = spatial_correlations_err_0025[spatial_correlations_err_0025>0], fmt="o", capsize=5, color = "#b662f1ff", label = r"$N_{\tau}=4800$", zorder=5)
    ax.plot(lengths_0025[lengths_0025>0], exp_fit(lengths_0025[lengths_0025>0], *popt_0025), color = "#161617ff", zorder=6)
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