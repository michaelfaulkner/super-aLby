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


def main(output_directory, propertime):
    propertime = float(propertime)
    
    timesteps = [3.0, 2.0, 1.0, 0.95, 0.9, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1, 0.075, 0.05, 0.025, 0.015]
    timestep_strs = ["3", "2", "1", "095", "09", "075", "05", "04", "03", "02", "01", "0075", "005", "0025", "0015"]


    for t_index, timestep in enumerate(timesteps):

        timestep_str = timestep_strs[t_index]

        output_array = np.load(f"{output_directory}/correlation_func_data_{timestep_str}.npy")

        spatial_correlations = output_array[:, 0]
        spatial_correlations_err = output_array[:, 1]
        spatial_correlations_pm_1 = output_array[:, 2]
        lengths = output_array[:, 3]


        fig, ax = plt.subplots(1,1)
        ax.errorbar(lengths[spatial_correlations_err>0], spatial_correlations[spatial_correlations_err>0],
                    yerr = spatial_correlations_err[spatial_correlations_err>0], fmt="o", capsize=5, color = "#e8799c")


        ax.set_xlabel(r"r")
        ax.set_ylabel("G(r)")
        ax.set_yscale("log")
        ax.set_xscale("linear")
        ax.set_title(r"$N_{\tau} = $" + f"{120/timestep:.0f}")
        #plt.legend()
        

        plt.savefig(f"correlation_func_{timestep_str}.pdf")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])