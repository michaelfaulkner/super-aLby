import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'

def main(iact_data_path, N):

    N = int(N)
    timestep_data = np.load(os.path.join(iact_data_path, "iact_ecmc_0.npy"))[:, 1]
    storage_arr = np.zeros((len(timestep_data), N))
    sorted_timestep = timestep_data[np.argsort(timestep_data)]

    for index in range(N):
        iact_timestep =  np.load(os.path.join(iact_data_path, f"iact_ecmc_{index}.npy"))
        iact_data = iact_timestep[:, 0] 
        timestep_data = iact_timestep[:, 1] 
        argsorted_data = np.argsort(timestep_data)
        timestep_argsorted = timestep_data[argsorted_data]
        iact_data = iact_data[argsorted_data] 
        storage_arr[:, index] = iact_data



    iact_mean_arr = np.mean(storage_arr, axis = 1)
    err = np.std(storage_arr, axis=1)

    fig, ax = plt.subplots(1, 1)
    #ax.scatter(timestep_data_0, iact_mean_arr)
    ax.errorbar(sorted_timestep, iact_mean_arr, err, fmt='o', capsize=3, markersize=3, color="purple")
    ax.set_xlabel(r"$\delta \tau$", fontsize=20, labelpad=-10)
    ax.set_ylabel("IACT", fontsize=15, labelpad=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    #ax.set_ylim(1e3, 1e7)
   


    plt.title(f"Integrated Autocorrelation Time for ECMC, with {N} Repeats")
    plt.tight_layout()
    plt.savefig("iact_ecmc.png")
    plt.clf()



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])