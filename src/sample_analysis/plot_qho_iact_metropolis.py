import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def main(iact_data_path, N):

    N = int(N)
    timestep_data = np.load(os.path.join(iact_data_path, "iact_metropolis_0.npy"))[:, 1]
    storage_arr = np.zeros((len(timestep_data), N))
    sorted_timestep = timestep_data[np.argsort(timestep_data)]

    for index in range(N):
        iact_timestep =  np.load(os.path.join(iact_data_path, f"iact_metropolis_{index}.npy"))
        iact_data = iact_timestep[:, 0] 
        timestep_data = iact_timestep[:, 1] 
        argsorted_data = np.argsort(timestep_data)
        timestep_argsorted = timestep_data[argsorted_data]
        iact_data = iact_data[argsorted_data] 
        storage_arr[:, index] = iact_data


    iact_mean_arr = np.mean(storage_arr[np.nonzero(sorted_timestep >= 0.01)], axis = 1)
    err = np.std(storage_arr[np.nonzero(sorted_timestep >= 0.01)], axis=1)
    
    fig, ax = plt.subplots(1, 1)
    #ax.scatter(timestep_data_0, iact_mean_arr)
    ax.errorbar(sorted_timestep[np.nonzero(sorted_timestep >= 0.01)], iact_mean_arr, err, fmt='o', capsize=3, markersize=3.5, color="orange")
    ax.set_xlabel(r"$\delta \tau$", fontsize=16, labelpad=-10)
    ax.set_ylabel("IACT", fontsize=15, labelpad=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    #ax.set_ylim(1e3, 1e7)
   


    plt.title(f"Integrated Autocorrelation Time for Metropolis MC, with {N} Repeats")
    plt.savefig("anharmonic_iact_metropolis.png")
    plt.clf()



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])