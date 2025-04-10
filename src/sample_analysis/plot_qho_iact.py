import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def main(iact_data_path_0, iact_data_path_1, iact_data_path_2):

    iact_timestep_0 = np.load(iact_data_path_0)
    iact_timestep_1 = np.load(iact_data_path_1)
    iact_timestep_2 = np.load(iact_data_path_2)


    iact_data_0 = iact_timestep_0[:, 0] 
    timestep_data_0 = iact_timestep_0[:, 1]
    iact_data_1 = iact_timestep_1[:, 0] 
    timestep_data_1 = iact_timestep_1[:, 1]
    iact_data_2 = iact_timestep_2[:, 0] 
    timestep_data_2 = iact_timestep_2[:, 1]
    
    iact_mean_arr = np.zeros(len(timestep_data_0))
    err = np.zeros(len(timestep_data_0))


    for index, timestep in enumerate(timestep_data_0):
        data_0 = iact_data_0[index]
        data_1 = iact_data_1[np.nonzero(timestep_data_1 == timestep)][0]
        data_2 = iact_data_2[np.nonzero(timestep_data_2 == timestep)][0]

        iact_mean_arr[index] = np.mean([data_0, data_1, data_2])
        err[index] = np.std([data_0, data_1, data_2])


    fig, ax = plt.subplots(1, 1)
    #ax.scatter(timestep_data_0, iact_mean_arr)
    ax.errorbar(timestep_data_0, iact_mean_arr, err, fmt='o', capsize=3)
    ax.set_xlabel("delta tau")
    ax.set_ylabel("iact")
    ax.set_xscale("log")
    ax.set_yscale("log")
   
    # # plt.title(f"Integrated Autocorrelation Time for ECMC, lambda = {distance_between_measurements}")
    # # plt.savefig(f"iact_ecmc_{distance_between_measurements}.png")

    plt.title(f"Integrated Autocorrelation Time for Metropolis MC")
    plt.savefig("iact_metropolis.png")
    plt.clf()



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])