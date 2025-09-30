import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
matplotlib.rcParams['mathtext.fontset'] = 'cm'

def main(iact_data_path, N):

    N = int(N)
    timestep_data = np.load(os.path.join(iact_data_path, "iact_ecmc_0.npy"))[:, 1]
    sorted_timestep = timestep_data[np.argsort(timestep_data)]
    storage_arr = np.zeros((len(timestep_data), N))

    for index in range(N):
        iact_timestep =  np.load(os.path.join(iact_data_path, f"iact_ecmc_{index}.npy"))
        iact_data = iact_timestep[:, 0] 
        timestep_data = iact_timestep[:, 1] 
        argsorted_data = np.argsort(timestep_data)
        timestep_argsorted = timestep_data[argsorted_data]
        iact_data = iact_data[argsorted_data] 
        storage_arr[:, index] = iact_data
    
    #print(storage_arr)
    min = 0.01
    fit_index = -1
    start_fit = -6

    sorted_timestep = np.trim_zeros(sorted_timestep, trim="f")
    iact_mean_arr = np.mean(storage_arr[np.nonzero(sorted_timestep >= min)], axis = 1)
    print(sorted_timestep)
    print(iact_mean_arr)

    err = np.std(storage_arr[np.nonzero(sorted_timestep >= min)], axis=1)
    sorted_N = 120 / sorted_timestep[np.nonzero(sorted_timestep >= min)]
    print(sorted_N)
    e_coeffs = np.polyfit(np.log(sorted_N[start_fit:]), np.log(iact_mean_arr[start_fit:]), deg=1)
    fitted_e = e_coeffs[1] + np.multiply(np.log(sorted_N[start_fit:]), e_coeffs[0])
    print(e_coeffs)

    fig, ax = plt.subplots(1, 1)
    ax.errorbar(sorted_N, iact_mean_arr, err, fmt='o', capsize=3, markersize=3.5, color="#ed1171ff")
    #ax.plot(sorted_N[start_fit:], np.exp(fitted_e), color="#d97dd9ff")

    ax.set_xlabel(r"$N_{\tau}$", fontsize=20, labelpad=-10)
    ax.set_ylabel("IACT", fontsize=15, labelpad=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    #print(ax.get_ylim())
    #ax.set_ylim(0, 1.5e1)
    #ax.set_xlim(38, 13000)
    plt.tight_layout()
    plt.savefig("iact_ff_b_01.pdf")
    plt.clf()



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])