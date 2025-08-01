import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'

def main(metropolis_iact_data_path, ecmc_iact_data_path, N):

    N = int(N)

    metropolis_timestep_data = np.load(os.path.join(metropolis_iact_data_path, "iact_metropolis_0.npy"))[:, 1]
    metropolis_storage_arr = np.zeros((len(metropolis_timestep_data), N))
    metropolis_sorted_timestep = metropolis_timestep_data[np.argsort(metropolis_timestep_data)]
    for index in range(N):
        metropolis_iact_timestep =  np.load(os.path.join(metropolis_iact_data_path, f"iact_metropolis_{index}.npy"))
        metropolis_iact_data = metropolis_iact_timestep[:, 0] 
        metropolis_timestep_data = metropolis_iact_timestep[:, 1] 
        argsorted_data = np.argsort(metropolis_timestep_data)
        metropolis_iact_data = metropolis_iact_data[argsorted_data] 
        metropolis_storage_arr[:, index] = metropolis_iact_data

    metropolis_iact_mean_arr = np.mean(metropolis_storage_arr, axis = 1)
    metropolis_iact_mean_arr = metropolis_iact_mean_arr[metropolis_sorted_timestep >= 0.05]
    metropolis_err = np.std(metropolis_storage_arr, axis=1)
    metropolis_err = metropolis_err[metropolis_sorted_timestep >= 0.05]
    metropolis_sorted_timestep = metropolis_sorted_timestep[metropolis_sorted_timestep >= 0.05]
    metropolis_sorted_N = 120 / metropolis_sorted_timestep


    ecmc_timestep_data = np.load(os.path.join(ecmc_iact_data_path, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr = np.zeros((len(ecmc_timestep_data), N))
    ecmc_sorted_timestep = ecmc_timestep_data[np.argsort(ecmc_timestep_data)]

    for index in range(N):
        ecmc_iact_timestep =  np.load(os.path.join(ecmc_iact_data_path, f"iact_ecmc_{index}.npy"))
        ecmc_iact_data = ecmc_iact_timestep[:, 0] 
        ecmc_timestep_data = ecmc_iact_timestep[:, 1] 
        argsorted_data = np.argsort(ecmc_timestep_data)
        ecmc_timestep_argsorted = ecmc_timestep_data[argsorted_data]
        ecmc_iact_data = ecmc_iact_data[argsorted_data] 
        ecmc_storage_arr[:, index] = ecmc_iact_data

    ecmc_iact_mean_arr = np.mean(ecmc_storage_arr, axis = 1)
    ecmc_iact_mean_arr = ecmc_iact_mean_arr[ecmc_sorted_timestep >= 0.01]
    ecmc_err = np.std(ecmc_storage_arr, axis=1)
    ecmc_err = ecmc_err[ecmc_sorted_timestep >= 0.01]
    ecmc_sorted_timestep = ecmc_sorted_timestep[ecmc_sorted_timestep >= 0.01]
    ecmc_sorted_N = 120 / ecmc_sorted_timestep

    m_coeffs = np.polyfit(np.log(metropolis_sorted_N[:-7]), np.log(metropolis_iact_mean_arr[:-7]), deg=1)
    e_coeffs = np.polyfit(np.log(ecmc_sorted_N[:-13]), np.log(ecmc_iact_mean_arr[:-13]), deg=1)

    fitted_m = m_coeffs[1] + np.multiply(np.log(metropolis_sorted_N[:-7]), m_coeffs[0])
    fitted_e = e_coeffs[1] + np.multiply(np.log(ecmc_sorted_N[:-13]), e_coeffs[0])

    print(m_coeffs)
    print(e_coeffs)

    fig, ax = plt.subplots(1, 1)
    ax.plot(metropolis_sorted_N[:-7], np.exp(fitted_m), color="#f9a37bff")
    ax.plot(ecmc_sorted_N[:-13], np.exp(fitted_e), color="#d97dd9ff")
    ax.errorbar(metropolis_sorted_N, metropolis_iact_mean_arr, metropolis_err, fmt='^', capsize=3, markersize=4, color="#e16f04ff", label="Metropolis MC")
    ax.errorbar(ecmc_sorted_N, ecmc_iact_mean_arr, ecmc_err, fmt='o', capsize=3, markersize=4, color="#e20acdff", label="ECMC")
    
   
    ax.set_xlabel(r"$N_{\tau}$", fontsize=20, labelpad=-10)
    ax.set_ylabel("IACT", fontsize=15, labelpad=-15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend()
    ax.set_ylim(0.17e5, 0.8e7)
   


   # plt.title(f"Integrated Autocorrelation Time for ECMC, with {N} Repeats")
    plt.tight_layout()
    plt.savefig("iact.pdf")
    plt.clf()



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])