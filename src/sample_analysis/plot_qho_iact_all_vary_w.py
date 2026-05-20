import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')

def main(metropolis_iact_data_path, ecmc_iact_data_path, factor_fields_iact_data_path, N, ff_N, propertime):

    N = int(N)
    ff_N = int(ff_N)
    propertime = float(propertime)
    try:
        metropolis_omega_data = np.load(os.path.join(metropolis_iact_data_path, "iact_metropolis_0_positions.npy"))[:, 1]
        metropolis_storage_arr = np.zeros((len(metropolis_omega_data), N))
        metropolis_sorted_omega = metropolis_omega_data[np.argsort(metropolis_omega_data)]
        for index in range(N):
            metropolis_iact_omega =  np.load(os.path.join(metropolis_iact_data_path, f"iact_metropolis_{index}_positions.npy"))
            metropolis_iact_data = metropolis_iact_omega[:, 0] 
            metropolis_omega_data = metropolis_iact_omega[:, 1] 
            argsorted_data = np.argsort(metropolis_omega_data)
            metropolis_iact_data = metropolis_iact_data[argsorted_data] 
            metropolis_storage_arr[:, index] = metropolis_iact_data

        metropolis_iact_mean_arr = np.mean(metropolis_storage_arr, axis = 1)
        #metropolis_iact_mean_arr = metropolis_iact_mean_arr[metropolis_sorted_omega]
        metropolis_err = np.std(metropolis_storage_arr, axis=1)
        #metropolis_err = metropolis_err[metropolis_sorted_omega]
        #metropolis_sorted_omega = metropolis_sorted_omega[metropolis_sorted_omega]
        #metropolis_sorted_N = propertime / metropolis_sorted_omega

    except:
        metrop_data = False

    ecmc_omega_data = np.load(os.path.join(ecmc_iact_data_path, "iact_ecmc_0_positions.npy"))[:, 1]
    ecmc_storage_arr = np.zeros((len(ecmc_omega_data), N))
    ecmc_sorted_omega = ecmc_omega_data[np.argsort(ecmc_omega_data)]

    for index in range(N):
        ecmc_iact_omega =  np.load(os.path.join(ecmc_iact_data_path, f"iact_ecmc_{index}_positions.npy"))
        ecmc_iact_data = ecmc_iact_omega[:, 0] 
        ecmc_omega_data = ecmc_iact_omega[:, 1] 
        argsorted_data = np.argsort(ecmc_omega_data)
        ecmc_omega_argsorted = ecmc_omega_data[argsorted_data]
        ecmc_iact_data = ecmc_iact_data[argsorted_data] 
        ecmc_storage_arr[:, index] = ecmc_iact_data

    ecmc_iact_mean_arr = np.mean(ecmc_storage_arr, axis = 1)
   # ecmc_iact_mean_arr = ecmc_iact_mean_arr[ecmc_sorted_omega]
    ecmc_err = np.std(ecmc_storage_arr, axis=1)
    #ecmc_err = ecmc_err[ecmc_sorted_omega]
    #ecmc_sorted_omega = ecmc_sorted_omega[ecmc_sorted_omega]
    #ecmc_sorted_N = propertime / ecmc_sorted_omega
    try:
        ff_omega_data = np.load(os.path.join(factor_fields_iact_data_path, "iact_ecmc_0_positions.npy"))[:, 1]
        ff_storage_arr = np.zeros((len(ff_omega_data), ff_N))
        ff_sorted_omega = ff_omega_data[np.argsort(ff_omega_data)]

        for index in range(ff_N):
            ff_iact_omega =  np.load(os.path.join(factor_fields_iact_data_path, f"iact_ecmc_{index}_positions.npy"))
            ff_iact_data = ff_iact_omega[:, 0] 
            ff_omega_data = ff_iact_omega[:, 1] 
            argsorted_data = np.argsort(ff_omega_data)
            ff_omega_argsorted = ff_omega_data[argsorted_data]
            ff_iact_data = ff_iact_data[argsorted_data] 
            ff_storage_arr[:, index] = ff_iact_data
        
        ff_iact_mean_arr = np.mean(ff_storage_arr, axis = 1)
        #ff_iact_mean_arr = ff_iact_mean_arr[ff_sorted_omega]
        ff_err = np.std(ff_storage_arr, axis=1)
        #ff_err = ff_err[ff_sorted_omega]
        #ff_sorted_omega = ff_sorted_omega[ff_sorted_omega]
        #ff_sorted_N = propertime / ff_sorted_omega
        ff_data = True

    except:
        ff_data = False

    m_fit_trim = -9#-7
    e_fit_trim = -13 #-13
    ff_fit_trim = -13


    fig, ax = plt.subplots(1, 1)
    ax.errorbar(ecmc_sorted_omega, ecmc_iact_mean_arr, ecmc_err, fmt='o', capsize=3, markersize=4, color="#e20acdff", label="ECMC")

    if metrop_data:
        ax.errorbar(metropolis_sorted_omega, metropolis_iact_mean_arr, metropolis_err, fmt='^', capsize=3, markersize=4, color="#e16f04ff", label="Metropolis MC")




    if ff_data:
        ax.errorbar(ff_sorted_omega, ff_iact_mean_arr, ff_err, fmt='o', capsize=3, markersize=4, color="#950834ff", label="ECMC with Factor Fields")
  
    



    
   
    ax.set_xlabel(r"$\omega ^2$", fontsize=20, labelpad=-10, weight = "bold")
    ax.set_ylabel("IACT", fontsize=15, weight = "bold")
    #ax.set_xscale("log")
    ax.set_yscale("log")
    #ax.set_ylim(10e-1, 40e2)
    legend_properties = {'weight':'bold'}
    plt.legend(prop=legend_properties)
    #ax.set_ylim(0.17e5, 0.8e7)
   


    plt.title(f"Integrated Autocorrelation Time for anharmonic oscillator with {N} Repeats")
    plt.tight_layout()
    plt.savefig("iact_anharmonic_vary_w_005.pdf")
    print(ecmc_sorted_omega)
    plt.clf()



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])