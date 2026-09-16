import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')

def main(ecmc_rmsd_data_path, N, propertime):

    N = int(N)
    propertime = float(propertime)

    ecmc_timestep_data = np.load(os.path.join(ecmc_rmsd_data_path, "rmsd_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr = np.zeros((len(ecmc_timestep_data), N))
    ecmc_er_storage_arr = np.zeros((len(ecmc_timestep_data), N))
    ecmc_sorted_timestep = ecmc_timestep_data[np.argsort(ecmc_timestep_data)]

    for index in range(N):
        ecmc_rmsd_timestep =  np.load(os.path.join(ecmc_rmsd_data_path, f"rmsd_ecmc_{index}.npy"))
        ecmc_rmsd_data = ecmc_rmsd_timestep[:, 0] 
        ecmc_timestep_data = ecmc_rmsd_timestep[:, 1]
        ecmc_event_rate_data = ecmc_rmsd_timestep[:, 2]
        argsorted_data = np.argsort(ecmc_timestep_data)
        ecmc_timestep_argsorted = ecmc_timestep_data[argsorted_data]
        ecmc_rmsd_data = ecmc_rmsd_data[argsorted_data] 
        ecmc_storage_arr[:, index] = ecmc_rmsd_data
        ecmc_event_rate_data = ecmc_event_rate_data[argsorted_data]
        ecmc_er_storage_arr[:, index] = ecmc_event_rate_data

    ecmc_rmsd_mean_arr = np.mean(ecmc_storage_arr, axis = 1)
    ecmc_rmsd_mean_arr = ecmc_rmsd_mean_arr[ecmc_sorted_timestep >= 0.01]
    ecmc_err = np.std(ecmc_storage_arr, axis=1)
    ecmc_err = ecmc_err[ecmc_sorted_timestep >= 0.01]
    ecmc_event_rate_mean_arr = np.mean(ecmc_er_storage_arr, axis=1)
    ecmc_event_rate_mean_arr = ecmc_event_rate_mean_arr[ecmc_sorted_timestep >= 0.01]
    ecmc_er_err = np.std(ecmc_er_storage_arr, axis=1)
    ecmc_er_err = ecmc_er_err[ecmc_sorted_timestep >= 0.01]
    
    ecmc_sorted_timestep = ecmc_sorted_timestep[ecmc_sorted_timestep >= 0.01]
    ecmc_sorted_N = propertime / ecmc_sorted_timestep

    print(ecmc_sorted_N)
    print(ecmc_event_rate_mean_arr)


    rmsd_fit_trim = -1
    rmsd_coeffs = np.polyfit(np.log(ecmc_sorted_N[:rmsd_fit_trim]), np.log(ecmc_rmsd_mean_arr[:rmsd_fit_trim]), deg=1)
    fitted_rmsd = rmsd_coeffs[1] + np.multiply(np.log(ecmc_sorted_N[:rmsd_fit_trim]), rmsd_coeffs[0])
    print(f"rmsd scaling: {rmsd_coeffs[0]}")

    
    er_fit_trim = -3
    er_coeffs = np.polyfit(np.log(ecmc_sorted_N[:er_fit_trim]), np.log(ecmc_event_rate_mean_arr[:er_fit_trim]), deg=1)
    fitted_er = er_coeffs[1] + np.multiply(np.log(ecmc_sorted_N[:er_fit_trim]), er_coeffs[0])
    print(f"event rate scaling: {er_coeffs[0]}, intercept: {er_coeffs[1]}")

    fig, ax = plt.subplots(1, 1)

    ax.plot(ecmc_sorted_N[:rmsd_fit_trim], np.exp(fitted_rmsd), color="#d97dd9ff")
    ax.errorbar(ecmc_sorted_N, ecmc_rmsd_mean_arr, ecmc_err, fmt='o', capsize=3, markersize=4, color="#e20acdff", label="RMSD")
    


    ax.set_xlabel(r"$N_{\tau}$", fontsize=20, labelpad=-10, weight = "bold")
    ax.set_ylabel("RMSD", fontsize=15, weight = "bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    legend_properties = {'weight':'bold'}
    plt.legend(prop=legend_properties)
    #ax.set_ylim(0.17e5, 0.8e7)
   


    #plt.title(f"IACT of x^2 for QHO with (x+2)^2")
    plt.tight_layout()
    plt.savefig("rmsd_ecmc_symm.png")#, transparent=True)
    plt.clf()

    fig, ax = plt.subplots(1, 1)

    ax.plot(ecmc_sorted_N[:er_fit_trim], np.exp(fitted_er), color="#1a97c4ff")
    ax.errorbar(ecmc_sorted_N, ecmc_event_rate_mean_arr, ecmc_er_err, fmt='o', capsize=3, markersize=4, color="#470ae2ff", label="Event rate")
    


    ax.set_xlabel(r"$N_{\tau}$", fontsize=20, labelpad=-10, weight = "bold")
    ax.set_ylabel("Event rate", fontsize=15, weight = "bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    legend_properties = {'weight':'bold'}
    plt.legend(prop=legend_properties)
    #ax.set_ylim(0.17e5, 0.8e7)
   


    #plt.title(f"IACT of x^2 for QHO with (x+2)^2")
    plt.tight_layout()
    plt.savefig("er_ecmc_symm.png")#, transparent=True)
    plt.clf()



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])