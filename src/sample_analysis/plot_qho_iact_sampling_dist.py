import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')

def main(l0_iact_data_path, l1_iact_data_path, l2_iact_data_path, N0, N1, N2, l0, l1, l2):

    N0 = int(N0)
    N1 = int(N1)
    N2 = int(N2)

    l0 = float(l0)
    l1 = float(l1)
    l2 = float(l2)

    l0_timestep_data = np.load(os.path.join(l0_iact_data_path, "iact_ecmc_0.npy"))[:, 1]
    l0_storage_arr = np.zeros((len(l0_timestep_data), N0))
    l0_sorted_timestep = l0_timestep_data[np.argsort(l0_timestep_data)]
    for index in range(N0):
        l0_iact_timestep =  np.load(os.path.join(l0_iact_data_path, f"iact_ecmc_{index}.npy"))
        l0_iact_data = l0_iact_timestep[:, 0] 
        l0_timestep_data = l0_iact_timestep[:, 1] 
        argsorted_data = np.argsort(l0_timestep_data)
        l0_iact_data = l0_iact_data[argsorted_data] 
        l0_storage_arr[:, index] = l0_iact_data

    l0_iact_mean_arr = np.mean(l0_storage_arr, axis = 1)
    l0_iact_mean_arr = l0_iact_mean_arr[l0_sorted_timestep >= 0.01]
    l0_err = np.std(l0_storage_arr, axis=1)
    l0_err = l0_err[l0_sorted_timestep >= 0.01]
    l0_sorted_timestep = l0_sorted_timestep[l0_sorted_timestep >= 0.01]
    l0_sorted_N = 120 / l0_sorted_timestep


    l1_timestep_data = np.load(os.path.join(l1_iact_data_path, "iact_ecmc_0.npy"))[:, 1]
    l1_storage_arr = np.zeros((len(l1_timestep_data), N1))
    l1_sorted_timestep = l1_timestep_data[np.argsort(l1_timestep_data)]

    for index in range(N1):
        l1_iact_timestep =  np.load(os.path.join(l1_iact_data_path, f"iact_ecmc_{index}.npy"))
        l1_iact_data = l1_iact_timestep[:, 0] 
        l1_timestep_data = l1_iact_timestep[:, 1] 
        argsorted_data = np.argsort(l1_timestep_data)
        l1_timestep_argsorted = l1_timestep_data[argsorted_data]
        l1_iact_data = l1_iact_data[argsorted_data] 
        l1_storage_arr[:, index] = l1_iact_data

    l1_iact_mean_arr = np.mean(l1_storage_arr, axis = 1)
    l1_iact_mean_arr = l1_iact_mean_arr[l1_sorted_timestep >= 0.01]
    l1_err = np.std(l1_storage_arr, axis=1)
    l1_err = l1_err[l1_sorted_timestep >= 0.01]
    l1_sorted_timestep = l1_sorted_timestep[l1_sorted_timestep >= 0.01]
    l1_sorted_N = 120 / l1_sorted_timestep
    try:
        l2_timestep_data = np.load(os.path.join(l2_iact_data_path, "iact_ecmc_0.npy"))[:, 1]
        l2_storage_arr = np.zeros((len(l2_timestep_data), N2))
        l2_sorted_timestep = l2_timestep_data[np.argsort(l2_timestep_data)]

        for index in range(N2):
            l2_iact_timestep =  np.load(os.path.join(l2_iact_data_path, f"iact_ecmc_{index}.npy"))
            l2_iact_data = l2_iact_timestep[:, 0] 
            l2_timestep_data = l2_iact_timestep[:, 1] 
            argsorted_data = np.argsort(l2_timestep_data)
            l2_timestep_argsorted = l2_timestep_data[argsorted_data]
            l2_iact_data = l2_iact_data[argsorted_data] 
            l2_storage_arr[:, index] = l2_iact_data
        
        l2_iact_mean_arr = np.mean(l2_storage_arr, axis = 1)
        l2_iact_mean_arr = l2_iact_mean_arr[l2_sorted_timestep >= 0.01]
        l2_err = np.std(l2_storage_arr, axis=1)
        l2_err = l2_err[l2_sorted_timestep >= 0.01]
        l2_sorted_timestep = l2_sorted_timestep[l2_sorted_timestep >= 0.01]
        l2_sorted_N = 120 / l2_sorted_timestep
        l2_data = True

    except:
        l2_data = False

    l0_fit_trim = -1
    l1_fit_trim = -1
    l2_fit_trim = -1
    l0_coeffs = np.polyfit(np.log(l0_sorted_N[:l0_fit_trim]), np.log(l0_iact_mean_arr[:l0_fit_trim]), deg=1)
    l1_coeffs = np.polyfit(np.log(l1_sorted_N[:l1_fit_trim]), np.log(l1_iact_mean_arr[:l1_fit_trim]), deg=1)


    fitted_l0 = l0_coeffs[1] + np.multiply(np.log(l0_sorted_N[:l0_fit_trim]), l0_coeffs[0])
    fitted_l1 = l1_coeffs[1] + np.multiply(np.log(l1_sorted_N[:l1_fit_trim]), l1_coeffs[0])


    print(l0_coeffs)
    print(l1_coeffs)

    fig, ax = plt.subplots(1, 1)

    #ax.plot(l0_sorted_N[:l0_fit_trim], np.exp(fitted_l0), color="#1b6b87ff")
    #ax.plot(l1_sorted_N[:l1_fit_trim], np.exp(fitted_l1), color="#a97ff1ff")

   
   
    l2_coeffs = np.polyfit(np.log(l2_sorted_N[:l2_fit_trim]), np.log(l2_iact_mean_arr[:l2_fit_trim]), deg=1)
    fitted_l2 = l2_coeffs[1] + np.multiply(np.log(l2_sorted_N[:l2_fit_trim]), l2_coeffs[0])
    print(l2_coeffs)
    #ax.plot(l2_sorted_N[:l2_fit_trim], np.exp(fitted_l2), color="#439946ff")
    ax.errorbar(l2_sorted_N, l2_iact_mean_arr, l2_err, fmt='o', capsize=3, markersize=4, color="#439946ff", label=f"l = {l2}")
    #ax.annotate(f"l = {l2} coeff: {l2_coeffs[0]:.2f}", xy = (4*10e2, 0.7*10e1))

    ax.errorbar(l0_sorted_N, l0_iact_mean_arr, l0_err, fmt='^', capsize=3, markersize=4, color="#1b6b87ff", label=f"l = {l0}")
    ax.errorbar(l1_sorted_N, l1_iact_mean_arr, l1_err, fmt='o', capsize=3, markersize=4, color="#a97ff1ff", label=f"l = {l1}")
    

    #ax.annotate(f"l = {l0} coeff: {l0_coeffs[0]:.2f}", xy = (8*10e1, 3*10e1))
    #ax.annotate(f"l = {l1} coeff: {l1_coeffs[0]:.2f}", xy = (4*10e2, 4*10e1))
    ecmc_check_timestep_data = np.load(os.path.join("output/re_run_qho_iact/iact/ecmc", "iact_ecmc_0.npy"))[:, 1]
    ecmc_check_storage_arr = np.zeros((len(ecmc_check_timestep_data), 100))
    ecmc_check_sorted_timestep = ecmc_check_timestep_data[np.argsort(ecmc_check_timestep_data)]
    for index in range(100):
        ecmc_check_iact_timestep =  np.load(os.path.join("output/re_run_qho_iact/iact/ecmc", f"iact_ecmc_{index}.npy"))
        ecmc_check_iact_data = ecmc_check_iact_timestep[:, 0] 
        ecmc_check_timestep_data = ecmc_check_iact_timestep[:, 1] 
        argsorted_data = np.argsort(ecmc_check_timestep_data)
        ecmc_check_timestep_argsorted = ecmc_check_timestep_data[argsorted_data]
        ecmc_check_iact_data = ecmc_check_iact_data[argsorted_data] 
        ecmc_check_storage_arr[:, index] = ecmc_check_iact_data

    ecmc_check_iact_mean_arr = np.mean(ecmc_check_storage_arr, axis = 1)
    ecmc_check_iact_mean_arr = ecmc_check_iact_mean_arr[ecmc_check_sorted_timestep >= 0.01]
    ecmc_check_err = np.std(ecmc_check_storage_arr, axis=1)
    ecmc_check_err = ecmc_check_err[ecmc_check_sorted_timestep >= 0.01]
    ecmc_check_sorted_timestep = ecmc_check_sorted_timestep[ecmc_check_sorted_timestep >= 0.01]
    ecmc_check_sorted_N = 120 / ecmc_check_sorted_timestep

    ax.errorbar(ecmc_check_sorted_N, ecmc_check_iact_mean_arr, ecmc_check_err, fmt='o', capsize=3, markersize=4, color="#e20acdff", label="ECMC")

   
    ax.set_xlabel(r"$N_{\tau}$", fontsize=20, labelpad=-10)
    ax.set_ylabel("IACT", fontsize=15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend()
    #ax.set_ylim(0.17e5, 0.8e7)
   


   # plt.title(f"Integrated Autocorrelation Time for ECMC, with {N} Repeats")
    plt.tight_layout()
    plt.savefig("iact_samp_dist.pdf")
    plt.clf()



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9])