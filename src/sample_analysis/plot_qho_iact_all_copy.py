import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')

def main(metropolis_iact_x0_2_data_path, metropolis_iact_x0_0_data_path, metropolis_iact_x0_10_data_path, ecmc_iact_x0_2_data_path, ecmc_iact_x0_0_data_path, ecmc_iact_x0_10_data_path, N, propertime):

    N = int(N)
    propertime = float(propertime)


    try:

        metropolis_timestep_data_x0_2 = np.load(os.path.join(metropolis_iact_x0_2_data_path, "iact_metropolis_0.npy"))[:, 1]
        metropolis_storage_arr_x0_2 = np.zeros((len(metropolis_timestep_data_x0_2), N))
        metropolis_sorted_timestep_x0_2 = metropolis_timestep_data_x0_2[np.argsort(metropolis_timestep_data_x0_2)]
        for index in range(N):
            metropolis_iact_timestep_x0_2 =  np.load(os.path.join(metropolis_iact_x0_2_data_path, f"iact_metropolis_{index}.npy"))
            metropolis_iact_data_x0_2 = metropolis_iact_timestep_x0_2[:, 0] 
            metropolis_timestep_data_x0_2 = metropolis_iact_timestep_x0_2[:, 1] 
            argsorted_data_x0_2 = np.argsort(metropolis_timestep_data_x0_2)
            metropolis_iact_data_x0_2 = metropolis_iact_data_x0_2[argsorted_data_x0_2] 
            metropolis_storage_arr_x0_2[:, index] = metropolis_iact_data_x0_2

        metropolis_iact_mean_arr_x0_2 = np.mean(metropolis_storage_arr_x0_2, axis = 1)
        metropolis_iact_mean_arr_x0_2 = metropolis_iact_mean_arr_x0_2[metropolis_sorted_timestep_x0_2 >= 0.01]
        metropolis_err_x0_2 = np.std(metropolis_storage_arr_x0_2, axis=1)
        metropolis_err_x0_2 = metropolis_err_x0_2[metropolis_sorted_timestep_x0_2 >= 0.01]
        metropolis_sorted_timestep_x0_2 = metropolis_sorted_timestep_x0_2[metropolis_sorted_timestep_x0_2 >= 0.01]
        metropolis_sorted_N_x0_2 = propertime / metropolis_sorted_timestep_x0_2

        metrop_data = True
    except:
        metrop_data = False
        print("metrop false x0=2")

    

    metropolis_timestep_data_x0_0 = np.load(os.path.join(metropolis_iact_x0_0_data_path, "iact_metropolis_0.npy"))[:, 1]
    metropolis_storage_arr_x0_0 = np.zeros((len(metropolis_timestep_data_x0_0), N))
    metropolis_sorted_timestep_x0_0 = metropolis_timestep_data_x0_0[np.argsort(metropolis_timestep_data_x0_0)]
    for index in range(N):
        metropolis_iact_timestep_x0_0 =  np.load(os.path.join(metropolis_iact_x0_0_data_path, f"iact_metropolis_{index}.npy"))
        metropolis_iact_data_x0_0 = metropolis_iact_timestep_x0_0[:, 0] 
        metropolis_timestep_data_x0_0 = metropolis_iact_timestep_x0_0[:, 1] 
        argsorted_data_x0_0 = np.argsort(metropolis_timestep_data_x0_0)
        metropolis_iact_data_x0_0 = metropolis_iact_data_x0_0[argsorted_data_x0_0] 
        metropolis_storage_arr_x0_0[:, index] = metropolis_iact_data_x0_0

    metropolis_iact_mean_arr_x0_0 = np.mean(metropolis_storage_arr_x0_0, axis = 1)
    metropolis_iact_mean_arr_x0_0 = metropolis_iact_mean_arr_x0_0[metropolis_sorted_timestep_x0_0 >= 0.01]
    metropolis_err_x0_0 = np.std(metropolis_storage_arr_x0_0, axis=1)
    metropolis_err_x0_0 = metropolis_err_x0_0[metropolis_sorted_timestep_x0_0 >= 0.01]
    metropolis_sorted_timestep_x0_0 = metropolis_sorted_timestep_x0_0[metropolis_sorted_timestep_x0_0 >= 0.01]
    metropolis_sorted_N_x0_0 = propertime / metropolis_sorted_timestep_x0_0

 
    
    metropolis_timestep_data_x0_10 = np.load(os.path.join(metropolis_iact_x0_10_data_path, "iact_metropolis_0.npy"))[:, 1]
    metropolis_storage_arr_x0_10 = np.zeros((len(metropolis_timestep_data_x0_10), N))
    metropolis_sorted_timestep_x0_10 = metropolis_timestep_data_x0_10[np.argsort(metropolis_timestep_data_x0_10)]
    for index in range(N):
        metropolis_iact_timestep_x0_10 =  np.load(os.path.join(metropolis_iact_x0_10_data_path, f"iact_metropolis_{index}.npy"))
        metropolis_iact_data_x0_10 = metropolis_iact_timestep_x0_10[:, 0] 
        metropolis_timestep_data_x0_10 = metropolis_iact_timestep_x0_10[:, 1] 
        argsorted_data_x0_10 = np.argsort(metropolis_timestep_data_x0_10)
        metropolis_iact_data_x0_10 = metropolis_iact_data_x0_10[argsorted_data_x0_10] 
        metropolis_storage_arr_x0_10[:, index] = metropolis_iact_data_x0_10

    metropolis_iact_mean_arr_x0_10 = np.mean(metropolis_storage_arr_x0_10, axis = 1)
    metropolis_iact_mean_arr_x0_10 = metropolis_iact_mean_arr_x0_10[metropolis_sorted_timestep_x0_10 >= 0.01]
    metropolis_err_x0_10 = np.std(metropolis_storage_arr_x0_10, axis=1)
    metropolis_err_x0_10 = metropolis_err_x0_10[metropolis_sorted_timestep_x0_10 >= 0.01]
    metropolis_sorted_timestep_x0_10 = metropolis_sorted_timestep_x0_10[metropolis_sorted_timestep_x0_10 >= 0.01]
    metropolis_sorted_N_x0_10 = propertime / metropolis_sorted_timestep_x0_10




    ecmc_timestep_data_x0_2 = np.load(os.path.join(ecmc_iact_x0_2_data_path, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_x0_2 = np.zeros((len(ecmc_timestep_data_x0_2), N))
    ecmc_sorted_timestep_x0_2 = ecmc_timestep_data_x0_2[np.argsort(ecmc_timestep_data_x0_2)]

    for index in range(N):
        ecmc_iact_timestep_x0_2 =  np.load(os.path.join(ecmc_iact_x0_2_data_path, f"iact_ecmc_{index}.npy"))
        ecmc_iact_data_x0_2 = ecmc_iact_timestep_x0_2[:, 0] 
        ecmc_timestep_data_x0_2 = ecmc_iact_timestep_x0_2[:, 1] 
        argsorted_data_x0_2 = np.argsort(ecmc_timestep_data_x0_2)
        ecmc_timestep_argsorted_x0_2 = ecmc_timestep_data_x0_2[argsorted_data_x0_2]
        ecmc_iact_data_x0_2 = ecmc_iact_data_x0_2[argsorted_data_x0_2] 
        ecmc_storage_arr_x0_2[:, index] = ecmc_iact_data_x0_2

    ecmc_iact_mean_arr_x0_2 = np.mean(ecmc_storage_arr_x0_2, axis = 1)
    ecmc_iact_mean_arr_x0_2 = ecmc_iact_mean_arr_x0_2[ecmc_sorted_timestep_x0_2 >= 0.01]
    ecmc_err_x0_2 = np.std(ecmc_storage_arr_x0_2, axis=1)
    ecmc_err_x0_2 = ecmc_err_x0_2[ecmc_sorted_timestep_x0_2 >= 0.01]
    ecmc_sorted_timestep_x0_2 = ecmc_sorted_timestep_x0_2[ecmc_sorted_timestep_x0_2 >= 0.01]
    ecmc_sorted_N_x0_2 = propertime / ecmc_sorted_timestep_x0_2


    ecmc_timestep_data_x0_0 = np.load(os.path.join(ecmc_iact_x0_0_data_path, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_x0_0 = np.zeros((len(ecmc_timestep_data_x0_0), N))
    ecmc_sorted_timestep_x0_0 = ecmc_timestep_data_x0_0[np.argsort(ecmc_timestep_data_x0_0)]

    for index in range(N):
        ecmc_iact_timestep_x0_0 =  np.load(os.path.join(ecmc_iact_x0_0_data_path, f"iact_ecmc_{index}.npy"))
        ecmc_iact_data_x0_0 = ecmc_iact_timestep_x0_0[:, 0] 
        ecmc_timestep_data_x0_0 = ecmc_iact_timestep_x0_0[:, 1] 
        argsorted_data_x0_0 = np.argsort(ecmc_timestep_data_x0_0)
        ecmc_timestep_argsorted_x0_0 = ecmc_timestep_data_x0_0[argsorted_data_x0_0]
        ecmc_iact_data_x0_0 = ecmc_iact_data_x0_0[argsorted_data_x0_0] 
        ecmc_storage_arr_x0_0[:, index] = ecmc_iact_data_x0_0

    ecmc_iact_mean_arr_x0_0 = np.mean(ecmc_storage_arr_x0_0, axis = 1)
    ecmc_iact_mean_arr_x0_0 = ecmc_iact_mean_arr_x0_0[ecmc_sorted_timestep_x0_0 >= 0.01]
    ecmc_err_x0_0 = np.std(ecmc_storage_arr_x0_0, axis=1)
    ecmc_err_x0_0 = ecmc_err_x0_0[ecmc_sorted_timestep_x0_0 >= 0.01]
    ecmc_sorted_timestep_x0_0 = ecmc_sorted_timestep_x0_0[ecmc_sorted_timestep_x0_0 >= 0.01]
    ecmc_sorted_N_x0_0 = propertime / ecmc_sorted_timestep_x0_0



    ecmc_timestep_data_x0_10 = np.load(os.path.join(ecmc_iact_x0_10_data_path, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_x0_10 = np.zeros((len(ecmc_timestep_data_x0_10), N))
    ecmc_sorted_timestep_x0_10 = ecmc_timestep_data_x0_10[np.argsort(ecmc_timestep_data_x0_10)]

    for index in range(N):
        ecmc_iact_timestep_x0_10 =  np.load(os.path.join(ecmc_iact_x0_10_data_path, f"iact_ecmc_{index}.npy"))
        ecmc_iact_data_x0_10 = ecmc_iact_timestep_x0_10[:, 0] 
        ecmc_timestep_data_x0_10 = ecmc_iact_timestep_x0_10[:, 1] 
        argsorted_data_x0_10 = np.argsort(ecmc_timestep_data_x0_10)
        ecmc_timestep_argsorted_x0_10 = ecmc_timestep_data_x0_10[argsorted_data_x0_10]
        ecmc_iact_data_x0_10 = ecmc_iact_data_x0_10[argsorted_data_x0_10] 
        ecmc_storage_arr_x0_10[:, index] = ecmc_iact_data_x0_10

    ecmc_iact_mean_arr_x0_10 = np.mean(ecmc_storage_arr_x0_10, axis = 1)
    ecmc_iact_mean_arr_x0_10 = ecmc_iact_mean_arr_x0_10[ecmc_sorted_timestep_x0_10 >= 0.01]
    ecmc_err_x0_10 = np.std(ecmc_storage_arr_x0_10, axis=1)
    ecmc_err_x0_10 = ecmc_err_x0_10[ecmc_sorted_timestep_x0_10 >= 0.01]
    ecmc_sorted_timestep_x0_10 = ecmc_sorted_timestep_x0_10[ecmc_sorted_timestep_x0_10 >= 0.01]
    ecmc_sorted_N_x0_10 = propertime / ecmc_sorted_timestep_x0_10


    fig, ax = plt.subplots(1, 1)


    ax.errorbar(metropolis_sorted_N_x0_0, metropolis_iact_mean_arr_x0_0, metropolis_err_x0_0, fmt='^', capsize=3, markersize=10, color="#1500ffff", label="Metropolis MC, x0 = 0.0", alpha = 1.0)
    ax.errorbar(metropolis_sorted_N_x0_2, metropolis_iact_mean_arr_x0_2, metropolis_err_x0_2, fmt='p', capsize=3, markersize=10, color="#000000ff", label="Metropolis MC, x0 = 2.0", alpha = 1.0)
    ax.errorbar(metropolis_sorted_N_x0_10, metropolis_iact_mean_arr_x0_10, metropolis_err_x0_10, fmt='x', capsize=3, markersize=4, color="#ff7700ff", label="Metropolis MC, x0 = 10.0", alpha = 1.0)


    ax.errorbar(ecmc_sorted_N_x0_0, ecmc_iact_mean_arr_x0_0, ecmc_err_x0_0, fmt='o', capsize=3, markersize=15, color="#ffee00ff", label="ECMC, x0 = 0.0", alpha = 0.5)
    ax.errorbar(ecmc_sorted_N_x0_2, ecmc_iact_mean_arr_x0_2, ecmc_err_x0_2, fmt='*', capsize=3, markersize=10, color="#ff0000ff", label="ECMC, x0 = 2.0", alpha = 0.5)
    ax.errorbar(ecmc_sorted_N_x0_10, ecmc_iact_mean_arr_x0_10, ecmc_err_x0_10, fmt='s', capsize=3, markersize=4, color="#60219cff", label="ECMC, x0 = 10.0", alpha = 1.0)


    ax.set_xlabel(r"$N_{\tau}$", fontsize=20, labelpad=-10, weight = "bold")
    ax.set_ylabel("IACT", fontsize=15, weight = "bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    legend_properties = {'weight':'bold'}
    plt.legend(prop=legend_properties)
  
    plt.tight_layout()
    plt.savefig("iact_vary_x0.png")
    plt.clf()



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])